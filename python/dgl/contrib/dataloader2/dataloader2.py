#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections.abc import Sequence
from queue import Queue
import threading
import torch
import torch.multiprocessing as mp
from ...dataloading import create_tensorized_dataset
from ...base import NID, EID, DGLError
from ...cuda import stream as dgl_stream

from torch.cuda import nvtx

NODES_TAG = "nodes"
INPUT_NODES_TAG = NODES_TAG + ":input"
OUTPUT_NODES_TAG = NODES_TAG + ":output"
EDGES_TAG = "edges"


class _LoaderInstance:
    def __init__(self, graph_source, feature_source, output_device):
        self._graph_source = graph_source
        self._feature_source = feature_source
        self._output_device = output_device
        if self._feature_source:
            self._featured_entities = self._feature_source.get_featured_entities()
        else:
            self._featured_entities = None

    def load(self, ids):
        # these graphs will have ids attached to nodes and edges, and may
        # have additional features as well
        graphs = self._graph_source.fetch_graph(ids, self._output_device)

        if self._feature_source:
            # TODO(@nv-dlasalle): Optimizations to consider:
            # - Merging requests into a single dictionary to require a single
            # message (but we need to figure out how to handle duplicate
            # node types in DGLBlocks.
            # - Uniquifying ids before sending them, this increases our
            # computation to reduce communication. This will only be helpful
            # because we have duplicate IDs between layers, or the sampling
            # method does not uniqify the src nodes.
            req = {}

            # we don't fetch features positive and negative graphs in link
            # prediciton, since we only care about the edges themselves
            if isinstance(graphs[-1], Sequence):
                blocks = graphs[-1]
            else:
                blocks = graphs

            for layer, graph in enumerate(blocks):
                if graph.is_block:
                    # dgl blocks treat src types and dst types as distinct
                    for ntype in graph.srctypes:
                        if NODES_TAG in self._featured_entities and \
                                ntype in self._featured_entities[NODES_TAG]:
                            key = NODES_TAG
                        elif layer == 0 and INPUT_NODES_TAG in self._featured_entities and \
                                ntype in self._featured_entities[INPUT_NODES_TAG]:
                            key = INPUT_NODES_TAG
                        else:
                            continue

                        node_ids = graph.srcdata[NID]
                        resp = self._feature_source.fetch_features(
                                {key: {ntype: node_ids}}, self._output_device)
                        for feat_name, tensor in resp[key][ntype].items():
                            graph.srcnodes[ntype].data[feat_name] = tensor
                    for ntype in graph.dsttypes:
                        if NODES_TAG in self._featured_entities and \
                                ntype in self._featured_entities[NODES_TAG]:
                            key = NODES_TAG
                        elif layer == len(graphs) - 1 and \
                                OUTPUT_NODES_TAG in self._featured_entities and \
                                ntype in self._featured_entities[OUTPUT_NODES_TAG]:
                            key = OUTPUT_NODES_TAG
                        else:
                            continue

                        node_ids = graph.dstdata[NID]
                        resp = self._feature_source.fetch_features(
                                {key: {ntype: node_ids}}, self._output_device)
                        for feat_name, tensor in resp[key][ntype].items():
                            graph.dstnodes[ntype].data[feat_name] = tensor
                else:
                    if NODES_TAG in self._featured_entities:
                        for ntype in graph.ntypes:
                            if ntype in self._featured_entities[NODES_TAG]:
                                node_ids = graph.ndata[NID]
                                resp = self._feature_source.fetch_features(
                                        {NODES_TAG: {ntype: node_ids}}, self._output_device)
                                for feat_name, tensor in resp[NODES_TAG][ntype].items():
                                    graph.nodes[ntype].data[feat_name] = tensor
                if EDGES_TAG in self._featured_entities:
                    for etype in graph.canonical_etypes:
                        if etype in self._featured_entities[EDGES_TAG]:
                            edge_ids = graph.edges[etype].data[EID]
                            resp = self._feature_source.fetch_features(
                                    {EDGES_TAG: {etype: edge_ids}}, self._output_device)
                            for feat_name, tensor in resp[EDGES_TAG][etype].items():
                                graph.edges[etype].data[feat_name] = tensor

        return graphs


def _worker_set_num_threads(worker_id):
    torch.set_num_threads(1)

class _ThreadedIter:
    def __init__(self, iterator, collate_fn, num_threads, stream, prefetch_factor):
        self._iter = iterator
        self._collate_fn = collate_fn
        self._stream = stream
        max_buffered = 1 + (max(0, prefetch_factor-1) * num_threads)
        self._queue = Queue(maxsize=max_buffered)
        self._done_threads = 0

        nvtx.range_push("start_threads")
        self._threads = [
            threading.Thread(target=self._thread_work,
                             daemon=True)
            for _ in range(num_threads)
        ]
        for t in self._threads:
            t.start()
        nvtx.range_pop()


    def _thread_work(self):
        try:
            while True:
                with torch.cuda.stream(self._stream):
                    with dgl_stream(self._stream):
                        task = next(self._iter)
                        nvtx.range_push("thread_collate")
                        batch = self._collate_fn(task)
                        nvtx.range_pop()
                        if self._stream is not None:
                            # make this stream waits for the work to finish
                            torch.cuda.default_stream().wait_stream(self._stream)
                        self._queue.put(batch)
        except StopIteration:
            # signal this thread is done
            self._queue.put(StopIteration)


    def __next__(self):
        n = self._queue.get()
        while n is StopIteration:
            self._done_threads += 1
            if self._done_threads == len(self._threads):
                raise StopIteration
            n = self._queue.get()
        return n


class _ThreadedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, collate_fn, num_threads, stream=None, \
            prefetch_factor=2):
        self._dataset = dataset
        self._collate_fn = collate_fn
        self._num_threads = num_threads
        self._prefetch_factor = prefetch_factor
        self._stream = stream

        # use methods from wrapped dataset
        self.shuffle = dataset.shuffle


    def __len__(self):
        return len(self._dataset)


    def __iter__(self):
        return _ThreadedIter(iterator=iter(self._dataset), \
                             collate_fn=self._collate_fn, \
                             num_threads=self._num_threads, \
                             stream=self._stream, \
                             prefetch_factor=self._prefetch_factor)


class DataLoader:
    def __init__(self, graph_source, feature_source, ids=None,
                 output_device=torch.device('cpu'),
                 batch_size=1, drop_last=False, shuffle=False, num_workers=0,
                 use_thread_workers=False,
                 use_ddp=False, ddp_seed=0, persistent_workers=False,
                 prefetch_factor=2):
        """ DataLoader for generating mini-batches of graph data.

        Parameters
        ----------
        graph_source : GraphSource-like object
            This object should have at least a `fetch_graphs()` method. The
            returned graphs can have features attached, or have node and edge
            IDs mapping features in `feature_source` object. This object can
            "pre-batch" items, in which case the input indices correspond to
            mini-batches rather than items within a mini-batch.
            See `graph_source.py` for more details.

        feature_source : FeatureSource-like object or None
            If provided, this object is queried for features corresponding to
            items in the mini-batch graphs and attached.
            If this parameter is None, no additional features are attached to
            the graphs returned by the `graph_source`.

        ids : Tensor or None
            The indices to load. If not provided, the `graph_source` object
            must implement the `__len__` method, and all indices that are
            loaded.
        """
        self._batch_loader = _LoaderInstance( \
            graph_source=graph_source, \
            feature_source=feature_source, \
            output_device=output_device)

        if ids is None:
            if not hasattr(graph_source, '__len__'):
                raise ValueError("If not 'ids' is not specified, the " \
                    "graph_source object must have a __len__ method to " \
                    "auto generate the ids.")
            # generate on the default device
            ids = torch.arange(len(graph_source))

        # the TensorizedDataset only takes in shuffle for the purpose of
        # outputting warnings, so we need to manually tell it to shuffle
        # each iteration
        self._shuffle = shuffle
        self._dataset = dataset=create_tensorized_dataset(
                indices=ids,
                batch_size=batch_size,
                drop_last=drop_last,
                use_ddp=use_ddp,
                ddp_seed=ddp_seed,
                shuffle=shuffle)

        # these parameters can only be set in the dataloader when
        # num_workers > 0
        dataloader_kwargs = {}
        collate_fn = self._batch_loader.load
        if num_workers > 0:
            if use_thread_workers:
                self._dataset = _ThreadedDataset(dataset, collate_fn, \
                    num_threads=num_workers, \
                    stream=torch.cuda.Stream(output_device), \
                    prefetch_factor=prefetch_factor)
                # setup dataloader arguments for manual collation and no
                # workers
                collate_fn = None
                num_workers = 0
            else:
                if output_device.type != 'cpu':
                    raise DGLError("Forking workers is not supported when output "
                        "is a CUDA context.")
                dataloader_kwargs['persistent_workers'] = persistent_workers
                dataloader_kwargs['prefetch_factor'] = prefetch_factor
        else:
            if use_thread_workers:
                raise DGLError("num_workers must be greater than o when " \
                    "use_thread_workers true.")

        # use pytorch's dataloader to handle workers
        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            shuffle=False,
            drop_last=False,
            batch_size=None,
            collate_fn=collate_fn,
            num_workers=num_workers,
            worker_init_fn=_worker_set_num_threads,
            **dataloader_kwargs)

    def __iter__(self):
        if self._shuffle:
            self._dataset.shuffle()
        return iter(self._dataloader)
