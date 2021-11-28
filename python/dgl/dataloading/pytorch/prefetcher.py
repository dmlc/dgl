from queue import Queue
import threading
import torch
from ..._ffi import streams as FS
from ...utils import recursive_apply, ExceptionWrapper
from ...base import NID, EID

def _worker_entry(iter_, queue, device, get_features_func):
    stream = torch.cuda.Stream(device=device)
    try:
        while True:
            result = get_features_func(next(iter_))
            with torch.cuda.stream(stream), FS.stream(stream):
                queue.put((
                    recursive_apply(result, lambda x: x.to(device)),
                    stream.record_event(),
                    None))
    except StopIteration:
        queue.put((None, None, None))
    except:
        queue.put((None, None, ExceptionWrapper(where='in CUDA async feature copy')))


def _slice_one_type(indices, features, pin_memory):
    """
    Parameters
    ----------
    indices : Tensor
    features : dict[str, Tensor]
    pin_memory : bool

    Returns
    -------
    dict[str, Tensor]
    """
    if len(features) == 0:
        return {}

    out_tensors = {}
    for k, v in features.items():
        out_tensor = torch.empty(
            indices.shape[0], *v.shape[1:], dtype=v.dtype, pin_memory=pin_memory)
        torch.index_select(v, 0, indices, out=out_tensor)
        out_tensors[k] = out_tensor
    return out_tensors


def _slice(features, indices, pin_memory=True):
    """
    Parameters
    ----------
    features : dict[str, Tensor] or dict[any, dict[str, Tensor]]
    indices : Tensor or dict[any, Tensor]
    pin_memory : bool

    Returns
    -------
    dict[str, Tensor] or dict[any, dict[str, Tensor]]
    """
    if torch.is_tensor(indices):
        return _slice_one_type(indices, features, pin_memory)
    print(indices, features)
    return {k: _slice_one_type(indices[k], features[k], pin_memory)
            for k in indices.keys() if k in features.keys()}

class CUDAAsyncCopyWrapper(object):
    """Wrapper of an iterator that allows asynchronous prefetching of features from CPU
    to GPU.

    The asynchronous copy works as follows:

    * NodeDataLoader's worker process retrieves sampled blocks/subgraphs together
      with some other information such as input/output node/edge IDs.
    * The wrapper's worker thread invokes :meth:`get_features` method, consuming
      the sampler's output and returning it together with the retrieved necessary features.
    * The wrapper's worker thread puts the result into a queue.
    * The main process retrieves the features and the sampled blocks/subgraphs from the queue.
    * The main process assigns the features to the block/subgraphs with
      :meth:`assign_features` method and returns it.
    """
    def __init__(self, iter_, device):
        # The queue contains triplets of
        # (1) The graph, with features sliced.
        self.queue = Queue(1)
        self.thread = threading.Thread(
                target=_worker_entry,
                args=(iter_, self.queue, device, self.get_features),
                daemon=True)
        self.device = device
        self.thread.start()

    def get_features(self, sample_result):
        """Gets the feature from CPU.
        """
        raise NotImplementedError

    def assign_features(self, queue_result):
        raise NotImplementedError

    def __next__(self):
        result, event, exception = self.queue.get()
        if result is None:
            if exception is None:
                raise StopIteration
            else:
                exception.reraise()
        event.wait(torch.cuda.default_stream())
        return self.assign_features(result)

def _update_data(dataview, view, data, types):
    if len(types) == 1:
        dataview.update(data)
    else:
        for k, v in data.items():
            view[k].data.update(v)


class CUDAAsyncCopyNodeDataLoaderWrapper(CUDAAsyncCopyWrapper):
    """
    Parameters
    ----------
    input_features : dict
        Input feature dictionary which can be either a dictionary of feature names and
        feature tensors, or a dictionary of node types and the above dictionaries.

        The features copied from this dictionary will be assigned to ``srcdata`` of
        the first sampled block.
    output_labels : dict
        Output label dictionary which can be either a dictionary of feature names and
        feature tensors, or a dictionary of node types and the above dictionaries.

        The features copied from this dictionary will be assigned to ``dstdata`` of
        the last sampled block.
    node_features : dict
        Node data dictionary which can be either a dictionary of feature names and
        feature tensors, or a dictionary of node types and the above dictionaries.

        The features copied from this dictionary will be assigned to ``srcdata``
        and ``dstdata`` of all blocks.
    edge_features : dict
        Edge data dictionary which can be either a dictionary of feature names and
        feature tensors, or a dictionary of edge types and the above dictionaries.

        The features copied from this dictionary will be assigned to ``edata`` of all
        blocks.
    """
    def __init__(self, iter_, device, graph, input_features, output_labels, node_features,
                 edge_features, is_block_sampler):
        super().__init__(iter_, device)

        self.graph = graph
        self.input_features = input_features
        self.output_labels = output_labels
        self.node_features = node_features
        self.edge_features = edge_features
        self.is_block_sampler = is_block_sampler

    def get_features(self, sample_result):
        # Async copy assumes that the sampled subgraphs are either all subgraphs or
        # all blocks.  In both cases, we assume that the input data is put into the
        # first block/subgraph and the output data is put into the last block/subgraph.
        # [TODO] If some of them are blocks and some of them are subgraphs,
        # we don't know in which subgraph we should put input data and output data.
        # Should we return input data and output data in place of the input node IDs and
        # output node IDs?
        input_nodes, output_nodes, subgs = sample_result
        input_data = _slice(self.input_features, input_nodes)
        if self.is_block_sampler:
            output_data = _slice(self.output_labels, output_nodes)
        else:
            # fetch the labels for all input nodes instead so that they can be put
            # in subgraph's ndata.
            output_data = _slice(self.output_labels, input_nodes)
        subgs_data = []
        for subg in subgs:
            assert subg.is_block == self.is_block_sampler, \
                "Async copy assumes that the sampler returns either all blocks or " \
                "all subgraphs."
            if self.is_block_sampler:
                block_srcdata = _slice(self.node_features, subg.srcdata[NID])
                block_dstdata = _slice(self.node_features, subg.dstdata[NID])
                block_edata = _slice(self.edge_features, subg.edata[EID], self.device)
                subgs_data.append((block_srcdata, block_dstdata, block_edata))
            else:
                subg_ndata = _slice(self.node_features, subg.ndata[NID])
                subg_edata = _slice(self.edge_features, subg.edata[EID])
                subgs_data.append((subg_ndata, subg_edata))
        return sample_result, (input_data, output_data, subgs_data)

    def assign_features(self, queue_result):
        (input_nodes, output_nodes, subgs), (input_data, output_data, subgs_data) = queue_result
        if self.is_block_sampler:
            for i, (subg, (block_srcdata, block_dstdata, block_edata)) in \
                    enumerate(zip(subgs, subgs_data)):
                if i == 0:      # inputs
                    _update_data(subg.srcdata, subg.srcnodes, input_data, subg.srctypes)
                if i == len(subgs) - 1:     # outputs
                    _update_data(subg.dstdata, subg.dstnodes, output_data, subg.dsttypes)
                # normal node data
                _update_data(subg.srcdata, subg.srcnodes, block_srcdata, subg.srctypes)
                _update_data(subg.dstdata, subg.dstnodes, block_dstdata, subg.dsttypes)
                # normal edge data
                _update_data(subg.edata, subg.edges, block_edata, subg.etypes)
        else:
            for i, (subg, (subg_ndata, subg_edata)) in enumerate(zip(subgs, subgs_data)):
                if i == 0:  # inputs
                    _update_data(subg.ndata, subg.nodes, input_data, subg.ntypes)
                if i == len(subgs) - 1:     # outputs
                    _update_data(subg.ndata, subg.nodes, output_data, subg.ntypes)
                # normal node data
                _update_data(subg.ndata, subg.nodes, subg_ndata, subg.ntypes)
                # normal edge data
                _update_data(subg.edata, subg.edges, subg_edata, subg.etypes)
        return input_nodes, output_nodes, subgs
