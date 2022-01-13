from collections.abc import Mapping, Awaitable
from functools import partial
from queue import Queue
import itertools
import threading
import asyncio
import torch
from dgl import DGLGraph, NID, EID
from dgl.utils import (
    recursive_apply, ExceptionWrapper, async_recursive_apply, recursive_apply_pair,
    set_num_threads)
from dgl.frame import Column, Marker
from .asyncio_wrapper import AsyncIO
from ..storages import TensorStorage, FeatureStorage
from .base import BlockSampler, EdgeWrapper, LinkWrapper

class _TensorizedDatasetIter(object):
    def __init__(self, dataset, batch_size, drop_last, mapping_keys):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.mapping_keys = mapping_keys
        self.index = 0

    # For PyTorch Lightning compatibility
    def __iter__(self):
        return self

    def _next_indices(self):
        num_items = self.dataset.shape[0]
        if self.index >= num_items:
            raise StopIteration
        end_idx = self.index + self.batch_size
        if end_idx > num_items:
            if self.drop_last:
                raise StopIteration
            end_idx = num_items
        batch = self.dataset[self.index:end_idx]
        self.index += self.batch_size

        return batch

    def __next__(self):
        batch = self._next_indices()
        if self.mapping_keys is None:
            return batch

        # convert the type-ID pairs to dictionary
        type_ids = batch[:, 0]
        indices = batch[:, 1]
        type_ids_sortidx = torch.argsort(type_ids)
        type_ids = type_ids[type_ids_sortidx]
        indices = indices[type_ids_sortidx]
        type_id_uniq, type_id_count = torch.unique_consecutive(type_ids, return_counts=True)
        type_id_uniq = type_id_uniq.tolist()
        type_id_offset = type_id_count.cumsum(0).tolist()
        type_id_offset.insert(0, 0)
        id_dict = {
            self.mapping_keys[type_id_uniq[i]]: indices[type_id_offset[i]:type_id_offset[i+1]]
            for i in range(len(type_id_uniq))}
        return id_dict


class TensorizedDataset(torch.utils.data.IterableDataset):
    """Custom Dataset wrapper that returns a minibatch as tensors or dicts of tensors.
    When the dataset is on the GPU, this significantly reduces the overhead.

    By default, using PyTorch's TensorDataset will return a list of scalar tensors
    which would introduce a lot of overhead, especially on GPU.
    """
    def __init__(self, indices, keys, batch_size, drop_last):
        if isinstance(indices, Mapping):
            self._init_with_mapping(indices, keys)
            self._mapping_keys = keys
        else:
            self._init_with_tensor(indices)
            self._mapping_keys = None
        self.batch_size = batch_size
        self.drop_last = drop_last

    def _init_with_mapping(self, indices, keys):
        device = next(iter(indices.values())).device
        self._device = device
        lengths = torch.LongTensor([indices[k].shape[0] for k in keys], device=device)
        type_ids = torch.arange(len(keys), device=device).repeat_interleave(lengths)
        all_indices = torch.cat([indices[k] for k in keys])
        self._tensor_dataset = torch.stack([type_ids, all_indices], 1)

    def _init_with_tensor(self, indices):
        self._tensor_dataset = indices
        self._device = indices.device

    def shuffle(self):
        # TODO: may need an in-place shuffle kernel
        perm = torch.randperm(self._tensor_dataset.shape[0], device=self._device)
        self._tensor_dataset[:] = self._tensor_dataset[perm]

    def _divide_by_worker(self, dataset):
        num_samples = dataset.shape[0]
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            chunk_size = num_samples // worker_info.num_workers
            left_over = num_samples % worker_info.num_workers
            start = (chunk_size * worker_info.id) + min(left_over, worker_info.id)
            end = start + chunk_size + (worker_info.id < left_over)
            assert worker_info.id < worker_info.num_workers - 1 or end == num_samples
            dataset = dataset[start:end]
        return dataset

    def __iter__(self):
        dataset = self._divide_by_worker(self._tensor_dataset)
        return _TensorizedDatasetIter(
            dataset, self.batch_size, self.drop_last, self._mapping_keys)


def _prefetch_for_column(id_, use_asyncio, storage, device, pin_memory):
    if use_asyncio:
        return asyncio.create_task(storage.async_fetch(id_, device, pin_memory=pin_memory))
    else:
        return storage.fetch(id_, device, pin_memory=pin_memory)


def _prefetch_update_feats(
        feats, frames, types, storages, id_name, use_asyncio, device, pin_memory):
    for tid, frame in enumerate(frames):
        type_ = types[tid]
        default_id = frame[id_name]
        for key in frame.keys():
            column = frame[key]
            if isinstance(column, Marker):
                parent_key = column.name or key
                feats[tid, key] = _prefetch_for_column(
                    column.id_ or default_id, use_asyncio, storages[type_, parent_key],
                    device, pin_memory)


# This class exists to avoid recursion into the feature dictionary returned by the
# prefetcher when calling recursive_apply().
class _PrefetchedGraphFeatures(object):
    __slots__ = ['node_feats', 'edge_feats']
    def __init__(self, node_feats, edge_feats):
        self.node_feats = node_feats
        self.edge_feats = edge_feats


def _prefetch_for_subgraph(subg, dataloader):
    g = dataloader.graph

    node_feats, edge_feats = {}, {}
    _prefetch_update_feats(
        node_feats, subg._node_frames, subg.ntypes, dataloader.node_data,
        NID, dataloader.use_asyncio, dataloader.device, dataloader.pin_memory)
    _prefetch_update_feats(
        edge_feats, subg._edge_frames, subg.canonical_etypes, dataloader.edge_data,
        EID, dataloader.use_asyncio, dataloader.device, dataloader.pin_memory)
    return _PrefetchedGraphFeatures(node_feats, edge_feats)


def _prefetch_for(item, dataloader):
    if isinstance(item, DGLGraph):
        return _prefetch_for_subgraph(item, dataloader)
    elif isinstance(item, Marker):
        return _prefetch_for_column(
            item.id_, dataloader.use_asyncio, dataloader.other_data[item.name],
            dataloader.device, dataloader.pin_memory)
    else:
        return None


def _prefetch(batch, dataloader, stream):
    # feats has the same nested structure of batch, except that
    # (1) each subgraph is replaced with a pair of node features and edge features, both
    #     being dictionaries whose keys are (type_id, column_name) and values are either
    #     tensors or futures.
    # (2) each Marker object is replaced with a tensor or future.
    # (3) everything else are replaced with None.
    with torch.cuda.stream(stream):
        feats = recursive_apply(batch, _prefetch_for, dataloader)
    return feats


async def _await_or_return(x):
    if isinstance(x, Awaitable):
        return await x
    elif isinstance(x, _PrefetchedGraphFeatures):
        node_feats = await async_recursive_apply(x.node_feats, _await_or_return)
        edge_feats = await async_recursive_apply(x.edge_feats, _await_or_return)
        return _PrefetchedGraphFeatures(node_feats, edge_feats)
    else:
        return x


async def _async_prefetch(batch, dataloader, stream):
    with torch.cuda.stream(stream):
        feats = _prefetch(batch, dataloader, stream)
        feats = await async_recursive_apply(feats, _await_or_return)
    return feats


def _assign_for(item, feat):
    if isinstance(item, DGLGraph):
        subg = item
        for (tid, key), value in feat.node_feats.items():
            assert isinstance(subg._node_frames[tid][key], Marker)
            subg._node_frames[tid][key] = value
        for (tid, key), value in feat.edge_feats.items():
            assert isinstance(subg._edge_frames[tid][key], Marker)
            subg._edge_frames[tid][key] = value
        return subg
    elif isinstance(item, Marker):
        return feat
    else:
        return item


def _prefetcher_entry(dataloader_it, dataloader, queue, num_threads):
    use_asyncio = dataloader.use_asyncio
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    stream = (
            torch.cuda.Stream(device=dataloader.device)
            if dataloader.device.type == 'cuda' else None)

    try:
        for batch in dataloader_it:
            if use_asyncio:
                feats = AsyncIO().run(_async_prefetch(batch, dataloader, stream))
            else:
                feats = _prefetch(batch, dataloader, stream)

            queue.put((
                # batch will be already in pinned memory as per the behavior of
                # PyTorch DataLoader.
                recursive_apply(batch, lambda x: x.to(dataloader.device, non_blocking=True)),
                feats,
                stream.record_event() if stream is not None else None,
                None))
        queue.put((None, None, None, None))
    except:     # pylint: disable=bare-except
        queue.put((None, None, None, ExceptionWrapper(where='in prefetcher')))


def remove_parent_storage_columns(item, g):
    if not isinstance(item, DGLGraph):
        return item

    for subframe, frame in zip(
            itertools.chain(item._node_frames, item._edge_frames),
            itertools.chain(g._node_frames, g._edge_frames)):
        for key in subframe.keys():
            subcol = subframe._columns[key]   # directly get the column object
            if isinstance(subcol, Marker):
                continue
            col = frame._columns.get(key, None)
            if col is None:
                continue
            if col.storage is subcol.storage:
                subcol.storage = None
    return item


def restore_parent_storage_columns(item, g):
    if not isinstance(item, DGLGraph):
        return item

    for subframe, frame in zip(
            itertools.chain(item._node_frames, item._edge_frames),
            itertools.chain(g._node_frames, g._edge_frames)):
        for key in subframe.keys():
            subcol = subframe._columns[key]
            if isinstance(subcol, Marker):
                continue
            col = frame._columns.get(key, None)
            if col is None:
                continue
            if subcol.storage is None:
                subcol.storage = col.storage
    return item


class _PrefetchingIter(object):
    def __init__(self, dataloader, dataloader_it, use_thread=False, num_threads=None):
        self.queue = Queue(1)
        self.dataloader_it = dataloader_it
        self.dataloader = dataloader
        self.graph_sampler = self.dataloader.graph_sampler
        self.pin_memory = self.dataloader.pin_memory
        self.num_threads = num_threads

        self.use_thread = use_thread
        if use_thread:
            thread = threading.Thread(
                target=_prefetcher_entry,
                args=(dataloader_it, dataloader, self.queue, num_threads),
                daemon=True)
            thread.start()
            self.thread = thread

    def __iter__(self):
        return self

    def _next_non_threaded(self):
        batch = next(self.dataloader_it)
        device = self.dataloader.device
        stream = torch.cuda.Stream(device=device)
        if self.dataloader.use_asyncio:
            feats = AsyncIO().run(_async_prefetch(batch, self.dataloader, stream))
        else:
            feats = _prefetch(batch, self.dataloader, stream)
        batch = recursive_apply(batch, lambda x: x.to(device, non_blocking=True))
        stream_event = stream.record_event()
        return batch, feats, stream_event

    def _next_threaded(self):
        batch, feats, stream_event, exception = self.queue.get()
        if batch is None:
            self.thread.join()
            if exception is None:
                raise StopIteration
            exception.reraise()
        return batch, feats, stream_event

    def __next__(self):
        batch, feats, stream_event = \
            self._next_non_threaded() if not self.use_thread else self._next_threaded()
        batch = recursive_apply_pair(batch, feats, _assign_for)
        batch = recursive_apply(batch, restore_parent_storage_columns, self.dataloader.graph)
        if stream_event is not None:
            stream_event.wait()
        return batch


def collate_wrapper(sample_func, g):
    def _sample(items):
        batch = sample_func(g, items)

        # Remove all the columns whose storages are parent storages (i.e. index is not
        # None)
        batch = recursive_apply(batch, remove_parent_storage_columns, g)
        return batch
    return _sample


def _wrap_worker_init_fn(func):
    def _func(worker_id):
        set_num_threads(1)
        if func is not None:
            func(worker_id)
    return _func


def _wrap_storage(storage):
    if torch.is_tensor(storage):
        return TensorStorage(storage)
    assert isinstance(storage, FeatureStorage), (
        "The frame column must be a tensor or a FeatureStorage object, got {}"
        .format(type(storage)))
    return storage


# Serves for two purposes: (1) to get around the overhead of calling
# g.ndata[key][ntype] etc., (2) to make tensors a TensorStorage.
def _prepare_storages_from_graph(graph, attr, types):
    storages = {}
    for key, value in getattr(graph, attr).items():
        if not isinstance(value, Mapping):
            assert len(types) == 1, "Expect a dict in {} if multiple types exist".format(attr)
            storages[types[0], key] = _wrap_storage(value)
        else:
            for type_, v in value.items():
                storages[type_, key] = v
    return storages


class NodeDataLoader(torch.utils.data.DataLoader):
    def __init__(self, graph, train_idx, graph_sampler, device='cpu', use_ddp=False,
                 ddp_seed=0, batch_size=1, drop_last=False, shuffle=False,
                 use_asyncio=False, use_prefetch_thread=False, **kwargs):
        self.graph = graph
        self.dataset = TensorizedDataset(train_idx, graph.ntypes, batch_size, drop_last)
        self.use_ddp = use_ddp
        self.ddp_seed = ddp_seed
        self._shuffle_dataset = shuffle
        self.graph_sampler = graph_sampler
        self.use_asyncio = use_asyncio
        self.device = torch.device(device)
        self.use_prefetch_thread = use_prefetch_thread
        worker_init_fn = _wrap_worker_init_fn(kwargs.get('worker_init_fn', None))

        # Instantiate all the formats if the number of workers is greater than 0.
        if kwargs.get('num_workers', 0) > 0 and hasattr(self.graph, 'create_formats_'):
            self.graph.create_formats_()

        self.node_data = _prepare_storages_from_graph(graph, 'ndata', graph.ntypes)
        self.edge_data = _prepare_storages_from_graph(graph, 'edata', graph.canonical_etypes)
        self.other_data = {}

        super().__init__(
            self.dataset,
            collate_fn=collate_wrapper(self.graph_sampler.sample, graph),
            batch_size=None,
            worker_init_fn=worker_init_fn,
            **kwargs)

    def __iter__(self):
        if self._shuffle_dataset:
            self.dataset.shuffle()
        # When using multiprocessing PyTorch sometimes set the number of PyTorch threads to 1
        # when spawning new Python threads.  This drastically slows down pinning features.
        num_threads = torch.get_num_threads() if self.num_workers > 0 else None
        return _PrefetchingIter(
            self, super().__iter__(), use_thread=self.use_prefetch_thread,
            num_threads=num_threads)

    # To allow data other than node/edge data to be prefetched.
    def attach_data(self, name, storage):
        self.other_data[name] = _wrap_storage(storage)


class EdgeDataLoader(torch.utils.data.DataLoader):
    def __init__(self, graph, train_idx, graph_sampler, device='cpu', use_ddp=False,
                 ddp_seed=0, batch_size=1, drop_last=False, shuffle=False,
                 use_asyncio=False, use_prefetch_thread=False, exclude=None,
                 reverse_eids=None, reverse_etypes=None, negative_sampler=None, **kwargs):
        self.graph = graph
        self.dataset = TensorizedDataset(train_idx, graph.ntypes, batch_size, drop_last)
        self.use_ddp = use_ddp
        self.ddp_seed = ddp_seed
        self._shuffle_dataset = shuffle
        self.graph_sampler = graph_sampler
        self.use_asyncio = use_asyncio
        self.device = torch.device(device)
        self.use_prefetch_thread = use_prefetch_thread
        worker_init_fn = _wrap_worker_init_fn(kwargs.get('worker_init_fn', None))

        self.node_data = _prepare_storages_from_graph(graph, 'ndata', graph.ntypes)
        self.edge_data = _prepare_storages_from_graph(graph, 'edata', graph.canonical_etypes)
        self.other_data = {}

        # Instantiate all the formats if the number of workers is greater than 0.
        if kwargs.get('num_workers', 0) > 0 and hasattr(self.graph, 'create_formats_'):
            self.graph.create_formats_()

        if isinstance(self.graph_sampler, BlockSampler):
            if negative_sampler is not None:
                self.graph_sampler = LinkWrapper(
                    self.graph_sampler, exclude=exclude, reverse_eids=reverse_eids,
                    reverse_etypes=reverse_etypes, negative_sampler=negative_sampler)
            else:
                self.graph_sampler = EdgeWrapper(
                    self.graph_sampler, exclude=exclude, reverse_eids=reverse_eids,
                    reverse_etypes=reverse_etypes)
            self.graph_sampler._freeze()

        super().__init__(
            self.dataset,
            collate_fn=collate_wrapper(self.graph_sampler.sample, graph),
            batch_size=None,
            worker_init_fn=worker_init_fn,
            **kwargs)

    def __iter__(self):
        if self._shuffle_dataset:
            self.dataset.shuffle()
        # When using multiprocessing PyTorch sometimes set the number of PyTorch threads to 1
        # when spawning new Python threads.  This drastically slows down pinning features.
        num_threads = torch.get_num_threads() if self.num_workers > 0 else None
        return _PrefetchingIter(
            self, super().__iter__(), use_thread=self.use_prefetch_thread,
            num_threads=num_threads)

    # To allow data other than node/edge data to be prefetched.
    def attach_data(self, name, storage):
        self.other_data[name] = _wrap_storage(storage)
