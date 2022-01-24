from collections.abc import Mapping, Awaitable
from functools import partial
from queue import Queue
import itertools
import threading
import asyncio
import random
import math
import torch
import torch.distributed as dist

from ..base import NID, EID
from ..heterograph import DGLHeteroGraph
from .. import ndarray as nd
from ..utils import (
    recursive_apply, ExceptionWrapper, async_recursive_apply, recursive_apply_pair,
    set_num_threads, create_shared_mem_array, get_shared_mem_array)
from ..frame import Column, LazyFeature
from .asyncio_wrapper import AsyncIO
from ..storages import TensorStorage, FeatureStorage
from .base import BlockSampler, EdgeBlockSampler

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


def _get_id_tensor_from_mapping(indices, device, keys):
    lengths = torch.LongTensor([
        (indices[k].shape[0] if k in indices else 0) for k in keys], device=device)
    type_ids = torch.arange(len(keys), device=device).repeat_interleave(lengths)
    all_indices = torch.cat([indices[k] for k in keys if k in indices])
    return torch.stack([type_ids, all_indices], 1)


def _divide_by_worker(dataset):
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


class TensorizedDataset(torch.utils.data.IterableDataset):
    """Custom Dataset wrapper that returns a minibatch as tensors or dicts of tensors.
    When the dataset is on the GPU, this significantly reduces the overhead.

    By default, using PyTorch's TensorDataset will return a list of scalar tensors
    which would introduce a lot of overhead, especially on GPU.
    """
    def __init__(self, indices, keys, batch_size, drop_last):
        if isinstance(indices, Mapping):
            self._device = next(iter(indices.values())).device
            self._tensor_dataset = _get_id_tensor_from_mapping(indices, self._device, keys)
            self._mapping_keys = keys
        else:
            self._tensor_dataset = indices
            self._device = indices.device
            self._mapping_keys = None
        self.batch_size = batch_size
        self.drop_last = drop_last

    def shuffle(self):
        # TODO: may need an in-place shuffle kernel
        perm = torch.randperm(self._tensor_dataset.shape[0], device=self._device)
        self._tensor_dataset[:] = self._tensor_dataset[perm]

    def __iter__(self):
        dataset = _divide_by_worker(self._tensor_dataset)
        return _TensorizedDatasetIter(
            dataset, self.batch_size, self.drop_last, self._mapping_keys)

def _get_shared_mem_name(id_):
    return f'ddp_{id_}'

def _generate_shared_mem_name_id():
    for _ in range(3):     # 3 trials
        id_ = random.getrandbits(32)
        name = _get_shared_mem_name(id_)
        if not nd.exist_shared_mem_array(name):
            return name, id_
    raise DGLError('Unable to generate a shared memory array')

class DDPTensorizedDataset(torch.utils.data.IterableDataset):
    def __init__(self, indices, keys, batch_size, drop_last, ddp_seed):
        self.rank = dist.get_rank()
        self.num_replicas = dist.get_world_size()
        self.seed = ddp_seed
        self.epoch = 0
        self.batch_size = batch_size
        self.drop_last = drop_last

        if self.drop_last and len(indices) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(indices) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(indices) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        # If drop_last is True, we create a shared memory array larger than the number
        # of indices since we will need to pad it after shuffling to make it evenly
        # divisible before every epoch.  If drop_last is False, we create an array
        # with the same size as the indices so we can trim it later.
        self.shared_mem_size = self.total_size if self.drop_last else len(indices)
        self.num_indices = len(indices)

        if self.rank == 0:
            name, id_ = _generate_shared_mem_name_id()
            if isinstance(indices, Mapping):
                device = next(iter(indices.values())).device
                id_tensor = _get_id_tensor_from_mapping(indices, device, keys)
                self._tensor_dataset = create_shared_mem_array(
                    name, (self.shared_mem_size, 2), torch.int64)
                self._tensor_dataset[:id_tensor.shape[0], :] = id_tensor
            else:
                self._tensor_dataset = create_shared_mem_array(
                    name, (self.shared_mem_size,), torch.int64)
                self._tensor_dataset[:len(indices)] = indices
            self._device = self._tensor_dataset.device
            meta_info = torch.LongTensor([id_, self._tensor_dataset.shape[0]])
        else:
            meta_info = torch.LongTensor([0, 0])

        if dist.get_backend() == 'nccl':
            # Use default CUDA device; PyTorch DDP required the users to set the CUDA
            # device for each process themselves so calling .cuda() should be safe.
            meta_info = meta_info.cuda()
        dist.broadcast(meta_info, src=0)

        if self.rank != 0:
            id_, num_samples = meta_info.tolist()
            name = _get_shared_mem_name(id_)
            if isinstance(indices, Mapping):
                indices_shared = get_shared_mem_array(name, (num_samples, 2), torch.int64)
            else:
                indices_shared = get_shared_mem_array(name, (num_samples,), torch.int64)
            self._tensor_dataset = indices_shared
            self._device = indices_shared.device

        if isinstance(indices, Mapping):
            self._mapping_keys = keys
        else:
            self._mapping_keys = None

    def shuffle(self):
        # Only rank 0 does the actual shuffling.  The other ranks wait for it.
        if self.rank == 0:
            self._tensor_dataset[:self.num_indices] = self._tensor_dataset[
                torch.randperm(self.num_indices, device=self._device)]
            if not self.drop_last:
                # pad extra
                self._tensor_dataset[self.num_indices:] = \
                    self._tensor_dataset[:self.total_size - self.num_indices]
        dist.barrier()

    def __iter__(self):
        start = self.num_samples * self.rank
        end = self.num_samples * (self.rank + 1)
        dataset = _divide_by_worker(self._tensor_dataset[start:end])
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
        default_id = frame.get(id_name, None)
        for key in frame.keys():
            column = frame[key]
            if isinstance(column, LazyFeature):
                parent_key = column.name or key
                if column.id_ is None and default_id is None:
                    raise DGLError(
                        'Found a LazyFeature with no ID specified, '
                        'and the graph does not have dgl.NID or dgl.EID columns')
                if (type_, parent_key) in storages:
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
    if isinstance(item, DGLHeteroGraph):
        return _prefetch_for_subgraph(item, dataloader)
    elif isinstance(item, LazyFeature):
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
    # (2) each LazyFeature object is replaced with a tensor or future.
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
    if isinstance(item, DGLHeteroGraph):
        subg = item
        for (tid, key), value in feat.node_feats.items():
            assert isinstance(subg._node_frames[tid][key], LazyFeature)
            subg._node_frames[tid][key] = value
        for (tid, key), value in feat.edge_feats.items():
            assert isinstance(subg._edge_frames[tid][key], LazyFeature)
            subg._edge_frames[tid][key] = value
        return subg
    elif isinstance(item, LazyFeature):
        return feat
    else:
        return item


def _prefetcher_entry(dataloader_it, dataloader, queue, num_threads, use_alternate_streams):
    use_asyncio = dataloader.use_asyncio
    # PyTorch will set the number of threads to 1 which slows down pin_memory() calls
    # in main process if a prefetching thread is created.
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    if use_alternate_streams:
        stream = (
                torch.cuda.Stream(device=dataloader.device)
                if dataloader.device.type == 'cuda' else None)
    else:
        stream = None

    try:
        for batch in dataloader_it:
            batch = recursive_apply(batch, restore_parent_storage_columns, dataloader.graph)
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


# DGLHeteroGraphs have the semantics of lazy feature slicing with subgraphs.  Such behavior depends
# on that DGLHeteroGraph's ndata and edata are maintained by Frames.  So to maintain compatibility
# with older code, DGLHeteroGraphs and other graph storages are handled separately: (1) DGLHeteroGraphs will
# preserve the lazy feature slicing for subgraphs.  (2) Other graph storages will not have
# lazy feature slicing; all feature slicing will be eager.
def remove_parent_storage_columns(item, g):
    if not isinstance(item, DGLHeteroGraph) or not isinstance(g, DGLHeteroGraph):
        return item

    for subframe, frame in zip(
            itertools.chain(item._node_frames, item._edge_frames),
            itertools.chain(g._node_frames, g._edge_frames)):
        for key in list(subframe.keys()):
            subcol = subframe._columns[key]   # directly get the column object
            if isinstance(subcol, LazyFeature):
                continue
            col = frame._columns.get(key, None)
            if col is None:
                continue
            if col.storage is subcol.storage:
                subcol.storage = None
    return item


def restore_parent_storage_columns(item, g):
    if not isinstance(item, DGLHeteroGraph) or not isinstance(g, DGLHeteroGraph):
        return item

    for subframe, frame in zip(
            itertools.chain(item._node_frames, item._edge_frames),
            itertools.chain(g._node_frames, g._edge_frames)):
        for key in subframe.keys():
            subcol = subframe._columns[key]
            if isinstance(subcol, LazyFeature):
                continue
            col = frame._columns.get(key, None)
            if col is None:
                continue
            if subcol.storage is None:
                subcol.storage = col.storage
    return item


class _PrefetchingIter(object):
    def __init__(self, dataloader, dataloader_it, use_thread=False, use_alternate_streams=True,
                 num_threads=None):
        self.queue = Queue(1)
        self.dataloader_it = dataloader_it
        self.dataloader = dataloader
        self.graph_sampler = self.dataloader.graph_sampler
        self.pin_memory = self.dataloader.pin_memory
        self.num_threads = num_threads

        self.use_thread = use_thread
        self.use_alternate_streams = use_alternate_streams
        if use_thread:
            thread = threading.Thread(
                target=_prefetcher_entry,
                args=(dataloader_it, dataloader, self.queue, num_threads, use_alternate_streams),
                daemon=True)
            thread.start()
            self.thread = thread

    def __iter__(self):
        return self

    def _next_non_threaded(self):
        batch = next(self.dataloader_it)
        batch = recursive_apply(batch, restore_parent_storage_columns, self.dataloader.graph)
        device = self.dataloader.device
        if self.use_alternate_streams:
            stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        else:
            stream = None
        if self.dataloader.use_asyncio:
            feats = AsyncIO().run(_async_prefetch(batch, self.dataloader, stream))
        else:
            feats = _prefetch(batch, self.dataloader, stream)
        batch = recursive_apply(batch, lambda x: x.to(device, non_blocking=True))
        stream_event = stream.record_event() if stream is not None else None
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
        if stream_event is not None:
            stream_event.wait()
        return batch


# Make them classes to work with pickling in mp.spawn
class CollateWrapper(object):
    def __init__(self, sample_func, g):
        self.sample_func = sample_func
        self.g = g

    def __call__(self, items):
        batch = self.sample_func(self.g, items)
        return recursive_apply(batch, remove_parent_storage_columns, self.g)


class WorkerInitWrapper(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, worker_id):
        set_num_threads(1)
        if self.func is not None:
            self.func(worker_id)


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
                storages[type_, key] = _wrap_storage(v)
    return storages


def create_tensorized_dataset(indices, keys, batch_size, drop_last, use_ddp, ddp_seed):
    if use_ddp:
        return DDPTensorizedDataset(indices, keys, batch_size, drop_last, ddp_seed)
    else:
        return TensorizedDataset(indices, keys, batch_size, drop_last)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, graph, indices, graph_sampler, device='cpu', use_ddp=False,
                 ddp_seed=0, batch_size=1, drop_last=False, shuffle=False,
                 use_asyncio=False, use_prefetch_thread=False, use_alternate_streams=True,
                 **kwargs):
        self.graph = graph
        if (torch.is_tensor(indices) or (
                isinstance(indices, Mapping) and
                all(torch.is_tensor(v) for v in indices.values()))):
            self.dataset = create_tensorized_dataset(
                indices, graph.ntypes, batch_size, drop_last, use_ddp, ddp_seed)
        else:
            self.dataset = indices
        self.ddp_seed = ddp_seed
        self._shuffle_dataset = shuffle
        self.graph_sampler = graph_sampler
        self.use_asyncio = use_asyncio
        self.device = torch.device(device)
        self.use_alternate_streams = use_alternate_streams
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.use_prefetch_thread = use_prefetch_thread
        worker_init_fn = WorkerInitWrapper(kwargs.get('worker_init_fn', None))

        # Instantiate all the formats if the number of workers is greater than 0.
        if kwargs.get('num_workers', 0) > 0 and hasattr(self.graph, 'create_formats_'):
            self.graph.create_formats_()

        self.node_data = _prepare_storages_from_graph(graph, 'ndata', graph.ntypes)
        self.edge_data = _prepare_storages_from_graph(graph, 'edata', graph.canonical_etypes)
        self.other_data = {}

        super().__init__(
            self.dataset,
            collate_fn=CollateWrapper(self.graph_sampler.sample, graph),
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
            use_alternate_streams=self.use_alternate_streams, num_threads=num_threads)

    # To allow data other than node/edge data to be prefetched.
    def attach_data(self, name, storage):
        self.other_data[name] = _wrap_storage(storage)


# Alias
class NodeDataLoader(DataLoader):
    pass


class EdgeDataLoader(DataLoader):
    def __init__(self, graph, indices, graph_sampler, device='cpu', use_ddp=False,
                 ddp_seed=0, batch_size=1, drop_last=False, shuffle=False,
                 use_asyncio=False, use_prefetch_thread=False, use_alternate_streams=True,
                 exclude=None, reverse_eids=None, reverse_etypes=None, negative_sampler=None,
                 **kwargs):
        if isinstance(graph_sampler, BlockSampler):
            graph_sampler = EdgeBlockSampler(
                graph_sampler, exclude=exclude, reverse_eids=reverse_eids,
                reverse_etypes=reverse_etypes, negative_sampler=negative_sampler)

        super().__init__(
            graph, indices, graph_sampler, device=device, use_ddp=use_ddp, ddp_seed=ddp_seed,
            batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, use_asyncio=use_asyncio,
            use_prefetch_thread=use_prefetch_thread, use_alternate_streams=use_alternate_streams,
            **kwargs)
