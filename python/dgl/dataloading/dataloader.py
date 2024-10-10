"""DGL PyTorch DataLoaders"""

import atexit
import inspect
import itertools
import math
import operator
import os
import re
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from functools import reduce
from queue import Empty, Full, Queue

import numpy as np
import psutil
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from .. import backend as F
from .._ffi.base import is_tensor_adaptor_enabled

from ..base import dgl_warning, DGLError, EID, NID
from ..batch import batch as batch_graphs
from ..cuda import GPUCache
from ..frame import LazyFeature
from ..heterograph import DGLGraph
from ..storages import wrap_storage
from ..utils import (
    dtype_of,
    ExceptionWrapper,
    get_num_threads,
    get_numa_nodes_cores,
    recursive_apply,
    recursive_apply_pair,
    set_num_threads,
)

PYTHON_EXIT_STATUS = False


def _set_python_exit_flag():
    global PYTHON_EXIT_STATUS
    PYTHON_EXIT_STATUS = True


atexit.register(_set_python_exit_flag)

prefetcher_timeout = int(os.environ.get("DGL_PREFETCHER_TIMEOUT", "30"))


class _TensorizedDatasetIter(object):
    def __init__(self, dataset, batch_size, drop_last, mapping_keys, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.mapping_keys = mapping_keys
        self.index = 0
        self.shuffle = shuffle

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
        batch = self.dataset[self.index : end_idx]
        self.index += self.batch_size

        return batch

    def __next__(self):
        batch = self._next_indices()
        if self.mapping_keys is None:
            # clone() fixes #3755, probably.  Not sure why.  Need to take a look afterwards.
            return batch.clone()

        # convert the type-ID pairs to dictionary
        type_ids = batch[:, 0]
        indices = batch[:, 1]
        _, type_ids_sortidx = torch.sort(type_ids, stable=True)
        type_ids = type_ids[type_ids_sortidx]
        indices = indices[type_ids_sortidx]
        type_id_uniq, type_id_count = torch.unique_consecutive(
            type_ids, return_counts=True
        )
        type_id_uniq = type_id_uniq.tolist()
        type_id_offset = type_id_count.cumsum(0).tolist()
        type_id_offset.insert(0, 0)
        id_dict = {
            self.mapping_keys[type_id_uniq[i]]: indices[
                type_id_offset[i] : type_id_offset[i + 1]
            ].clone()
            for i in range(len(type_id_uniq))
        }
        return id_dict


def _get_id_tensor_from_mapping(indices, device, keys):
    dtype = dtype_of(indices)
    id_tensor = torch.empty(
        sum(v.shape[0] for v in indices.values()), 2, dtype=dtype, device=device
    )

    offset = 0
    for i, k in enumerate(keys):
        if k not in indices:
            continue
        index = indices[k]
        length = index.shape[0]
        id_tensor[offset : offset + length, 0] = i
        id_tensor[offset : offset + length, 1] = index
        offset += length
    return id_tensor


def _split_to_local_id_tensor_from_mapping(
    indices, keys, local_lower_bound, local_upper_bound
):
    dtype = dtype_of(indices)
    device = next(iter(indices.values())).device
    num_samples = local_upper_bound - local_lower_bound
    id_tensor = torch.empty(num_samples, 2, dtype=dtype, device=device)

    index_offset = 0
    split_id_offset = 0
    for i, k in enumerate(keys):
        if k not in indices:
            continue
        index = indices[k]
        length = index.shape[0]
        index_offset2 = index_offset + length
        lower = max(local_lower_bound, index_offset)
        upper = min(local_upper_bound, index_offset2)
        if upper > lower:
            split_id_offset2 = split_id_offset + (upper - lower)
            assert split_id_offset2 <= num_samples
            id_tensor[split_id_offset:split_id_offset2, 0] = i
            id_tensor[split_id_offset:split_id_offset2, 1] = index[
                lower - index_offset : upper - index_offset
            ]
            split_id_offset += upper - lower
            if split_id_offset2 == num_samples:
                break
        index_offset = index_offset2
    return id_tensor


def _split_to_local_id_tensor(indices, local_lower_bound, local_upper_bound):
    dtype = dtype_of(indices)
    device = indices.device
    num_samples = local_upper_bound - local_lower_bound
    id_tensor = torch.empty(num_samples, dtype=dtype, device=device)

    if local_upper_bound > len(indices):
        remainder = len(indices) - local_lower_bound
        id_tensor[0:remainder] = indices[local_lower_bound:]
    else:
        id_tensor = indices[local_lower_bound:local_upper_bound]
    return id_tensor


def _divide_by_worker(dataset, batch_size, drop_last):
    num_samples = dataset.shape[0]
    worker_info = torch.utils.data.get_worker_info()
    if worker_info:
        num_batches = (
            num_samples + (0 if drop_last else batch_size - 1)
        ) // batch_size
        num_batches_per_worker = num_batches // worker_info.num_workers
        left_over = num_batches % worker_info.num_workers
        start = (num_batches_per_worker * worker_info.id) + min(
            left_over, worker_info.id
        )
        end = start + num_batches_per_worker + (worker_info.id < left_over)
        start *= batch_size
        end = min(end * batch_size, num_samples)
        dataset = dataset[start:end]
    return dataset


class TensorizedDataset(torch.utils.data.IterableDataset):
    """Custom Dataset wrapper that returns a minibatch as tensors or dicts of tensors.
    When the dataset is on the GPU, this significantly reduces the overhead.
    """

    def __init__(
        self, indices, batch_size, drop_last, shuffle, use_shared_memory
    ):
        if isinstance(indices, Mapping):
            self._mapping_keys = list(indices.keys())
            self._device = next(iter(indices.values())).device
            self._id_tensor = _get_id_tensor_from_mapping(
                indices, self._device, self._mapping_keys
            )
        else:
            self._id_tensor = indices
            self._device = indices.device
            self._mapping_keys = None
        # Use a shared memory array to permute indices for shuffling.  This is to make sure that
        # the worker processes can see it when persistent_workers=True, where self._indices
        # would not be duplicated every epoch.
        self._indices = torch.arange(
            self._id_tensor.shape[0], dtype=torch.int64
        )
        if use_shared_memory:
            self._indices.share_memory_()
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._shuffle = shuffle

    def shuffle(self):
        """Shuffle the dataset."""
        np.random.shuffle(self._indices.numpy())

    def __iter__(self):
        indices = _divide_by_worker(
            self._indices, self.batch_size, self.drop_last
        )
        id_tensor = self._id_tensor[indices]
        return _TensorizedDatasetIter(
            id_tensor,
            self.batch_size,
            self.drop_last,
            self._mapping_keys,
            self._shuffle,
        )

    def __len__(self):
        num_samples = self._id_tensor.shape[0]
        return (
            num_samples + (0 if self.drop_last else (self.batch_size - 1))
        ) // self.batch_size


def _decompose_one_dimension(length, world_size, rank, drop_last):
    if drop_last:
        num_samples = math.floor(length / world_size)
    else:
        num_samples = math.ceil(length / world_size)
    sta = rank * num_samples
    end = (rank + 1) * num_samples
    return sta, end


class DDPTensorizedDataset(torch.utils.data.IterableDataset):
    """Custom Dataset wrapper that returns a minibatch as tensors or dicts of tensors.
    When the dataset is on the GPU, this significantly reduces the overhead.

    This class additionally saves the index tensor in shared memory and therefore
    avoids duplicating the same index tensor during shuffling.
    """

    def __init__(self, indices, batch_size, drop_last, ddp_seed, shuffle):
        if isinstance(indices, Mapping):
            self._mapping_keys = list(indices.keys())
            len_indices = sum(len(v) for v in indices.values())
        else:
            self._mapping_keys = None
            len_indices = len(indices)

        self.rank = dist.get_rank()
        self.num_replicas = dist.get_world_size()
        self.seed = ddp_seed
        self.epoch = 0
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._shuffle = shuffle
        (
            self.local_lower_bound,
            self.local_upper_bound,
        ) = _decompose_one_dimension(
            len_indices, self.num_replicas, self.rank, drop_last
        )
        self.num_samples = self.local_upper_bound - self.local_lower_bound
        self.local_num_indices = self.num_samples
        if self.local_upper_bound > len_indices:
            assert not drop_last
            self.local_num_indices = len_indices - self.local_lower_bound

        if isinstance(indices, Mapping):
            self._id_tensor = _split_to_local_id_tensor_from_mapping(
                indices,
                self._mapping_keys,
                self.local_lower_bound,
                self.local_upper_bound,
            )
        else:
            self._id_tensor = _split_to_local_id_tensor(
                indices, self.local_lower_bound, self.local_upper_bound
            )
        self._device = self._id_tensor.device
        # padding self._indices when drop_last = False (self._indices always on cpu)
        self._indices = torch.empty(self.num_samples, dtype=torch.int64)
        torch.arange(
            self.local_num_indices, out=self._indices[: self.local_num_indices]
        )
        if not drop_last:
            torch.arange(
                self.num_samples - self.local_num_indices,
                out=self._indices[self.local_num_indices :],
            )
        assert len(self._id_tensor) == self.num_samples

    def shuffle(self):
        """Shuffles the dataset."""
        np.random.shuffle(self._indices[: self.local_num_indices].numpy())
        if not self.drop_last:
            # pad extra from local indices
            self._indices[self.local_num_indices :] = self._indices[
                : self.num_samples - self.local_num_indices
            ]

    def __iter__(self):
        indices = _divide_by_worker(
            self._indices, self.batch_size, self.drop_last
        )
        id_tensor = self._id_tensor[indices]
        return _TensorizedDatasetIter(
            id_tensor,
            self.batch_size,
            self.drop_last,
            self._mapping_keys,
            self._shuffle,
        )

    def __len__(self):
        return (
            self.num_samples + (0 if self.drop_last else (self.batch_size - 1))
        ) // self.batch_size


def _numel_of_shape(shape):
    return reduce(operator.mul, shape, 1)


def _init_gpu_caches(graph, gpu_caches):
    if not hasattr(graph, "_gpu_caches"):
        graph._gpu_caches = {"node": {}, "edge": {}}
    if gpu_caches is None:
        return
    assert isinstance(gpu_caches, dict), "GPU cache argument should be a dict"
    for i, frames in enumerate([graph._node_frames, graph._edge_frames]):
        node_or_edge = ["node", "edge"][i]
        cache_inf = gpu_caches.get(node_or_edge, {})
        for tid, frame in enumerate(frames):
            type_ = [graph.ntypes, graph.canonical_etypes][i][tid]
            for key in frame.keys():
                if key in cache_inf and cache_inf[key] > 0:
                    column = frame._columns[key]
                    if (key, type_) not in graph._gpu_caches[node_or_edge]:
                        cache = GPUCache(
                            cache_inf[key],
                            _numel_of_shape(column.shape),
                            graph.idtype,
                        )
                        graph._gpu_caches[node_or_edge][key, type_] = (
                            cache,
                            column.shape,
                        )


def _prefetch_update_feats(
    feats,
    frames,
    types,
    get_storage_func,
    id_name,
    device,
    pin_prefetcher,
    gpu_caches,
):
    for tid, frame in enumerate(frames):
        type_ = types[tid]
        default_id = frame.get(id_name, None)
        for key in frame.keys():
            column = frame._columns[key]
            if isinstance(column, LazyFeature):
                parent_key = column.name or key
                if column.id_ is None and default_id is None:
                    raise DGLError(
                        "Found a LazyFeature with no ID specified, "
                        "and the graph does not have dgl.NID or dgl.EID columns"
                    )
                ids = column.id_ or default_id
                if (parent_key, type_) in gpu_caches:
                    cache, item_shape = gpu_caches[parent_key, type_]
                    values, missing_index, missing_keys = cache.query(ids)
                    missing_values = get_storage_func(parent_key, type_).fetch(
                        missing_keys, device, pin_prefetcher
                    )
                    cache.replace(
                        missing_keys, F.astype(missing_values, F.float32)
                    )
                    values = F.astype(values, F.dtype(missing_values))
                    F.scatter_row_inplace(values, missing_index, missing_values)
                    # Reshape the flattened result to match the original shape.
                    F.reshape(values, (values.shape[0],) + item_shape)
                    values.__cache_miss__ = missing_keys.shape[0] / ids.shape[0]
                    feats[tid, key] = values
                else:
                    feats[tid, key] = get_storage_func(parent_key, type_).fetch(
                        ids, device, pin_prefetcher
                    )


# This class exists to avoid recursion into the feature dictionary returned by the
# prefetcher when calling recursive_apply().
class _PrefetchedGraphFeatures(object):
    __slots__ = ["node_feats", "edge_feats"]

    def __init__(self, node_feats, edge_feats):
        self.node_feats = node_feats
        self.edge_feats = edge_feats


def _prefetch_for_subgraph(subg, dataloader):
    node_feats, edge_feats = {}, {}
    _prefetch_update_feats(
        node_feats,
        subg._node_frames,
        subg.ntypes,
        dataloader.graph.get_node_storage,
        NID,
        dataloader.device,
        dataloader.pin_prefetcher,
        dataloader.graph._gpu_caches["node"],
    )
    _prefetch_update_feats(
        edge_feats,
        subg._edge_frames,
        subg.canonical_etypes,
        dataloader.graph.get_edge_storage,
        EID,
        dataloader.device,
        dataloader.pin_prefetcher,
        dataloader.graph._gpu_caches["edge"],
    )
    return _PrefetchedGraphFeatures(node_feats, edge_feats)


def _prefetch_for(item, dataloader):
    if isinstance(item, DGLGraph):
        return _prefetch_for_subgraph(item, dataloader)
    elif isinstance(item, LazyFeature):
        return dataloader.other_storages[item.name].fetch(
            item.id_, dataloader.device, dataloader.pin_prefetcher
        )
    else:
        return None


def _await_or_return(x):
    if hasattr(x, "wait"):
        return x.wait()
    elif isinstance(x, _PrefetchedGraphFeatures):
        node_feats = recursive_apply(x.node_feats, _await_or_return)
        edge_feats = recursive_apply(x.edge_feats, _await_or_return)
        return _PrefetchedGraphFeatures(node_feats, edge_feats)
    else:
        return x


def _record_stream(x, stream):
    if stream is None:
        return x
    if hasattr(x, "record_stream"):
        x.record_stream(stream)
        return x
    elif isinstance(x, _PrefetchedGraphFeatures):
        node_feats = recursive_apply(x.node_feats, _record_stream, stream)
        edge_feats = recursive_apply(x.edge_feats, _record_stream, stream)
        return _PrefetchedGraphFeatures(node_feats, edge_feats)
    else:
        return x


def _prefetch(batch, dataloader, stream):
    # feats has the same nested structure of batch, except that
    # (1) each subgraph is replaced with a pair of node features and edge features, both
    #     being dictionaries whose keys are (type_id, column_name) and values are either
    #     tensors or futures.
    # (2) each LazyFeature object is replaced with a tensor or future.
    # (3) everything else are replaced with None.
    #
    # Once the futures are fetched, this function waits for them to complete by
    # calling its wait() method.
    if stream is not None:
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(stream)
    else:
        current_stream = None
    with torch.cuda.stream(stream):
        # fetch node/edge features
        feats = recursive_apply(batch, _prefetch_for, dataloader)
        feats = recursive_apply(feats, _await_or_return)
        feats = recursive_apply(feats, _record_stream, current_stream)
        # transfer input nodes/seed nodes/subgraphs
        batch = recursive_apply(
            batch, lambda x: x.to(dataloader.device, non_blocking=True)
        )
        batch = recursive_apply(batch, _record_stream, current_stream)
    stream_event = stream.record_event() if stream is not None else None
    return batch, feats, stream_event


def _assign_for(item, feat):
    if isinstance(item, DGLGraph):
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


def _put_if_event_not_set(queue, result, event):
    while not event.is_set():
        try:
            queue.put(result, timeout=1.0)
            break
        except Full:
            continue


def _prefetcher_entry(
    dataloader_it, dataloader, queue, num_threads, stream, done_event
):
    # PyTorch will set the number of threads to 1 which slows down pin_memory() calls
    # in main process if a prefetching thread is created.
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    try:
        while not done_event.is_set():
            try:
                batch = next(dataloader_it)
            except StopIteration:
                break
            batch = recursive_apply(
                batch, restore_parent_storage_columns, dataloader.graph
            )
            batch, feats, stream_event = _prefetch(batch, dataloader, stream)
            _put_if_event_not_set(
                queue, (batch, feats, stream_event, None), done_event
            )
        _put_if_event_not_set(queue, (None, None, None, None), done_event)
    except:  # pylint: disable=bare-except
        _put_if_event_not_set(
            queue,
            (None, None, None, ExceptionWrapper(where="in prefetcher")),
            done_event,
        )


# DGLGraphs have the semantics of lazy feature slicing with subgraphs.  Such behavior depends
# on that DGLGraph's ndata and edata are maintained by Frames.  So to maintain compatibility
# with older code, DGLGraphs and other graph storages are handled separately: (1)
# DGLGraphs will preserve the lazy feature slicing for subgraphs.  (2) Other graph storages
# will not have lazy feature slicing; all feature slicing will be eager.
def remove_parent_storage_columns(item, g):
    """Removes the storage objects in the given graphs' Frames if it is a sub-frame of the
    given parent graph, so that the storages are not serialized during IPC from PyTorch
    DataLoader workers.
    """
    if not isinstance(item, DGLGraph) or not isinstance(g, DGLGraph):
        return item

    for subframe, frame in zip(
        itertools.chain(item._node_frames, item._edge_frames),
        itertools.chain(g._node_frames, g._edge_frames),
    ):
        for key in list(subframe.keys()):
            subcol = subframe._columns[key]  # directly get the column object
            if isinstance(subcol, LazyFeature):
                continue
            col = frame._columns.get(key, None)
            if col is None:
                continue
            if col.storage is subcol.storage:
                subcol.storage = None
    return item


def restore_parent_storage_columns(item, g):
    """Restores the storage objects in the given graphs' Frames if it is a sub-frame of the
    given parent graph (i.e. when the storage object is None).
    """
    if not isinstance(item, DGLGraph) or not isinstance(g, DGLGraph):
        return item

    for subframe, frame in zip(
        itertools.chain(item._node_frames, item._edge_frames),
        itertools.chain(g._node_frames, g._edge_frames),
    ):
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
    def __init__(self, dataloader, dataloader_it, num_threads=None):
        self.queue = Queue(1)
        self.dataloader_it = dataloader_it
        self.dataloader = dataloader
        self.num_threads = num_threads

        self.use_thread = dataloader.use_prefetch_thread
        self.use_alternate_streams = dataloader.use_alternate_streams
        self.device = self.dataloader.device
        if self.use_alternate_streams and self.device.type == "cuda":
            self.stream = torch.cuda.Stream(device=self.device)
        else:
            self.stream = None
        self._shutting_down = False
        if self.use_thread:
            self._done_event = threading.Event()
            thread = threading.Thread(
                target=_prefetcher_entry,
                args=(
                    dataloader_it,
                    dataloader,
                    self.queue,
                    num_threads,
                    self.stream,
                    self._done_event,
                ),
                daemon=True,
            )
            thread.start()
            self.thread = thread

    def __iter__(self):
        return self

    def _shutdown(self):
        # Sometimes when Python is exiting complicated operations like
        # self.queue.get_nowait() will hang.  So we set it to no-op and let Python handle
        # the rest since the thread is daemonic.
        # PyTorch takes the same solution.
        if PYTHON_EXIT_STATUS is True or PYTHON_EXIT_STATUS is None:
            return
        if not self._shutting_down:
            try:
                self._shutting_down = True
                self._done_event.set()

                try:
                    self.queue.get_nowait()  # In case the thread is blocking on put().
                except:  # pylint: disable=bare-except
                    pass

                self.thread.join()
            except:  # pylint: disable=bare-except
                pass

    def __del__(self):
        if self.use_thread:
            self._shutdown()

    def _next_non_threaded(self):
        batch = next(self.dataloader_it)
        batch = recursive_apply(
            batch, restore_parent_storage_columns, self.dataloader.graph
        )
        batch, feats, stream_event = _prefetch(
            batch, self.dataloader, self.stream
        )
        return batch, feats, stream_event

    def _next_threaded(self):
        try:
            batch, feats, stream_event, exception = self.queue.get(
                timeout=prefetcher_timeout
            )
        except Empty:
            raise RuntimeError(
                f"Prefetcher thread timed out at {prefetcher_timeout} seconds."
            )
        if batch is None:
            self.thread.join()
            if exception is None:
                raise StopIteration
            exception.reraise()
        return batch, feats, stream_event

    def __next__(self):
        batch, feats, stream_event = (
            self._next_non_threaded()
            if not self.use_thread
            else self._next_threaded()
        )
        batch = recursive_apply_pair(batch, feats, _assign_for)
        if stream_event is not None:
            stream_event.wait()
        return batch


# Make them classes to work with pickling in mp.spawn
class CollateWrapper(object):
    """Wraps a collate function with :func:`remove_parent_storage_columns` for serializing
    from PyTorch DataLoader workers.
    """

    def __init__(self, sample_func, g, use_uva, device):
        self.sample_func = sample_func
        self.g = g
        self.use_uva = use_uva
        self.device = device

    def __call__(self, items):
        graph_device = getattr(self.g, "device", None)
        if self.use_uva or (graph_device != torch.device("cpu")):
            # Only copy the indices to the given device if in UVA mode or the graph
            # is not on CPU.
            items = recursive_apply(items, lambda x: x.to(self.device))
        batch = self.sample_func(self.g, items)
        return recursive_apply(batch, remove_parent_storage_columns, self.g)


class WorkerInitWrapper(object):
    """Wraps the :attr:`worker_init_fn` argument of the DataLoader to set the number of DGL
    OMP threads to 1 for PyTorch DataLoader workers.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, worker_id):
        set_num_threads(1)
        if self.func is not None:
            self.func(worker_id)


def create_tensorized_dataset(
    indices,
    batch_size,
    drop_last,
    use_ddp,
    ddp_seed,
    shuffle,
    use_shared_memory,
):
    """Converts a given indices tensor to a TensorizedDataset, an IterableDataset
    that returns views of the original tensor, to reduce overhead from having
    a list of scalar tensors in default PyTorch DataLoader implementation.
    """
    if use_ddp:
        # DDP always uses shared memory
        return DDPTensorizedDataset(
            indices, batch_size, drop_last, ddp_seed, shuffle
        )
    else:
        return TensorizedDataset(
            indices, batch_size, drop_last, shuffle, use_shared_memory
        )


def _get_device(device):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


class DataLoader(torch.utils.data.DataLoader):
    """Sampled graph data loader. Wrap a :class:`~dgl.DGLGraph` and a
    :class:`~dgl.dataloading.Sampler` into an iterable over mini-batches of samples.

    DGL's ``DataLoader`` extends PyTorch's ``DataLoader`` by handling creation
    and transmission of graph samples. It supports iterating over a set of nodes,
    edges or any kinds of indices to get samples in the form of ``DGLGraph``, message
    flow graphs (MFGS), or any other structures necessary to train a graph neural network.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    indices : Tensor or dict[ntype, Tensor]
        The set of indices.  It can either be a tensor of integer indices or a dictionary
        of types and indices.

        The actual meaning of the indices is defined by the :meth:`sample` method of
        :attr:`graph_sampler`.
    graph_sampler : dgl.dataloading.Sampler
        The subgraph sampler.
    device : device context, optional
        The device of the generated MFGs in each iteration, which should be a
        PyTorch device object (e.g., ``torch.device``).

        By default this value is None. If :attr:`use_uva` is True, MFGs and graphs will
        generated in torch.cuda.current_device(), otherwise generated in the same device
        of :attr:`g`.
    use_ddp : boolean, optional
        If True, tells the DataLoader to split the training set for each
        participating process appropriately using
        :class:`torch.utils.data.distributed.DistributedSampler`.

        Overrides the :attr:`sampler` argument of :class:`torch.utils.data.DataLoader`.
    ddp_seed : int, optional
        The seed for shuffling the dataset in
        :class:`torch.utils.data.distributed.DistributedSampler`.

        Only effective when :attr:`use_ddp` is True.
    use_uva : bool, optional
        Whether to use Unified Virtual Addressing (UVA) to directly sample the graph
        and slice the features from CPU into GPU.  Setting it to True will pin the
        graph and feature tensors into pinned memory.

        If True, requires that :attr:`indices` must have the same device as the
        :attr:`device` argument.

        Default: False.
    use_prefetch_thread : bool, optional
        (Advanced option)
        Spawns a new Python thread to perform feature slicing
        asynchronously.  Can make things faster at the cost of GPU memory.

        Default: True if the graph is on CPU and :attr:`device` is CUDA.  False otherwise.
    use_alternate_streams : bool, optional
        (Advanced option)
        Whether to slice and transfers the features to GPU on a non-default stream.

        Default: True if the graph is on CPU, :attr:`device` is CUDA, and :attr:`use_uva`
        is False.  False otherwise.
    pin_prefetcher : bool, optional
        (Advanced option)
        Whether to pin the feature tensors into pinned memory.

        Default: True if the graph is on CPU and :attr:`device` is CUDA.  False otherwise.
    gpu_cache : dict[dict], optional
        Which node and edge features to cache using HugeCTR gpu_cache. Example:
        {"node": {"features": 500000}, "edge": {"types": 4000000}} would
        indicate that we want to cache 500k of the node "features" and 4M of the
        edge "types" in GPU caches.

        Is supported only on NVIDIA GPUs with compute capability 70 or above.
        The dictionary holds the keys of features along with the corresponding
        cache sizes. Please see
        https://github.com/NVIDIA-Merlin/HugeCTR/blob/main/gpu_cache/ReadMe.md
        for further reference.
    kwargs : dict
        Key-word arguments to be passed to the parent PyTorch
        :py:class:`torch.utils.data.DataLoader` class. Common arguments are:

          - ``batch_size`` (int): The number of indices in each batch.
          - ``drop_last`` (bool): Whether to drop the last incomplete batch.
          - ``shuffle`` (bool): Whether to randomly shuffle the indices at each epoch.


    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from 15 neighbors on the
    first layer, 10 neighbors on the second, and 5 neighbors on the third (assume
    the backend is PyTorch):

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> dataloader = dgl.dataloading.DataLoader(
    ...     g, train_nid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     train_on(input_nodes, output_nodes, blocks)

    **Using with Distributed Data Parallel**

    If you are using PyTorch's distributed training (e.g. when using
    :mod:`torch.nn.parallel.DistributedDataParallel`), you can train the model by turning
    on the `use_ddp` option:

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> dataloader = dgl.dataloading.DataLoader(
    ...     g, train_nid, sampler, use_ddp=True,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for epoch in range(start_epoch, n_epochs):
    ...     for input_nodes, output_nodes, blocks in dataloader:
    ...         train_on(input_nodes, output_nodes, blocks)

    Notes
    -----
    Please refer to
    :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`
    and :ref:`User Guide Section 6 <guide-minibatch>` for usage.

    **Tips for selecting the proper device**

    * If the input graph :attr:`g` is on GPU, the output device :attr:`device` must be the same GPU
      and :attr:`num_workers` must be zero. In this case, the sampling and subgraph construction
      will take place on the GPU. This is the recommended setting when using a single-GPU and
      the whole graph fits in GPU memory.

    * If the input graph :attr:`g` is on CPU while the output device :attr:`device` is GPU, then
      depending on the value of :attr:`use_uva`:

      - If :attr:`use_uva` is set to True, the sampling and subgraph construction will happen
        on GPU even if the GPU itself cannot hold the entire graph. This is the recommended
        setting unless there are operations not supporting UVA. :attr:`num_workers` must be 0
        in this case.

      - Otherwise, both the sampling and subgraph construction will take place on the CPU.
    """

    def __init__(
        self,
        graph,
        indices,
        graph_sampler,
        device=None,
        use_ddp=False,
        ddp_seed=0,
        batch_size=1,
        drop_last=False,
        shuffle=False,
        use_prefetch_thread=None,
        use_alternate_streams=None,
        pin_prefetcher=None,
        use_uva=False,
        gpu_cache=None,
        **kwargs,
    ):
        # (BarclayII) PyTorch Lightning sometimes will recreate a DataLoader from an existing
        # DataLoader with modifications to the original arguments.  The arguments are retrieved
        # from the attributes with the same name, and because we change certain arguments
        # when calling super().__init__() (e.g. batch_size attribute is None even if the
        # batch_size argument is not, so the next DataLoader's batch_size argument will be
        # None), we cannot reinitialize the DataLoader with attributes from the previous
        # DataLoader directly.
        # A workaround is to check whether "collate_fn" appears in kwargs.  If "collate_fn"
        # is indeed in kwargs and it's already a CollateWrapper object, we can assume that
        # the arguments come from a previously created DGL DataLoader, and directly initialize
        # the new DataLoader from kwargs without any changes.
        if isinstance(kwargs.get("collate_fn", None), CollateWrapper):
            assert batch_size is None  # must be None
            # restore attributes
            self.graph = graph
            self.indices = indices
            self.graph_sampler = graph_sampler
            self.device = device
            self.use_ddp = use_ddp
            self.ddp_seed = ddp_seed
            self.shuffle = shuffle
            self.drop_last = drop_last
            self.use_prefetch_thread = use_prefetch_thread
            self.use_alternate_streams = use_alternate_streams
            self.pin_prefetcher = pin_prefetcher
            self.use_uva = use_uva
            kwargs["batch_size"] = None
            super().__init__(**kwargs)
            return

        # (BarclayII) I hoped that pin_prefetcher can be merged into PyTorch's native
        # pin_memory argument.  But our neighbor samplers and subgraph samplers
        # return indices, which could be CUDA tensors (e.g. during UVA sampling)
        # hence cannot be pinned.  PyTorch's native pin memory thread does not ignore
        # CUDA tensors when pinning and will crash.  To enable pin memory for prefetching
        # features and disable pin memory for sampler's return value, I had to use
        # a different argument.  Of course I could change the meaning of pin_memory
        # to pinning prefetched features and disable pin memory for sampler's returns
        # no matter what, but I doubt if it's reasonable.
        self.graph = graph
        self.indices = indices  # For PyTorch-Lightning
        num_workers = kwargs.get("num_workers", 0)

        indices_device = None
        try:
            if isinstance(indices, Mapping):
                indices = {
                    k: (torch.tensor(v) if not torch.is_tensor(v) else v)
                    for k, v in indices.items()
                }
                indices_device = next(iter(indices.values())).device
            else:
                indices = (
                    torch.tensor(indices)
                    if not torch.is_tensor(indices)
                    else indices
                )
                indices_device = indices.device
        except:  # pylint: disable=bare-except
            # ignore when it fails to convert to torch Tensors.
            pass

        if indices_device is None:
            if not hasattr(indices, "device"):
                raise AttributeError(
                    'Custom indices dataset requires a "device" \
                attribute indicating where the indices is.'
                )
            indices_device = indices.device

        if device is None:
            if use_uva:
                device = torch.cuda.current_device()
            else:
                device = self.graph.device
        self.device = _get_device(device)

        # Sanity check - we only check for DGLGraphs.
        if isinstance(self.graph, DGLGraph):
            # Check graph and indices device as well as num_workers
            if use_uva:
                if self.graph.device.type != "cpu":
                    raise ValueError(
                        "Graph must be on CPU if UVA sampling is enabled."
                    )
                if num_workers > 0:
                    raise ValueError(
                        "num_workers must be 0 if UVA sampling is enabled."
                    )

                # Create all the formats and pin the features - custom GraphStorages
                # will need to do that themselves.
                self.graph.create_formats_()
                self.graph.pin_memory_()
            else:
                if self.graph.device != indices_device:
                    raise ValueError(
                        "Expect graph and indices to be on the same device when use_uva=False. "
                    )
                if self.graph.device.type == "cuda" and num_workers > 0:
                    raise ValueError(
                        "num_workers must be 0 if graph and indices are on CUDA."
                    )
                if self.graph.device.type == "cpu" and num_workers > 0:
                    # Instantiate all the formats if the number of workers is greater than 0.
                    self.graph.create_formats_()

            # Check pin_prefetcher and use_prefetch_thread - should be only effective
            # if performing CPU sampling but output device is CUDA
            if (
                self.device.type == "cuda"
                and self.graph.device.type == "cpu"
                and not use_uva
            ):
                if pin_prefetcher is None:
                    pin_prefetcher = True
                if use_prefetch_thread is None:
                    use_prefetch_thread = True
            else:
                if pin_prefetcher is True:
                    raise ValueError(
                        "pin_prefetcher=True is only effective when device=cuda and "
                        "sampling is performed on CPU."
                    )
                if pin_prefetcher is None:
                    pin_prefetcher = False

                if use_prefetch_thread is True:
                    raise ValueError(
                        "use_prefetch_thread=True is only effective when device=cuda and "
                        "sampling is performed on CPU."
                    )
                if use_prefetch_thread is None:
                    use_prefetch_thread = False

            # Check use_alternate_streams
            if use_alternate_streams is None:
                use_alternate_streams = (
                    self.device.type == "cuda"
                    and self.graph.device.type == "cpu"
                    and not use_uva
                    and is_tensor_adaptor_enabled()
                )
            elif use_alternate_streams and not is_tensor_adaptor_enabled():
                dgl_warning(
                    "use_alternate_streams is turned off because "
                    "TensorAdaptor is not available."
                )
                use_alternate_streams = False

        if torch.is_tensor(indices) or (
            isinstance(indices, Mapping)
            and all(torch.is_tensor(v) for v in indices.values())
        ):
            self.dataset = create_tensorized_dataset(
                indices,
                batch_size,
                drop_last,
                use_ddp,
                ddp_seed,
                shuffle,
                kwargs.get("persistent_workers", False),
            )
        else:
            self.dataset = indices

        self.ddp_seed = ddp_seed
        self.use_ddp = use_ddp
        self.use_uva = use_uva
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.graph_sampler = graph_sampler
        self.use_alternate_streams = use_alternate_streams
        self.pin_prefetcher = pin_prefetcher
        self.use_prefetch_thread = use_prefetch_thread
        self.cpu_affinity_enabled = False

        worker_init_fn = WorkerInitWrapper(kwargs.pop("worker_init_fn", None))

        self.other_storages = {}

        _init_gpu_caches(self.graph, gpu_cache)

        super().__init__(
            self.dataset,
            collate_fn=CollateWrapper(
                self.graph_sampler.sample, graph, self.use_uva, self.device
            ),
            batch_size=None,
            pin_memory=self.pin_prefetcher,
            worker_init_fn=worker_init_fn,
            **kwargs,
        )

    def __iter__(self):
        if (
            self.device.type == "cpu"
            and hasattr(psutil.Process, "cpu_affinity")
            and not self.cpu_affinity_enabled
        ):
            link = "https://docs.dgl.ai/tutorials/cpu/cpu_best_practises.html"
            dgl_warning(
                f"Dataloader CPU affinity opt is not enabled, consider switching it on "
                f"(see enable_cpu_affinity() or CPU best practices for DGL [{link}])"
            )

        if self.shuffle:
            self.dataset.shuffle()
        # When using multiprocessing PyTorch sometimes set the number of PyTorch threads to 1
        # when spawning new Python threads.  This drastically slows down pinning features.
        num_threads = torch.get_num_threads() if self.num_workers > 0 else None
        return _PrefetchingIter(
            self, super().__iter__(), num_threads=num_threads
        )

    @contextmanager
    def enable_cpu_affinity(
        self, loader_cores=None, compute_cores=None, verbose=True
    ):
        """Helper method for enabling cpu affinity for compute threads and dataloader workers
        Only for CPU devices
        Uses only NUMA node 0 by default for multi-node systems

        Parameters
        ----------
        loader_cores : [int] (optional)
            List of cpu cores to which dataloader workers should affinitize to.
            default: node0_cores[0:num_workers]

        compute_cores : [int] (optional)
            List of cpu cores to which compute threads should affinitize to
            default: node0_cores[num_workers:]

        verbose : bool (optional)
            If True, affinity information will be printed to the console

        Usage
        -----
        with dataloader.enable_cpu_affinity():
            <training loop>
        """
        if self.device.type == "cpu":
            if not self.num_workers > 0:
                raise Exception(
                    "ERROR: affinity should be used with at least one DL worker"
                )
            if loader_cores and len(loader_cores) != self.num_workers:
                raise Exception(
                    "ERROR: cpu_affinity incorrect "
                    "number of loader_cores={} for num_workers={}".format(
                        loader_cores, self.num_workers
                    )
                )

            # False positive E0203 (access-member-before-definition) linter warning
            worker_init_fn_old = self.worker_init_fn  # pylint: disable=E0203
            affinity_old = psutil.Process().cpu_affinity()
            nthreads_old = get_num_threads()

            compute_cores = compute_cores[:] if compute_cores else []
            loader_cores = loader_cores[:] if loader_cores else []

            def init_fn(worker_id):
                try:
                    psutil.Process().cpu_affinity([loader_cores[worker_id]])
                except:
                    raise Exception(
                        "ERROR: cannot use affinity id={} cpu={}".format(
                            worker_id, loader_cores
                        )
                    )

                worker_init_fn_old(worker_id)

            if not loader_cores or not compute_cores:
                numa_info = get_numa_nodes_cores()
                if numa_info and len(numa_info[0]) > self.num_workers:
                    # take one thread per each node 0 core
                    node0_cores = [cpus[0] for core_id, cpus in numa_info[0]]
                else:
                    node0_cores = list(range(psutil.cpu_count(logical=False)))

                if len(node0_cores) < self.num_workers:
                    raise Exception("ERROR: more workers than available cores")

                loader_cores = loader_cores or node0_cores[0 : self.num_workers]
                compute_cores = [
                    cpu for cpu in node0_cores if cpu not in loader_cores
                ]

            try:
                psutil.Process().cpu_affinity(compute_cores)
                set_num_threads(len(compute_cores))
                self.worker_init_fn = init_fn

                self.cpu_affinity_enabled = True
                if verbose:
                    print(
                        f"{self.num_workers} DL workers are assigned to cpus "
                        f"{loader_cores}, main process will use cpus "
                        f"{compute_cores}"
                    )

                yield
            finally:
                # restore omp_num_threads and cpu affinity
                psutil.Process().cpu_affinity(affinity_old)
                set_num_threads(nthreads_old)
                self.worker_init_fn = worker_init_fn_old

                self.cpu_affinity_enabled = False
        else:
            yield

    # To allow data other than node/edge data to be prefetched.
    def attach_data(self, name, data):
        """Add a data other than node and edge features for prefetching."""
        self.other_storages[name] = wrap_storage(data)


######## Graph DataLoaders ########
# GraphDataLoader loads a set of graphs so it's not relevant to the above.  They are currently
# copied from the old DataLoader implementation.


def _create_dist_sampler(dataset, dataloader_kwargs, ddp_seed):
    # Note: will change the content of dataloader_kwargs
    dist_sampler_kwargs = {"shuffle": dataloader_kwargs.get("shuffle", False)}
    dataloader_kwargs["shuffle"] = False
    dist_sampler_kwargs["seed"] = ddp_seed
    dist_sampler_kwargs["drop_last"] = dataloader_kwargs.get("drop_last", False)
    dataloader_kwargs["drop_last"] = False

    return DistributedSampler(dataset, **dist_sampler_kwargs)


class GraphCollator(object):
    """Given a set of graphs as well as their graph-level data, the collate function will batch the
    graphs into a batched graph, and stack the tensors into a single bigger tensor.  If the
    example is a container (such as sequences or mapping), the collate function preserves
    the structure and collates each of the elements recursively.

    If the set of graphs has no graph-level data, the collate function will yield a batched graph.

    Examples
    --------
    To train a GNN for graph classification on a set of graphs in ``dataset`` (assume
    the backend is PyTorch):

    >>> dataloader = dgl.dataloading.GraphDataLoader(
    ...     dataset, batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for batched_graph, labels in dataloader:
    ...     train_on(batched_graph, labels)
    """

    def __init__(self):
        self.graph_collate_err_msg_format = (
            "graph_collate: batch must contain DGLGraph, tensors, numpy arrays, "
            "numbers, dicts or lists; found {}"
        )
        self.np_str_obj_array_pattern = re.compile(r"[SaUO]")

    # This implementation is based on torch.utils.data._utils.collate.default_collate
    def collate(self, items):
        """This function is similar to ``torch.utils.data._utils.collate.default_collate``.
        It combines the sampled graphs and corresponding graph-level data
        into a batched graph and tensors.

        Parameters
        ----------
        items : list of data points or tuples
            Elements in the list are expected to have the same length.
            Each sub-element will be batched as a batched graph, or a
            batched tensor correspondingly.

        Returns
        -------
        A tuple of the batching results.
        """
        elem = items[0]
        elem_type = type(elem)
        if isinstance(elem, DGLGraph):
            batched_graphs = batch_graphs(items)
            return batched_graphs
        elif F.is_tensor(elem):
            return F.stack(items, 0)
        elif (
            elem_type.__module__ == "numpy"
            and elem_type.__name__ != "str_"
            and elem_type.__name__ != "string_"
        ):
            if (
                elem_type.__name__ == "ndarray"
                or elem_type.__name__ == "memmap"
            ):
                # array of string classes and object
                if (
                    self.np_str_obj_array_pattern.search(elem.dtype.str)
                    is not None
                ):
                    raise TypeError(
                        self.graph_collate_err_msg_format.format(elem.dtype)
                    )

                return self.collate([F.tensor(b) for b in items])
            elif elem.shape == ():  # scalars
                return F.tensor(items)
        elif isinstance(elem, float):
            return F.tensor(items, dtype=F.float64)
        elif isinstance(elem, int):
            return F.tensor(items)
        elif isinstance(elem, (str, bytes)):
            return items
        elif isinstance(elem, Mapping):
            return {key: self.collate([d[key] for d in items]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
            return elem_type(
                *(self.collate(samples) for samples in zip(*items))
            )
        elif isinstance(elem, Sequence):
            # check to make sure that the elements in batch have consistent size
            item_iter = iter(items)
            elem_size = len(next(item_iter))
            if not all(len(elem) == elem_size for elem in item_iter):
                raise RuntimeError(
                    "each element in list of batch should be of equal size"
                )
            transposed = zip(*items)
            return [self.collate(samples) for samples in transposed]

        raise TypeError(self.graph_collate_err_msg_format.format(elem_type))


class GraphDataLoader(torch.utils.data.DataLoader):
    """Batched graph data loader.

    PyTorch dataloader for batch-iterating over a set of graphs, generating the batched
    graph and corresponding label tensor (if provided) of the said minibatch.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to load graphs from.
    collate_fn : Function, default is None
        The customized collate function. Will use the default collate
        function if not given.
    use_ddp : boolean, optional
        If True, tells the DataLoader to split the training set for each
        participating process appropriately using
        :class:`torch.utils.data.distributed.DistributedSampler`.

        Overrides the :attr:`sampler` argument of :class:`torch.utils.data.DataLoader`.
    ddp_seed : int, optional
        The seed for shuffling the dataset in
        :class:`torch.utils.data.distributed.DistributedSampler`.

        Only effective when :attr:`use_ddp` is True.
    kwargs : dict
        Key-word arguments to be passed to the parent PyTorch
        :py:class:`torch.utils.data.DataLoader` class. Common arguments are:

          - ``batch_size`` (int): The number of indices in each batch.
          - ``drop_last`` (bool): Whether to drop the last incomplete batch.
          - ``shuffle`` (bool): Whether to randomly shuffle the indices at each epoch.

    Examples
    --------
    To train a GNN for graph classification on a set of graphs in ``dataset``:

    >>> dataloader = dgl.dataloading.GraphDataLoader(
    ...     dataset, batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for batched_graph, labels in dataloader:
    ...     train_on(batched_graph, labels)

    **With Distributed Data Parallel**

    If you are using PyTorch's distributed training (e.g. when using
    :mod:`torch.nn.parallel.DistributedDataParallel`), you can train the model by
    turning on the :attr:`use_ddp` option:

    >>> dataloader = dgl.dataloading.GraphDataLoader(
    ...     dataset, use_ddp=True, batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for epoch in range(start_epoch, n_epochs):
    ...     dataloader.set_epoch(epoch)
    ...     for batched_graph, labels in dataloader:
    ...         train_on(batched_graph, labels)
    """

    collator_arglist = inspect.getfullargspec(GraphCollator).args

    def __init__(
        self, dataset, collate_fn=None, use_ddp=False, ddp_seed=0, **kwargs
    ):
        collator_kwargs = {}
        dataloader_kwargs = {}
        for k, v in kwargs.items():
            if k in self.collator_arglist:
                collator_kwargs[k] = v
            else:
                dataloader_kwargs[k] = v

        self.use_ddp = use_ddp
        if use_ddp:
            self.dist_sampler = _create_dist_sampler(
                dataset, dataloader_kwargs, ddp_seed
            )
            dataloader_kwargs["sampler"] = self.dist_sampler

        if collate_fn is None and kwargs.get("batch_size", 1) is not None:
            collate_fn = GraphCollator(**collator_kwargs).collate

        super().__init__(
            dataset=dataset, collate_fn=collate_fn, **dataloader_kwargs
        )

    def set_epoch(self, epoch):
        """Sets the epoch number for the underlying sampler which ensures all replicas
        to use a different ordering for each epoch.

        Only available when :attr:`use_ddp` is True.

        Calls :meth:`torch.utils.data.distributed.DistributedSampler.set_epoch`.

        Parameters
        ----------
        epoch : int
            The epoch number.
        """
        if self.use_ddp:
            self.dist_sampler.set_epoch(epoch)
        else:
            raise DGLError("set_epoch is only available when use_ddp is True.")


class NodeCollator:
    """Deprecated. Please use :class:`~dgl.distributed.NodeCollator` instead."""

    def __new__(cls, *args, **kwargs):
        dgl_warning(
            "NodeCollator is defined in dgl.distributed This class is for "
            "backward compatibility and will be removed soon. Please update "
            "your code to use `dgl.distributed.NodeCollator`."
        )
        from ..distributed import NodeCollator as NewNodeCollator

        return NewNodeCollator(*args, **kwargs)


class EdgeCollator:
    """Deprecated. Please use :class:`~dgl.distributed.EdgeCollator` instead."""

    def __new__(cls, *args, **kwargs):
        dgl_warning(
            "EdgeCollator is defined in dgl.distributed This class is for "
            "backward compatibility and will be removed soon. Please update "
            "your code to use `dgl.distributed.EdgeCollator`."
        )
        from ..distributed import EdgeCollator as NewEdgeCollator

        return NewEdgeCollator(*args, **kwargs)


def _remove_kwargs_dist(kwargs):
    """Deprecated."""
    if "num_workers" in kwargs:
        del kwargs["num_workers"]
    if "pin_memory" in kwargs:
        del kwargs["pin_memory"]
        print("Distributed DataLoaders do not support pin_memory.")
    return kwargs


class DistDataLoader:
    """Deprecated. Please use :class:`~dgl.distributed.DistDataLoader` instead."""

    def __new__(cls, *args, **kwargs):
        dgl_warning(
            "DistDataLoader is defined in dgl.distributed This class is for "
            "backward compatibility and will be removed soon. Please update "
            "your code to use `dgl.distributed.DistDataLoader`."
        )
        from ..distributed import DistDataLoader as NewDistDataLoader

        return NewDistDataLoader(*args, **kwargs)


class DistNodeDataLoader:
    """Deprecated. Please use :class:`~dgl.distributed.DistNodeDataLoader`
    instead.
    """

    def __new__(cls, *args, **kwargs):
        dgl_warning(
            "dgl.dataloading.DistNodeDataLoader has been moved to "
            "dgl.distributed.DistNodeDataLoader. This old class is deprecated "
            "and will be removed soon. Please update your code to use the new "
            "class."
        )
        from ..distributed import DistNodeDataLoader as NewDistNodeDataLoader

        return NewDistNodeDataLoader(*args, **kwargs)


class DistEdgeDataLoader:
    """Deprecated. Please use :class:`~dgl.distributed.DistEdgeDataLoader`
    instead.
    """

    def __new__(cls, *args, **kwargs):
        dgl_warning(
            "dgl.dataloading.DistEdgeDataLoader has been moved to "
            "dgl.distributed.DistEdgeDataLoader. This old class is deprecated "
            "and will be removed soon. Please update your code to use the new "
            "class."
        )
        from ..distributed import DistEdgeDataLoader as NewDistEdgeDataLoader

        return NewDistEdgeDataLoader(*args, **kwargs)
