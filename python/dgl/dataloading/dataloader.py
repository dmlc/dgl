"""DGL PyTorch DataLoaders"""
from collections.abc import Mapping, Sequence
from queue import Queue, Empty, Full
import itertools
import threading
from distutils.version import LooseVersion
import random
import math
import inspect
import re
import atexit
import os

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from ..base import NID, EID, dgl_warning
from ..batch import batch as batch_graphs
from ..heterograph import DGLHeteroGraph
from .. import ndarray as nd
from ..utils import (
    recursive_apply, ExceptionWrapper, recursive_apply_pair, set_num_threads,
    create_shared_mem_array, get_shared_mem_array, context_of, pin_memory_inplace)
from ..frame import LazyFeature
from ..storages import wrap_storage
from .base import BlockSampler, as_edge_prediction_sampler
from .. import backend as F
from ..distributed import DistGraph

PYTHON_EXIT_STATUS = False
def _set_python_exit_flag():
    global PYTHON_EXIT_STATUS
    PYTHON_EXIT_STATUS = True
atexit.register(_set_python_exit_flag)

prefetcher_timeout = int(os.environ.get('DGL_PREFETCHER_TIMEOUT', '30'))

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
            # clone() fixes #3755, probably.  Not sure why.  Need to take a look afterwards.
            return batch.clone()

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
            self.mapping_keys[type_id_uniq[i]]:
                indices[type_id_offset[i]:type_id_offset[i+1]].clone()
            for i in range(len(type_id_uniq))}
        return id_dict


def _get_id_tensor_from_mapping(indices, device, keys):
    lengths = torch.LongTensor([
        (indices[k].shape[0] if k in indices else 0) for k in keys]).to(device)
    type_ids = torch.arange(len(keys), device=device).repeat_interleave(lengths)
    all_indices = torch.cat([indices[k] for k in keys if k in indices])
    return torch.stack([type_ids, all_indices], 1)


def _divide_by_worker(dataset, batch_size, drop_last):
    num_samples = dataset.shape[0]
    worker_info = torch.utils.data.get_worker_info()
    if worker_info:
        num_batches = (num_samples + (0 if drop_last else batch_size - 1)) // batch_size
        num_batches_per_worker = num_batches // worker_info.num_workers
        left_over = num_batches % worker_info.num_workers
        start = (num_batches_per_worker * worker_info.id) + min(left_over, worker_info.id)
        end = start + num_batches_per_worker + (worker_info.id < left_over)
        start *= batch_size
        end = min(end * batch_size, num_samples)
        dataset = dataset[start:end]
    return dataset


class TensorizedDataset(torch.utils.data.IterableDataset):
    """Custom Dataset wrapper that returns a minibatch as tensors or dicts of tensors.
    When the dataset is on the GPU, this significantly reduces the overhead.
    """
    def __init__(self, indices, batch_size, drop_last):
        name, _ = _generate_shared_mem_name_id()
        if isinstance(indices, Mapping):
            self._mapping_keys = list(indices.keys())
            self._device = next(iter(indices.values())).device
            self._id_tensor = _get_id_tensor_from_mapping(
                indices, self._device, self._mapping_keys)
        else:
            self._id_tensor = indices
            self._device = indices.device
            self._mapping_keys = None
        # Use a shared memory array to permute indices for shuffling.  This is to make sure that
        # the worker processes can see it when persistent_workers=True, where self._indices
        # would not be duplicated every epoch.
        self._indices = create_shared_mem_array(name, (self._id_tensor.shape[0],), torch.int64)
        self._indices[:] = torch.arange(self._id_tensor.shape[0])
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shared_mem_name = name
        self.shared_mem_size = self._indices.shape[0]

    def shuffle(self):
        """Shuffle the dataset."""
        # TODO: may need an in-place shuffle kernel
        self._indices[:] = self._indices[torch.randperm(self._indices.shape[0])]

    def __iter__(self):
        indices = _divide_by_worker(self._indices, self.batch_size, self.drop_last)
        id_tensor = self._id_tensor[indices.to(self._device)]
        return _TensorizedDatasetIter(
            id_tensor, self.batch_size, self.drop_last, self._mapping_keys)

    def __len__(self):
        num_samples = self._id_tensor.shape[0]
        return (num_samples + (0 if self.drop_last else (self.batch_size - 1))) // self.batch_size

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
    """Custom Dataset wrapper that returns a minibatch as tensors or dicts of tensors.
    When the dataset is on the GPU, this significantly reduces the overhead.

    This class additionally saves the index tensor in shared memory and therefore
    avoids duplicating the same index tensor during shuffling.
    """
    def __init__(self, indices, batch_size, drop_last, ddp_seed):
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

        if self.drop_last and len_indices % self.num_replicas != 0:
            self.num_samples = math.ceil((len_indices - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len_indices / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        # If drop_last is True, we create a shared memory array larger than the number
        # of indices since we will need to pad it after shuffling to make it evenly
        # divisible before every epoch.  If drop_last is False, we create an array
        # with the same size as the indices so we can trim it later.
        self.shared_mem_size = self.total_size if not self.drop_last else len_indices
        self.num_indices = len_indices

        if isinstance(indices, Mapping):
            self._device = next(iter(indices.values())).device
            self._id_tensor = _get_id_tensor_from_mapping(
                    indices, self._device, self._mapping_keys)
        else:
            self._id_tensor = indices
            self._device = self._id_tensor.device

        if self.rank == 0:
            name, id_ = _generate_shared_mem_name_id()
            self._indices = create_shared_mem_array(
                name, (self.shared_mem_size,), torch.int64)
            self._indices[:self._id_tensor.shape[0]] = torch.arange(self._id_tensor.shape[0])
            meta_info = torch.LongTensor([id_, self._indices.shape[0]])
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
            indices_shared = get_shared_mem_array(name, (num_samples,), torch.int64)
            self._indices = indices_shared
        self.shared_mem_name = name

    def shuffle(self):
        """Shuffles the dataset."""
        # Only rank 0 does the actual shuffling.  The other ranks wait for it.
        if self.rank == 0:
            self._indices[:self.num_indices] = self._indices[
                torch.randperm(self.num_indices, device=self._device)]
            if not self.drop_last:
                # pad extra
                self._indices[self.num_indices:] = \
                    self._indices[:self.total_size - self.num_indices]
        dist.barrier()

    def __iter__(self):
        start = self.num_samples * self.rank
        end = self.num_samples * (self.rank + 1)
        indices = _divide_by_worker(self._indices[start:end], self.batch_size, self.drop_last)
        id_tensor = self._id_tensor[indices.to(self._device)]
        return _TensorizedDatasetIter(
            id_tensor, self.batch_size, self.drop_last, self._mapping_keys)

    def __len__(self):
        return (self.num_samples + (0 if self.drop_last else (self.batch_size - 1))) // \
            self.batch_size


def _prefetch_update_feats(feats, frames, types, get_storage_func, id_name, device, pin_prefetcher):
    for tid, frame in enumerate(frames):
        type_ = types[tid]
        default_id = frame.get(id_name, None)
        for key in frame.keys():
            column = frame._columns[key]
            if isinstance(column, LazyFeature):
                parent_key = column.name or key
                if column.id_ is None and default_id is None:
                    raise DGLError(
                        'Found a LazyFeature with no ID specified, '
                        'and the graph does not have dgl.NID or dgl.EID columns')
                feats[tid, key] = get_storage_func(parent_key, type_).fetch(
                    column.id_ or default_id, device, pin_prefetcher)


# This class exists to avoid recursion into the feature dictionary returned by the
# prefetcher when calling recursive_apply().
class _PrefetchedGraphFeatures(object):
    __slots__ = ['node_feats', 'edge_feats']
    def __init__(self, node_feats, edge_feats):
        self.node_feats = node_feats
        self.edge_feats = edge_feats


def _prefetch_for_subgraph(subg, dataloader):
    node_feats, edge_feats = {}, {}
    _prefetch_update_feats(
        node_feats, subg._node_frames, subg.ntypes, dataloader.graph.get_node_storage,
        NID, dataloader.device, dataloader.pin_prefetcher)
    _prefetch_update_feats(
        edge_feats, subg._edge_frames, subg.canonical_etypes, dataloader.graph.get_edge_storage,
        EID, dataloader.device, dataloader.pin_prefetcher)
    return _PrefetchedGraphFeatures(node_feats, edge_feats)


def _prefetch_for(item, dataloader):
    if isinstance(item, DGLHeteroGraph):
        return _prefetch_for_subgraph(item, dataloader)
    elif isinstance(item, LazyFeature):
        return dataloader.other_storages[item.name].fetch(
            item.id_, dataloader.device, dataloader.pin_prefetcher)
    else:
        return None


def _await_or_return(x):
    if hasattr(x, 'wait'):
        return x.wait()
    elif isinstance(x, _PrefetchedGraphFeatures):
        node_feats = recursive_apply(x.node_feats, _await_or_return)
        edge_feats = recursive_apply(x.edge_feats, _await_or_return)
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
    with torch.cuda.stream(stream):
        feats = recursive_apply(batch, _prefetch_for, dataloader)
        feats = recursive_apply(feats, _await_or_return)
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

def _put_if_event_not_set(queue, result, event):
    while not event.is_set():
        try:
            queue.put(result, timeout=1.0)
            break
        except Full:
            continue

def _prefetcher_entry(
        dataloader_it, dataloader, queue, num_threads, use_alternate_streams,
        done_event):
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
        while not done_event.is_set():
            try:
                batch = next(dataloader_it)
            except StopIteration:
                break
            batch = recursive_apply(batch, restore_parent_storage_columns, dataloader.graph)
            feats = _prefetch(batch, dataloader, stream)

            _put_if_event_not_set(queue, (
                # batch will be already in pinned memory as per the behavior of
                # PyTorch DataLoader.
                recursive_apply(
                    batch, lambda x: x.to(dataloader.device, non_blocking=True)),
                feats,
                stream.record_event() if stream is not None else None,
                None),
                done_event)
        _put_if_event_not_set(queue, (None, None, None, None), done_event)
    except:     # pylint: disable=bare-except
        _put_if_event_not_set(
            queue, (None, None, None, ExceptionWrapper(where='in prefetcher')), done_event)


# DGLHeteroGraphs have the semantics of lazy feature slicing with subgraphs.  Such behavior depends
# on that DGLHeteroGraph's ndata and edata are maintained by Frames.  So to maintain compatibility
# with older code, DGLHeteroGraphs and other graph storages are handled separately: (1)
# DGLHeteroGraphs will preserve the lazy feature slicing for subgraphs.  (2) Other graph storages
# will not have lazy feature slicing; all feature slicing will be eager.
def remove_parent_storage_columns(item, g):
    """Removes the storage objects in the given graphs' Frames if it is a sub-frame of the
    given parent graph, so that the storages are not serialized during IPC from PyTorch
    DataLoader workers.
    """
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
    """Restores the storage objects in the given graphs' Frames if it is a sub-frame of the
    given parent graph (i.e. when the storage object is None).
    """
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
        self.pin_prefetcher = self.dataloader.pin_prefetcher
        self.num_threads = num_threads

        self.use_thread = use_thread
        self.use_alternate_streams = use_alternate_streams
        self._shutting_down = False
        if use_thread:
            self._done_event = threading.Event()
            thread = threading.Thread(
                target=_prefetcher_entry,
                args=(dataloader_it, dataloader, self.queue, num_threads,
                      use_alternate_streams, self._done_event),
                daemon=True)
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
                    self.queue.get_nowait()     # In case the thread is blocking on put().
                except:     # pylint: disable=bare-except
                    pass

                self.thread.join()
            except:         # pylint: disable=bare-except
                pass

    def __del__(self):
        if self.use_thread:
            self._shutdown()

    def _next_non_threaded(self):
        batch = next(self.dataloader_it)
        batch = recursive_apply(batch, restore_parent_storage_columns, self.dataloader.graph)
        device = self.dataloader.device
        if self.use_alternate_streams:
            stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        else:
            stream = None
        feats = _prefetch(batch, self.dataloader, stream)
        batch = recursive_apply(batch, lambda x: x.to(device, non_blocking=True))
        stream_event = stream.record_event() if stream is not None else None
        return batch, feats, stream_event

    def _next_threaded(self):
        try:
            batch, feats, stream_event, exception = self.queue.get(timeout=prefetcher_timeout)
        except Empty:
            raise RuntimeError(
                f'Prefetcher thread timed out at {prefetcher_timeout} seconds.')
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
    """Wraps a collate function with :func:`remove_parent_storage_columns` for serializing
    from PyTorch DataLoader workers.
    """
    def __init__(self, sample_func, g):
        self.sample_func = sample_func
        self.g = g

    def __call__(self, items):
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


def create_tensorized_dataset(indices, batch_size, drop_last, use_ddp, ddp_seed):
    """Converts a given indices tensor to a TensorizedDataset, an IterableDataset
    that returns views of the original tensor, to reduce overhead from having
    a list of scalar tensors in default PyTorch DataLoader implementation.
    """
    if use_ddp:
        return DDPTensorizedDataset(indices, batch_size, drop_last, ddp_seed)
    else:
        return TensorizedDataset(indices, batch_size, drop_last)


def _get_device(device):
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return device

class DataLoader(torch.utils.data.DataLoader):
    """PyTorch dataloader for batch-iterating over a set of nodes, edges or any other
    kinds of indices.  The minibatch of such indices will be then passed to a sampler
    generating subgraphs, message flow graphs (MFGs), or any other structures necessary
    to compute the representations.

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

        By default this value is the same as the device of :attr:`g`.
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
    batch_size : int, optional
    drop_last : bool, optional
    shuffle : bool, optional
    kwargs : dict
        Arguments being passed to :py:class:`torch.utils.data.DataLoader`.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from all neighbors (assume
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
    def __init__(self, graph, indices, graph_sampler, device='cpu', use_ddp=False,
                 ddp_seed=0, batch_size=1, drop_last=False, shuffle=False,
                 use_prefetch_thread=None, use_alternate_streams=None,
                 pin_prefetcher=None, use_uva=False, **kwargs):
        if isinstance(graph, DistGraph):
            raise TypeError(
                'Please use dgl.dataloading.DistNodeDataLoader or '
                'dgl.datalaoding.DistEdgeDataLoader for DistGraphs.')
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
        self.indices = indices      # For PyTorch-Lightning
        num_workers = kwargs.get('num_workers', 0)

        try:
            if isinstance(indices, Mapping):
                indices = {k: (torch.tensor(v) if not torch.is_tensor(v) else v)
                           for k, v in indices.items()}
                indices_device = next(iter(indices.values())).device
            else:
                indices = torch.tensor(indices) if not torch.is_tensor(indices) else indices
                indices_device = indices.device
        except:     # pylint: disable=bare-except
            # ignore when it fails to convert to torch Tensors.
            pass

        self.device = _get_device(device)

        # Sanity check - we only check for DGLGraphs.
        if isinstance(self.graph, DGLHeteroGraph):
            # Check graph and indices device as well as num_workers
            if use_uva:
                if self.graph.device.type != 'cpu':
                    raise ValueError('Graph must be on CPU if UVA sampling is enabled.')
                if indices_device != self.device:
                    raise ValueError(
                        f'Indices must be on the same device as the device argument '
                        f'({self.device})')
                if num_workers > 0:
                    raise ValueError('num_workers must be 0 if UVA sampling is enabled.')

                # Create all the formats and pin the features - custom GraphStorages
                # will need to do that themselves.
                self.graph.create_formats_()
                self.graph.pin_memory_()
                for frame in itertools.chain(self.graph._node_frames, self.graph._edge_frames):
                    for col in frame._columns.values():
                        pin_memory_inplace(col.data)
            else:
                if self.graph.device != indices_device:
                    raise ValueError(
                        'Expect graph and indices to be on the same device. '
                        'If you wish to use UVA sampling, please set use_uva=True.')
                if self.graph.device.type == 'cuda' and num_workers > 0:
                    raise ValueError('num_workers must be 0 if graph and indices are on CUDA.')
                if self.graph.device.type == 'cpu' and num_workers > 0:
                    # Instantiate all the formats if the number of workers is greater than 0.
                    self.graph.create_formats_()

            # Check pin_prefetcher and use_prefetch_thread - should be only effective
            # if performing CPU sampling but output device is CUDA
            if self.device.type == 'cuda' and self.graph.device.type == 'cpu' and not use_uva:
                if pin_prefetcher is None:
                    pin_prefetcher = True
                if use_prefetch_thread is None:
                    use_prefetch_thread = True
            else:
                if pin_prefetcher is True:
                    raise ValueError(
                        'pin_prefetcher=True is only effective when device=cuda and '
                        'sampling is performed on CPU.')
                if pin_prefetcher is None:
                    pin_prefetcher = False

                if use_prefetch_thread is True:
                    raise ValueError(
                        'use_prefetch_thread=True is only effective when device=cuda and '
                        'sampling is performed on CPU.')
                if use_prefetch_thread is None:
                    use_prefetch_thread = False

            # Check use_alternate_streams
            if use_alternate_streams is None:
                use_alternate_streams = (
                    self.device.type == 'cuda' and self.graph.device.type == 'cpu' and
                    not use_uva)

        if (torch.is_tensor(indices) or (
                isinstance(indices, Mapping) and
                all(torch.is_tensor(v) for v in indices.values()))):
            self.dataset = create_tensorized_dataset(
                indices, batch_size, drop_last, use_ddp, ddp_seed)
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
        worker_init_fn = WorkerInitWrapper(kwargs.get('worker_init_fn', None))

        self.other_storages = {}

        super().__init__(
            self.dataset,
            collate_fn=CollateWrapper(self.graph_sampler.sample, graph),
            batch_size=None,
            worker_init_fn=worker_init_fn,
            **kwargs)

    def __iter__(self):
        if self.shuffle:
            self.dataset.shuffle()
        # When using multiprocessing PyTorch sometimes set the number of PyTorch threads to 1
        # when spawning new Python threads.  This drastically slows down pinning features.
        num_threads = torch.get_num_threads() if self.num_workers > 0 else None
        return _PrefetchingIter(
            self, super().__iter__(), use_thread=self.use_prefetch_thread,
            use_alternate_streams=self.use_alternate_streams, num_threads=num_threads)

    # To allow data other than node/edge data to be prefetched.
    def attach_data(self, name, data):
        """Add a data other than node and edge features for prefetching."""
        self.other_storages[name] = wrap_storage(data)


# Alias
class NodeDataLoader(DataLoader):
    """PyTorch dataloader for batch-iterating over a set of nodes, generating the list
    of message flow graphs (MFGs) as computation dependency of the said minibatch.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    indices : Tensor or dict[ntype, Tensor]
        The node set to compute outputs.
    graph_sampler : dgl.dataloading.Sampler
        The subgraph sampler.
    device : device context, optional
        The device of the generated MFGs in each iteration, which should be a
        PyTorch device object (e.g., ``torch.device``).

        By default this value is the same as the device of :attr:`g`.
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
    batch_size : int, optional
    drop_last : bool, optional
    shuffle : bool, optional
    kwargs : dict
        Arguments being passed to :py:class:`torch.utils.data.DataLoader`.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from all neighbors (assume
    the backend is PyTorch):

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> dataloader = dgl.dataloading.NodeDataLoader(
    ...     g, train_nid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     train_on(input_nodes, output_nodes, blocks)

    **Using with Distributed Data Parallel**

    If you are using PyTorch's distributed training (e.g. when using
    :mod:`torch.nn.parallel.DistributedDataParallel`), you can train the model by turning
    on the `use_ddp` option:

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> dataloader = dgl.dataloading.NodeDataLoader(
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


class EdgeDataLoader(DataLoader):
    """PyTorch dataloader for batch-iterating over a set of edges, generating the list
    of message flow graphs (MFGs) as computation dependency of the said minibatch for
    edge classification, edge regression, and link prediction.

    For more details, please refer to :ref:`guide-minibatch-edge-classification-sampler`
    and :ref:`guide-minibatch-link-classification-sampler`.

    Parameters
    ----------
    g : DGLGraph
        The graph.
    indices : Tensor or dict[etype, Tensor]
        The edge set in graph :attr:`g` to compute outputs.
    graph_sampler : dgl.dataloading.Sampler
        The subgraph sampler
    device : device context, optional
        The device of the generated MFGs and graphs in each iteration, which should be a
        PyTorch device object (e.g., ``torch.device``).

        By default this value is the same as the device of :attr:`g`.
    use_ddp : boolean, optional
        If True, tells the DataLoader to split the training set for each
        participating process appropriately using
        :class:`torch.utils.data.distributed.DistributedSampler`.

        Overrides the :attr:`sampler` argument of :class:`torch.utils.data.DataLoader`.
    ddp_seed : int, optional
        The seed for shuffling the dataset in
        :class:`torch.utils.data.distributed.DistributedSampler`.

        Only effective when :attr:`use_ddp` is True.
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
    use_uva : bool, optional
        Whether to use Unified Virtual Addressing (UVA) to directly sample the graph
        and slice the features from CPU into GPU.  Setting it to True will pin the
        graph and feature tensors into pinned memory.

        If True, requires that :attr:`indices` must have the same device as the
        :attr:`device` argument.

        Default: False.
    exclude : str, optional
    reverse_eids : Tensor or dict[etype, Tensor], optional
    reverse_etypes : dict[etype, etype], optional
    negative_sampler : callable, optional
        Deprecated and will be passed to :func:`dgl.dataloading.as_edge_prediction_sampler`.
    batch_size : int, optional
    drop_last : bool, optional
    shuffle : bool, optional
    kwargs : dict
        Arguments being passed to :py:class:`torch.utils.data.DataLoader`.

    Examples
    --------
    The following example shows how to train a 3-layer GNN for edge classification on a
    set of edges ``train_eid`` on a homogeneous undirected graph. Each node takes
    messages from all neighbors.

    Say that you have an array of source node IDs ``src`` and another array of destination
    node IDs ``dst``.  One can make it bidirectional by adding another set of edges
    that connects from ``dst`` to ``src``:

    >>> g = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])))

    One can then know that the ID difference of an edge and its reverse edge is ``|E|``,
    where ``|E|`` is the length of your source/destination array.  The reverse edge
    mapping can be obtained by

    >>> E = len(src)
    >>> reverse_eids = torch.cat([torch.arange(E, 2 * E), torch.arange(0, E)])

    Note that the sampled edges as well as their reverse edges are removed from
    computation dependencies of the incident nodes.  That is, the edge will not
    involve in neighbor sampling and message aggregation.  This is a common trick
    to avoid information leakage.

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> dataloader = dgl.dataloading.EdgeDataLoader(
    ...     g, train_eid, sampler, exclude='reverse_id',
    ...     reverse_eids=reverse_eids,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pair_graph, blocks in dataloader:
    ...     train_on(input_nodes, pair_graph, blocks)

    To train a 3-layer GNN for link prediction on a set of edges ``train_eid`` on a
    homogeneous graph where each node takes messages from all neighbors (assume the
    backend is PyTorch), with 5 uniformly chosen negative samples per edge:

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> neg_sampler = dgl.dataloading.negative_sampler.Uniform(5)
    >>> dataloader = dgl.dataloading.EdgeDataLoader(
    ...     g, train_eid, sampler, exclude='reverse_id',
    ...     reverse_eids=reverse_eids, negative_sampler=neg_sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pos_pair_graph, neg_pair_graph, blocks in dataloader:
    ...     train_on(input_nodse, pair_graph, neg_pair_graph, blocks)

    For heterogeneous graphs, the reverse of an edge may have a different edge type
    from the original edge.  For instance, consider that you have an array of
    user-item clicks, representated by a user array ``user`` and an item array ``item``.
    You may want to build a heterogeneous graph with a user-click-item relation and an
    item-clicked-by-user relation.

    >>> g = dgl.heterograph({
    ...     ('user', 'click', 'item'): (user, item),
    ...     ('item', 'clicked-by', 'user'): (item, user)})

    To train a 3-layer GNN for edge classification on a set of edges ``train_eid`` with
    type ``click``, you can write

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> dataloader = dgl.dataloading.EdgeDataLoader(
    ...     g, {'click': train_eid}, sampler, exclude='reverse_types',
    ...     reverse_etypes={'click': 'clicked-by', 'clicked-by': 'click'},
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pair_graph, blocks in dataloader:
    ...     train_on(input_nodes, pair_graph, blocks)

    To train a 3-layer GNN for link prediction on a set of edges ``train_eid`` with type
    ``click``, you can write

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> neg_sampler = dgl.dataloading.negative_sampler.Uniform(5)
    >>> dataloader = dgl.dataloading.EdgeDataLoader(
    ...     g, train_eid, sampler, exclude='reverse_types',
    ...     reverse_etypes={'click': 'clicked-by', 'clicked-by': 'click'},
    ...     negative_sampler=neg_sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pos_pair_graph, neg_pair_graph, blocks in dataloader:
    ...     train_on(input_nodes, pair_graph, neg_pair_graph, blocks)

    **Using with Distributed Data Parallel**

    If you are using PyTorch's distributed training (e.g. when using
    :mod:`torch.nn.parallel.DistributedDataParallel`), you can train the model by
    turning on the :attr:`use_ddp` option:

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> dataloader = dgl.dataloading.EdgeDataLoader(
    ...     g, train_eid, sampler, use_ddp=True, exclude='reverse_id',
    ...     reverse_eids=reverse_eids,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for epoch in range(start_epoch, n_epochs):
    ...     for input_nodes, pair_graph, blocks in dataloader:
    ...         train_on(input_nodes, pair_graph, blocks)

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
    def __init__(self, graph, indices, graph_sampler, device='cpu', use_ddp=False,
                 ddp_seed=0, batch_size=1, drop_last=False, shuffle=False,
                 use_prefetch_thread=False, use_alternate_streams=True,
                 pin_prefetcher=False,
                 exclude=None, reverse_eids=None, reverse_etypes=None, negative_sampler=None,
                 use_uva=False, **kwargs):
        device = _get_device(device)

        if isinstance(graph_sampler, BlockSampler):
            dgl_warning(
                'EdgeDataLoader directly taking a BlockSampler will be deprecated '
                'and it will not support feature prefetching. '
                'Please use dgl.dataloading.as_edge_prediction_sampler to wrap it.')
            if reverse_eids is not None:
                if use_uva:
                    reverse_eids = recursive_apply(reverse_eids, lambda x: x.to(device))
                else:
                    reverse_eids_device = context_of(reverse_eids)
                    indices_device = context_of(indices)
                    if indices_device != reverse_eids_device:
                        raise ValueError('Expect the same device for indices and reverse_eids')
            graph_sampler = as_edge_prediction_sampler(
                graph_sampler, exclude=exclude, reverse_eids=reverse_eids,
                reverse_etypes=reverse_etypes, negative_sampler=negative_sampler)

        super().__init__(
            graph, indices, graph_sampler, device=device, use_ddp=use_ddp, ddp_seed=ddp_seed,
            batch_size=batch_size, drop_last=drop_last, shuffle=shuffle,
            use_prefetch_thread=use_prefetch_thread, use_alternate_streams=use_alternate_streams,
            pin_prefetcher=pin_prefetcher, use_uva=use_uva,
            **kwargs)


######## Graph DataLoaders ########
# GraphDataLoader loads a set of graphs so it's not relevant to the above.  They are currently
# copied from the old DataLoader implementation.

PYTORCH_VER = LooseVersion(torch.__version__)
PYTORCH_16 = PYTORCH_VER >= LooseVersion("1.6.0")
PYTORCH_17 = PYTORCH_VER >= LooseVersion("1.7.0")

def _create_dist_sampler(dataset, dataloader_kwargs, ddp_seed):
    # Note: will change the content of dataloader_kwargs
    dist_sampler_kwargs = {'shuffle': dataloader_kwargs.get('shuffle', False)}
    dataloader_kwargs['shuffle'] = False
    if PYTORCH_16:
        dist_sampler_kwargs['seed'] = ddp_seed
    if PYTORCH_17:
        dist_sampler_kwargs['drop_last'] = dataloader_kwargs.get('drop_last', False)
        dataloader_kwargs['drop_last'] = False

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
            "numbers, dicts or lists; found {}")
        self.np_str_obj_array_pattern = re.compile(r'[SaUO]')

    #This implementation is based on torch.utils.data._utils.collate.default_collate
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
        if isinstance(elem, DGLHeteroGraph):
            batched_graphs = batch_graphs(items)
            return batched_graphs
        elif F.is_tensor(elem):
            return F.stack(items, 0)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if self.np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(self.graph_collate_err_msg_format.format(elem.dtype))

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
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.collate(samples) for samples in zip(*items)))
        elif isinstance(elem, Sequence):
            # check to make sure that the elements in batch have consistent size
            item_iter = iter(items)
            elem_size = len(next(item_iter))
            if not all(len(elem) == elem_size for elem in item_iter):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = zip(*items)
            return [self.collate(samples) for samples in transposed]

        raise TypeError(self.graph_collate_err_msg_format.format(elem_type))

class GraphDataLoader(torch.utils.data.DataLoader):
    """PyTorch dataloader for batch-iterating over a set of graphs, generating the batched
    graph and corresponding label tensor (if provided) of the said minibatch.

    Parameters
    ----------
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
        Arguments being passed to :py:class:`torch.utils.data.DataLoader`.

    Examples
    --------
    To train a GNN for graph classification on a set of graphs in ``dataset`` (assume
    the backend is PyTorch):

    >>> dataloader = dgl.dataloading.GraphDataLoader(
    ...     dataset, batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for batched_graph, labels in dataloader:
    ...     train_on(batched_graph, labels)

    **Using with Distributed Data Parallel**

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

    def __init__(self, dataset, collate_fn=None, use_ddp=False, ddp_seed=0, **kwargs):
        collator_kwargs = {}
        dataloader_kwargs = {}
        for k, v in kwargs.items():
            if k in self.collator_arglist:
                collator_kwargs[k] = v
            else:
                dataloader_kwargs[k] = v

        if collate_fn is None:
            self.collate = GraphCollator(**collator_kwargs).collate
        else:
            self.collate = collate_fn

        self.use_ddp = use_ddp
        if use_ddp:
            self.dist_sampler = _create_dist_sampler(dataset, dataloader_kwargs, ddp_seed)
            dataloader_kwargs['sampler'] = self.dist_sampler

        super().__init__(dataset=dataset, collate_fn=self.collate, **dataloader_kwargs)

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
            raise DGLError('set_epoch is only available when use_ddp is True.')
