"""DGL PyTorch DataLoaders"""
from collections.abc import Mapping, Sequence
from queue import Queue
import itertools
import threading
from distutils.version import LooseVersion
import random
import math
import inspect
import re

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from ..base import NID, EID, dgl_warning
from ..batch import batch as batch_graphs
from ..heterograph import DGLHeteroGraph
from .. import ndarray as nd
from ..utils import (
    recursive_apply, ExceptionWrapper, recursive_apply_pair, set_num_threads,
    create_shared_mem_array, get_shared_mem_array)
from ..frame import LazyFeature
from ..storages import wrap_storage
from .base import BlockSampler, EdgeBlockSampler
from .. import backend as F

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
    """
    def __init__(self, indices, batch_size, drop_last):
        if isinstance(indices, Mapping):
            self._mapping_keys = list(indices.keys())
            self._device = next(iter(indices.values())).device
            self._tensor_dataset = _get_id_tensor_from_mapping(
                indices, self._device, self._mapping_keys)
        else:
            self._tensor_dataset = indices
            self._device = indices.device
            self._mapping_keys = None
        self.batch_size = batch_size
        self.drop_last = drop_last

    def shuffle(self):
        """Shuffle the dataset."""
        # TODO: may need an in-place shuffle kernel
        perm = torch.randperm(self._tensor_dataset.shape[0], device=self._device)
        self._tensor_dataset[:] = self._tensor_dataset[perm]

    def __iter__(self):
        dataset = _divide_by_worker(self._tensor_dataset)
        return _TensorizedDatasetIter(
            dataset, self.batch_size, self.drop_last, self._mapping_keys)

    def __len__(self):
        num_samples = self._tensor_dataset.shape[0]
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
        else:
            self._mapping_keys = None

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
        self.shared_mem_size = self.total_size if not self.drop_last else len(indices)
        self.num_indices = len(indices)

        if self.rank == 0:
            name, id_ = _generate_shared_mem_name_id()
            if isinstance(indices, Mapping):
                device = next(iter(indices.values())).device
                id_tensor = _get_id_tensor_from_mapping(indices, device, self._mapping_keys)
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

    def shuffle(self):
        """Shuffles the dataset."""
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

    def __len__(self):
        return (self.num_samples + (0 if self.drop_last else (self.batch_size - 1))) // \
            self.batch_size


def _prefetch_update_feats(feats, frames, types, get_storage_func, id_name, device, pin_memory):
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
                feats[tid, key] = get_storage_func(parent_key, type_).fetch(
                    column.id_ or default_id, device, pin_memory)


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
        NID, dataloader.device, dataloader.pin_memory)
    _prefetch_update_feats(
        edge_feats, subg._edge_frames, subg.canonical_etypes, dataloader.graph.get_edge_storage,
        EID, dataloader.device, dataloader.pin_memory)
    return _PrefetchedGraphFeatures(node_feats, edge_feats)


def _prefetch_for(item, dataloader):
    if isinstance(item, DGLHeteroGraph):
        return _prefetch_for_subgraph(item, dataloader)
    elif isinstance(item, LazyFeature):
        return dataloader.other_storages[item.name].fetch(
            item.id_, dataloader.device, dataloader.pin_memory)
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


def _prefetcher_entry(dataloader_it, dataloader, queue, num_threads, use_alternate_streams):
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


class DataLoader(torch.utils.data.DataLoader):
    """DataLoader class."""
    def __init__(self, graph, indices, graph_sampler, device='cpu', use_ddp=False,
                 ddp_seed=0, batch_size=1, drop_last=False, shuffle=False,
                 use_prefetch_thread=False, use_alternate_streams=True, **kwargs):
        self.graph = graph

        try:
            if isinstance(indices, Mapping):
                indices = {k: (torch.tensor(v) if not torch.is_tensor(v) else v)
                           for k, v in indices.items()}
            else:
                indices = torch.tensor(indices) if not torch.is_tensor(indices) else indices
        except:     # pylint: disable=bare-except
            # ignore when it fails to convert to torch Tensors.
            pass

        if (torch.is_tensor(indices) or (
                isinstance(indices, Mapping) and
                all(torch.is_tensor(v) for v in indices.values()))):
            self.dataset = create_tensorized_dataset(
                indices, batch_size, drop_last, use_ddp, ddp_seed)
        else:
            self.dataset = indices

        self.ddp_seed = ddp_seed
        self._shuffle_dataset = shuffle
        self.graph_sampler = graph_sampler
        self.device = torch.device(device)
        self.use_alternate_streams = use_alternate_streams
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.use_prefetch_thread = use_prefetch_thread
        worker_init_fn = WorkerInitWrapper(kwargs.get('worker_init_fn', None))

        # Instantiate all the formats if the number of workers is greater than 0.
        if kwargs.get('num_workers', 0) > 0 and hasattr(self.graph, 'create_formats_'):
            self.graph.create_formats_()

        self.other_storages = {}

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
    def attach_data(self, name, data):
        """Add a data other than node and edge features for prefetching."""
        self.other_storages[name] = wrap_storage(data)


# Alias
class NodeDataLoader(DataLoader):
    """NodeDataLoader class."""


class EdgeDataLoader(DataLoader):
    """EdgeDataLoader class."""
    def __init__(self, graph, indices, graph_sampler, device='cpu', use_ddp=False,
                 ddp_seed=0, batch_size=1, drop_last=False, shuffle=False,
                 use_prefetch_thread=False, use_alternate_streams=True,
                 exclude=None, reverse_eids=None, reverse_etypes=None, negative_sampler=None,
                 g_sampling=None, **kwargs):
        if g_sampling is not None:
            dgl_warning(
                "g_sampling is deprecated. "
                "Please merge g_sampling and the original graph into one graph and use "
                "the exclude argument to specify which edges you don't want to sample.")
        if isinstance(graph_sampler, BlockSampler):
            graph_sampler = EdgeBlockSampler(
                graph_sampler, exclude=exclude, reverse_eids=reverse_eids,
                reverse_etypes=reverse_etypes, negative_sampler=negative_sampler)

        super().__init__(
            graph, indices, graph_sampler, device=device, use_ddp=use_ddp, ddp_seed=ddp_seed,
            batch_size=batch_size, drop_last=drop_last, shuffle=shuffle,
            use_prefetch_thread=use_prefetch_thread, use_alternate_streams=use_alternate_streams,
            **kwargs)


######## Graph DataLoaders ########
# GraphDataLoader loads a set of graphs so it's not relevant to the above.  They are currently
# copied from the old DataLoader implementation.

PYTORCH_VER = LooseVersion(torch.__version__)
PYTORCH_16 = PYTORCH_VER >= LooseVersion("1.6.0")
PYTORCH_17 = PYTORCH_VER >= LooseVersion("1.7.0")

def _create_dist_sampler(dataset, dataloader_kwargs, ddp_seed):
    # Note: will change the content of dataloader_kwargs
    dist_sampler_kwargs = {'shuffle': dataloader_kwargs['shuffle']}
    dataloader_kwargs['shuffle'] = False
    if PYTORCH_16:
        dist_sampler_kwargs['seed'] = ddp_seed
    if PYTORCH_17:
        dist_sampler_kwargs['drop_last'] = dataloader_kwargs['drop_last']
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
