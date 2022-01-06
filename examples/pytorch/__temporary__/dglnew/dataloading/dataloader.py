from collections.abc import Mapping, Awaitable
from functools import partial
from queue import Queue
import threading
import asyncio
import torch
from dgl._ffi import streams as FS
from dgl.utils import recursive_apply, ExceptionWrapper
from dgl.frame import Column
from .asyncio_wrapper import AsyncIO

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


def _prefetch_for_column(name, id_, use_asyncio, storage, device, pin_memory):
    if use_asyncio:
        return asyncio.create_task(storage.async_fetch(id_, device, pin_memory=pin_memory))
    else:
        return storage.fetch(id_, device, pin_memory=pin_memory)


def _prefetch_update_feats(
        feats, frames, types, get_type_id_func, storages, id_name, use_asyncio, device,
        pin_memory):
    for tid, frame in enumerate(frames):
        type_ = types[tid]
        parent_tid = get_type_id_func(type_)
        default_id = frame[id_name]
        for key in frame.keys():
            column = frame[key]
            if isinstance(column, Marker):
                feats[tid, key] = _prefetch_for_column(
                    column.name or key, column.id_ or default_id, use_asyncio,
                    storages[type_, name], device, pin_memory)


# This class exists to avoid recursion into the feature dictionary returned by the
# prefetcher when calling recursive_apply().
class _PrefetchedGraphFeatures(object):
    __slots__ = ['node_feats', 'edge_feats']
    def __init__(self, node_feats, edge_feats):
        self.node_feats = node_feats
        self.edge_feats = edge_feats

    def topair(self):
        return self.node_feats, self.edge_feats


def _prefetch_for_subgraph(subg, dataloader):
    g = dataloader.graph

    node_feats, edge_feats = {}, {}
    _prefetch_update_feats(
        feats, subg._node_frames, subg.ntypes, dataloader.graph.get_ntype_id,
        dataloader.node_data, dgl.NID, dataloader.use_asyncio, dataloader.device,
        dataloader.pin_memory)
    _prefetch_update_feats(
        feats, subg._edge_frames, subg.canonical_etypes, dataloader.graph.get_etype_id,
        dataloader.edge_data, dgl.EID, dataloader.use_asyncio, dataloader.device,
        dataloader.pin_memory)
    return _PrefetchedGraphFeatures(node_feats, edge_feats)


def _prefetch_for(item, dataloader):
    if isinstance(item, dgl.DGLGraph):
        return _prefetch_for_subgraph(item, dataloader)
    elif isinstance(item, Marker):
        return _prefetch_for_column(
            item.name, item.id_, dataloader.use_asyncio, dataloader.other_data[item.name],
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
        feats = recursive_apply(_prefetch_for, batch)
    return feats


async def _async_prefetch(batch, device, stream, pin_memory, sampler):
    with torch.cuda.stream(stream):
        feats = _prefetch(batch, device, stream, pin_memory, sampler, True)
        feats = recursive_apply(lambda x: await x if isinstance(x, Awaitable) else x, feats)
    return feats


# Using Python thread makes one epoch 19s instead of 7.1s.
def _prefetcher_entry(dataloader_it, dataloader, queue):
    # TODO: figure out why PyTorch sets the number of threads to 1
    use_asyncio = dataloader.use_asyncio
    torch.set_num_threads(16)

    try:
        for batch in dataloader_it:
            if use_asyncio:
                feats = AsyncIO().run(_async_prefetch(batch, device, stream, pin_memory, sampler))
            else:
                feats = _prefetch(batch, device, stream, pin_memory, sampler, False)

            queue.put((
                # batch will be already in pinned memory as per the behavior of
                # PyTorch DataLoader.
                ### TODO: recursive apply to() for _PrefetchedGraphFeatures
                recursive_apply(batch, lambda x: x.to(device, non_blocking=True)),
                feats,
                stream.record_event(),
                None))
        queue.put((None, None, None, None))
    except:     # pylint: disable=bare-except
        queue.put((None, None, None, ExceptionWrapper(where='in prefetcher')))


class _PrefetchingIter(object):
    def __init__(self, dataloader, dataloader_it):
        self.queue = Queue(1)
        self.dataloader_it = dataloader_it
        self.dataloader = dataloader
        self.graph_sampler = self.dataloader.graph_sampler
        self.pin_memory = self.dataloader.pin_memory

        thread = threading.Thread(
            target=_prefetcher_entry,
            args=(dataloader_it, dataloader, self.queue),
            daemon=True)
        thread.start()
        self.thread = thread

    def __iter__(self):
        return self

    def __next__(self):
        batch, feats, stream_event, exception = self.queue.get()
        if batch is None:
            self.thread.join()
            if exception is None:
                raise StopIteration
            exception.reraise()
        #batch = next(self.dataloader_it)
        #device = self.dataloader.device
        #stream = torch.cuda.Stream(device=device)
        #if self.use_asyncio:
        #    feats = AsyncIO().run(_async_prefetch(
        #        batch, device, stream, self.pin_memory, self.graph_sampler))
        #else:
        #    feats = _prefetch(
        #        batch, device, stream, self.pin_memory, self.graph_sampler, False)
        #batch = recursive_apply(batch, lambda x: x.to(device, non_blocking=True))
        #stream_event = stream.record_event()

        # Assign the prefetched features back to batch
        target_dict = {}
        for storage_key, storage_dict in self.graph_sampler._storages.items():
            target_dict[storage_key] = getattr(
                self.graph_sampler,
                f'__{storage_key}_storages__')(batch)
        for (storage_key, i, feat_key), feat in feats.items():
            target = target_dict[storage_key][i]
            target[feat_key] = feat

        stream_event.wait()
        return batch


def _remove_parent_storage_columns(item):
    if isinstance(item, dgl.DGLGraph):
        for frame in itertools.chain(item._node_frames, item._edge_frames):
            for key in list(frame.keys()):
                col = frame[key]
                if isinstance(col, Column) and col.index is not None:
                    del frame[col]

def collate_wrapper(sample_func, g):
    def _sample(items):
        batch = sample_func(g, items)

        # Remove all the columns whose storages are parent storages (i.e. index is not
        # None)
        batch = recursive_apply(_remove_parent_storage_columns, batch)
        return batch
    return _sample


class NodeDataLoader(torch.utils.data.DataLoader):
    def __init__(self, graph, train_idx, graph_sampler, device='cpu', use_ddp=False,
                 ddp_seed=0, batch_size=1, drop_last=False, shuffle=False,
                 use_asyncio=False, **kwargs):
        self.dataset = TensorizedDataset(train_idx, graph.ntypes, batch_size, drop_last)
        self.use_ddp = use_ddp
        self.ddp_seed = ddp_seed
        self._shuffle_dataset = shuffle
        self.graph_sampler = graph_sampler
        self.use_asyncio = use_asyncio
        self.device = torch.device(device)

        super().__init__(
            self.dataset,
            collate_fn=collate_wrapper(graph_sampler.sample, graph),
            batch_size=None,
            **kwargs)

    def __iter__(self):
        if self._shuffle_dataset:
            self.dataset.shuffle()
        return _PrefetchingIter(self, super().__iter__())
