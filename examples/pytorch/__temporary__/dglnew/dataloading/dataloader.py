from collections.abc import Mapping
from functools import partial
from queue import Queue
import threading
import torch
from dgl._ffi import streams as FS
from dgl.utils import recursive_apply, ExceptionWrapper

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


def _prefetcher_entry(dataloader_it, dataloader, queue):
    device = dataloader.device
    stream = torch.cuda.Stream(device=device)
    pin_memory = dataloader.pin_memory
    sampler = dataloader.graph_sampler
    sampler_cls = sampler.__class__

    try:
        for batch in dataloader_it:
            # The futures and the retrieved features will be indexed as:
            #
            #     foo[storage_key][target_idx][feat_key]
            future_or_values = defaultdict(list)

            # Copy the sampled subgraphs to GPU first since setting GPU features require
            # having the graph on GPU first.
            with torch.cuda.stream(stream):
                for storage_key, storage_dict in sampler._storages.items():
                    targets = getattr(
                        sampler,
                        f'__{storage_key}_storages__')(batch)
                    for target in targets:
                        indices = target['_ID']     # depends on dgl.NID and dgl.EID being the same
                        target_future_or_value = {}

                        for feat_key, storage in storage_dict.items():
                            if dataloader.prefetch == 'async' and hasattr(storage, 'async_fetch'):
                                future = storage.async_fetch(
                                    indices, device, pin_memory=pin_memory)
                                target_future_or_value[feat_key] = future
                            else:
                                target_future_or_value[feat_key] = storage.fetch(
                                    indices, device, pin_memory=pin_memory)

                        future_or_values[storage_key].append(target_future_or_value)
                queue.put((
                    # batch will be already in pinned memory as per the behavior of
                    # PyTorch DataLoader.
                    recursive_apply(batch, lambda x: x.to(device, non_blocking=True)),
                    future_or_values,
                    stream.record_event(),
                    None))
        queue.put((None, None, None, None))
    except:     # pylint: disable=bare-except
        queue.put((None, None, None, ExceptionWrapper(where='in prefetcher')))


# Prefetcher thread is responsible for issuing the requests to the storages to fetch
# features from the given indices.
class _PrefetchingIter(object):
    def __init__(self, dataloader, dataloader_it):
        self.queue = Queue(1)
        self.thread = threading.Thread(
            target=_prefetcher_entry,
            args=(dataloader_it, dataloader, self.queue),
            daemon=True)

        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        batch, future_or_values, stream_event, exception = self.queue.get()
        if result is None:
            self.thread.join()
            if exception is None:
                raise StopIteration
            exception.reraise()

        # If there are remaining futures, wait for them and put them back in result
        for storage_key, future_or_value_list in future_or_values.items():
            targets = getattr(
                sampler,
                f'__{storage_key}_storages__')(batch)
            for target, target_futures in zip(targets, future_or_value_list):
                for feat_key, future_or_value in target_futures.items():
                    target[feat_key] = (
                        (await future_or_value) if isinstance(future_or_value, Awaitable)
                        else future_or_value)

        stream_event.wait()
        return batch


class NodeDataLoader(torch.utils.data.DataLoader):
    """
    Parameters
    ----------
    prefetch : None, ``"sync"`` or ``"async"``
        If None, go without feature prefetching.  Otherwise, a Python thread will spawn
        to 
    """
    def __init__(self, graph, train_idx, graph_sampler, device='cpu', use_ddp=False,
                 ddp_seed=0, batch_size=1, drop_last=False, shuffle=False,
                 prefetch=None, **kwargs):
        self.dataset = TensorizedDataset(train_idx, graph.ntypes, batch_size, drop_last)
        self.use_ddp = use_ddp
        self.ddp_seed = ddp_seed
        self._shuffle_dataset = shuffle
        self.graph_sampler = graph_sampler
        self.prefetch = prefetch
        self.device = torch.device(device)

        super().__init__(
            self.dataset,
            collate_fn=graph_sampler.sample,
            **kwargs)

    def __setattr__(self, name, val):
        if name == 'shuffle':
            # redirect to _shuffle_dataset
            self._shuffle_dataset = val
        else:
            super().__setattr__(name, val)

    def __iter__(self):
        if self._shuffle_dataset:
            self.dataset.shuffle()
        if self.prefetch:
            return _PrefetchingIter(self, super().__iter__())
        else:
            return super().__iter__()
