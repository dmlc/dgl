"""Multiprocess dataloader for distributed training"""
import multiprocessing as mp
import dgl
from . import shutdown_servers, finalize_client

DGL_QUEUE_TIMEOUT = 10

__all__ = ["DistDataLoader"]


def close():
    """Finalize client and close servers when finished"""
    shutdown_servers()
    finalize_client()


def init_fn(collate_fn, mp_queue):
    """"""
    global DGL_GLOBAL_COLLATE_FN
    global DGL_GLOBAL_MP_QUEUE
    DGL_GLOBAL_MP_QUEUE = mp_queue
    DGL_GLOBAL_COLLATE_FN = collate_fn
    import atexit
    atexit.register(close)


def deregister_torch_ipc():
    from multiprocessing.reduction import ForkingPickler
    import torch
    ForkingPickler._extra_reducers.pop(torch.cuda.Event)
    for t in torch._storage_classes:
        ForkingPickler._extra_reducers.pop(t)
    for t in torch._tensor_classes:
        ForkingPickler._extra_reducers.pop(t)
    ForkingPickler._extra_reducers.pop(torch.Tensor)
    ForkingPickler._extra_reducers.pop(torch.nn.parameter.Parameter)


def call_collate_fn(next_data):
    result = DGL_GLOBAL_COLLATE_FN(next_data)
    DGL_GLOBAL_MP_QUEUE.put(result)
    return 1


class DistDataLoader:
    """DGL customized multiprocessing dataloader"""
    def __init__(self, dataset, batch_size, num_workers, collate_fn, drop_last, queue_size=None):
        """
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        queue_size (int): Size of multiprocessing queue
        """
        assert num_workers > 0
        if queue_size is None:
            queue_size = num_workers * 4
        self.queue_size = queue_size
        self.dataset = dataset
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.collate_fn = collate_fn
        self.current_pos = 0
        self.num_workers = num_workers
        self.m = mp.Manager()
        self.queue = self.m.Queue(maxsize=queue_size)
        ctx = mp.get_context("spawn")
        self.drop_last = drop_last
        self.expected_idxs = len(dataset) // batch_size
        if not drop_last and len(dataset) % batch_size != 0:
            self.expected_idxs += 1
        self.send_idxs = 0
        self.recv_idxs = 0
        self.pool = ctx.Pool(
            num_workers, initializer=init_fn, initargs=(collate_fn, self.queue))
        for _ in range(queue_size):
            self._request_next_batch()

    def __next__(self):
        self._request_next_batch()
        if self.recv_idxs < self.expected_idxs:
            result = self.queue.get(timeout=DGL_QUEUE_TIMEOUT)
            self.recv_idxs += 1
            return result
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def _request_next_batch(self):
        next_data = self._next_data()
        if next_data is None:
            return None
        else:
            async_result = self.pool.apply_async(
                call_collate_fn, args=(next_data, ))
            self.send_idxs += 1
            return async_result

    def _next_data(self):
        end_pos = 0
        if self.current_pos + self.batch_size > len(self.dataset):
            if self.drop_last:
                return None
            else:
                end_pos = len(self.dataset)
        else:
            end_pos = self.current_pos + self.batch_size
        ret = self.dataset[self.current_pos:end_pos]
        self.current_pos = end_pos
        return ret

    def close(self):
        """Close the multiprocessing pool"""
        self.pool.close()
        self.pool.join()


deregister_torch_ipc()
