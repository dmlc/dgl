import multiprocessing as mp
import dgl
from . import DistGraphServer, DistGraph, shutdown_servers, finalize_client
import numpy as np

DGL_QUEUE_TIMEOUT = 10

__all__ = ["DistDataLoader"]

def close():
    shutdown_servers()
    finalize_client()


def init_fn(collate_fn, mp_queue):
    global gfn
    global queue
    queue = mp_queue
    gfn = collate_fn
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
    result = gfn(next_data)
    queue.put(result)
    return 1


class DistDataLoader:
    """"""
    def __init__(self, dataset, batch_size, collate_fn, num_workers, drop_last, queue_size=None):
        """"""
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
        self.pool.close()
        self.pool.join()


deregister_torch_ipc()
