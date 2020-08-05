# pylint: disable=global-variable-undefined, invalid-name
"""Multiprocess dataloader for distributed training"""
import multiprocessing as mp
import time
import traceback

from .dist_context import get_sampler_pool
from .. import backend as F

__all__ = ["DistDataLoader"]


def call_collate_fn(next_data):
    """Call collate function"""
    try:
        result = DGL_GLOBAL_COLLATE_FN(next_data)
        DGL_GLOBAL_MP_QUEUE.put(result)
    except Exception as e:
        traceback.print_exc()
        print(e)
        raise e
    return 1


def init_fn(collate_fn, queue):
    """Initialize setting collate function and mp.Queue in the subprocess"""
    global DGL_GLOBAL_COLLATE_FN
    global DGL_GLOBAL_MP_QUEUE
    DGL_GLOBAL_MP_QUEUE = queue
    DGL_GLOBAL_COLLATE_FN = collate_fn
    # sleep here is to ensure this function is executed in all worker processes
    # probably need better solution in the future
    time.sleep(1)
    return 1


def enable_mp_debug():
    """Print multiprocessing debug information. This is only
    for debug usage"""
    import logging
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.DEBUG)


class DistDataLoader:
    """DGL customized multiprocessing dataloader, which is designed for using with DistGraph."""

    def __init__(self, dataset, batch_size, shuffle=False, collate_fn=None, drop_last=False,
                 queue_size=None):
        """
        This class will utilize the worker process created by dgl.distributed.initialize function

        Note that the iteration order is not guaranteed with this class. For example,
         if dataset = [1, 2, 3, 4], batch_size = 2 and shuffle = False, the order of [1, 2]
         and [3, 4] is not guaranteed.

        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        queue_size (int, optional): Size of multiprocessing queue
        """
        self.pool, num_workers = get_sampler_pool()
        assert num_workers > 0, "DistDataloader only supports num_workers>0 for now. if you \
            want to use single process dataloader, please use PyTorch dataloader for now"
        if queue_size is None:
            queue_size = num_workers * 4
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.collate_fn = collate_fn
        self.current_pos = 0
        self.num_workers = num_workers
        self.m = mp.Manager()
        self.queue = self.m.Queue(maxsize=queue_size)
        self.drop_last = drop_last
        self.recv_idxs = 0
        self.started = False
        self.shuffle = shuffle
        self.is_closed = False

        if self.pool is None:
            ctx = mp.get_context("spawn")
            self.pool = ctx.Pool(num_workers)

        self.dataset = F.tensor(dataset)
        self.expected_idxs = len(dataset) // self.batch_size
        if not self.drop_last and len(dataset) % self.batch_size != 0:
            self.expected_idxs += 1

    def __next__(self):
        if not self.started:
            for _ in range(self.queue_size):
                self._request_next_batch()
        self._request_next_batch()
        if self.recv_idxs < self.expected_idxs:
            result = self.queue.get(timeout=9999)
            self.recv_idxs += 1
            return result
        else:
            self.recv_idxs = 0
            self.current_pos = 0
            raise StopIteration

    def __iter__(self):
        if self.shuffle:
            self.dataset = F.rand_shuffle(self.dataset)

        results = []
        for _ in range(self.num_workers):
            results.append(self.pool.apply_async(
                init_fn, args=(self.collate_fn, self.queue)))
        for res in results:
            res.get()
        return self

    def _request_next_batch(self):
        next_data = self._next_data()
        if next_data is None:
            return None
        else:
            async_result = self.pool.apply_async(
                call_collate_fn, args=(next_data, ))
            return async_result

    def _next_data(self):
        if self.current_pos == len(self.dataset):
            return None

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
