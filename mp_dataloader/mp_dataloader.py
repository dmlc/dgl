import multiprocessing as mp
import dgl
import torch as th
from dgl.distributed import DistGraphServer, DistGraph
import numpy as np
# import multiprocessing, logging
# logger = multiprocessing.log_to_stderr()
# logger.setLevel(logging.DEBUG)
from multiprocessing.util import Finalize

DGL_QUEUE_TIMEOUT = 10


def close():
    dgl.distributed.shutdown_servers()
    dgl.distributed.finalize_client()


def init_dist_graph(dist_gclient_config, queue):
    global dist_gclient
    dist_gclient = DistGraph(**dist_gclient_config)
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


def sample_blocks(seeds, dist_g, fanouts):
    # fanouts = kwargs.get("fanouts", None)
    assert fanouts is not None, "Fanouts is not specified"
    seeds = th.LongTensor(np.asarray(seeds))
    blocks = []
    for fanout in fanouts:
        # For each seed node, sample ``fanout`` neighbors.
        frontier = dgl.distributed.sampling.sample_neighbors(
            dist_g, seeds, fanout, replace=True)
        block = dgl.to_block(frontier, seeds)
        # Obtain the seed nodes for next layer.
        seeds = block.srcdata[dgl.NID]
        blocks.insert(0, block)
    return blocks


def queue_wrapper(fn, queue, seeds, sample_config):
    """Should change to decorator like implementation later"""
    """Use kwargs to pass variable later"""
    sample_config['dist_g'] = dist_gclient
    sample_config['seeds'] = seeds
    result = fn(**sample_config)
    queue.put(result)
    return 1


spawn_ctx = mp.get_context("spawn")


class DistDataLoader:
    def __init__(self, dataset, batch_size, collate_fn, num_workers, queue_size, drop_last, dist_gclient_config, sample_config):
        assert num_workers > 0
        # self.pool.join()
        self.sample_config = sample_config
        self.dataset = dataset
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.collate_fn = collate_fn
        self.current_pos = 0
        self.num_workers = num_workers
        self.m = mp.Manager()
        self.queue = self.m.Queue(maxsize=queue_size)
        ctx = spawn_ctx

        self.drop_last = drop_last

        self.expected_idxs = len(dataset) // batch_size
        if not drop_last and len(dataset) % batch_size != 0:
            self.expected_idxs += 1

        self.send_idxs = 0
        self.recv_idxs = 0

        self.pool = ctx.Pool(
            num_workers, initializer=init_dist_graph, initargs=(dist_gclient_config, self.queue))
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
            async_result = self.pool.apply_async(queue_wrapper, args=(
                self.collate_fn, self.queue, next_data, self.sample_config))
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
