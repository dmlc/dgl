import multiprocessing as mp
import dgl
import torch as th
from dgl.distributed import DistGraphServer, DistGraph
import numpy as np

def init_dist_graph(dist_gclient_config, queue):
    global dist_gclient
    dist_gclient = DistGraph(**dist_gclient_config)
    print("DONDONDONDODNODNODN")
    queue.put(1)
    # dist_gclient.start()


def sample_blocks(seeds, dist_g, fanouts):
    # fanouts = kwargs.get("fanouts", None)
    assert fanouts is not None, "Fanouts is not specified"
    seeds = th.LongTensor(np.asarray(seeds))
    blocks = []
    for fanout in fanouts:
        # For each seed node, sample ``fanout`` neighbors.
        print("before frontier")
        frontier = dgl.distributed.sampling.sample_neighbors(
            dist_g, seeds, fanout, replace=True)
        print("after frontier")
        # Then we compact the frontier into a bipartite graph for message passing.
        print("Frontier: ", frontier)
        print("Frontier: ", frontier.edges())
        print("Seeds: ", seeds)
        print("Seeds: ", seeds.dtype)
        try:
            block = dgl.to_block(frontier, seeds)
        except Exception as e:
            print(e)
            
        # Obtain the seed nodes for next layer.
        seeds = block.srcdata[dgl.NID]
        print("done block")
        blocks.insert(0, block)
    return blocks


def queue_wrapper(fn, queue, seeds, sample_config):
    """Should change to decorator like implementation later"""
    """Use kwargs to pass variable later"""
    sample_config['dist_g'] = dist_gclient
    sample_config['seeds'] = seeds
    print("Sample Start")
    result = fn(**sample_config)
    print("Sample done")
    queue.put(result)
    return 1


class DistDataLoader:
    def __init__(self, dataset, batch_size, collate_fn, num_workers, queue_size, dist_gclient_config, sample_config):
        assert num_workers > 0
        # self.pool.join()
        self.sample_config = sample_config
        self.dataset = dataset
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.collate_fn = collate_fn
        self.current_pos = 0
        self.m = mp.Manager()
        self.queue = self.m.Queue(maxsize=queue_size)
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(
            num_workers, initializer=init_dist_graph, initargs=(dist_gclient_config, self.queue))
        for i in range(num_workers):
            self.queue.get()
        print("111111111")
        for _ in range(queue_size):
            self._request_next_batch()

    def __next__(self):
        print("@@@@@@@@@@@")
        result = self.queue.get()
        self._request_next_batch()
        return result

    def __iter__(self):
        return self

    def _request_next_batch(self):
        print("HHHHHHHHH")
        kkk = self.pool.apply_async(queue_wrapper, args=(
            self.collate_fn, self.queue, self._next_data(), self.sample_config))
        kkk.get()

    def _next_data(self):
        if self.current_pos + self.batch_size > len(self.dataset):
            raise StopIteration
        else:
            ret = self.dataset[self.current_pos:self.current_pos+self.batch_size]
            self.current_pos += self.batch_size
            print(f"Ret: {ret}")
            return ret
