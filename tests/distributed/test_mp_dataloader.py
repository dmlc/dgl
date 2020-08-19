import dgl
import unittest
import os
from dgl.data import CitationGraphDataset
from dgl.distributed import sample_neighbors
from dgl.distributed import partition_graph, load_partition, load_partition_book
import sys
import multiprocessing as mp
import numpy as np
import time
from utils import get_local_usable_addr
from pathlib import Path
from dgl.distributed import DistGraphServer, DistGraph, DistDataLoader
import pytest
import backend as F

class NeighborSampler(object):
    def __init__(self, g, fanouts, sample_neighbors):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors

    def sample_blocks(self, seeds):
        import torch as th
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = self.sample_neighbors(
                self.g, seeds, fanout, replace=True)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return blocks


def start_server(rank, tmpdir, disable_shared_mem, num_clients):
    import dgl
    print('server: #clients=' + str(num_clients))
    g = DistGraphServer(rank, "mp_ip_config.txt", 1, num_clients,
                        tmpdir / 'test_sampling.json', disable_shared_mem=disable_shared_mem)
    g.start()


def start_dist_dataloader(rank, tmpdir, disable_shared_mem, num_workers, drop_last):
    import dgl
    import torch as th
    dgl.distributed.initialize("mp_ip_config.txt", 1, num_workers=num_workers)
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _ = load_partition(tmpdir / 'test_sampling.json', rank)
    num_nodes_to_sample = 202
    batch_size = 32
    train_nid = th.arange(num_nodes_to_sample)
    dist_graph = DistGraph("test_mp", gpb=gpb, part_config=tmpdir / 'test_sampling.json')

    # Create sampler
    sampler = NeighborSampler(dist_graph, [5, 10],
                              dgl.distributed.sample_neighbors)

    # We need to test creating DistDataLoader multiple times.
    for i in range(2):
        # Create DataLoader for constructing blocks
        dataloader = DistDataLoader(
            dataset=train_nid.numpy(),
            batch_size=batch_size,
            collate_fn=sampler.sample_blocks,
            shuffle=False,
            drop_last=drop_last)

        groundtruth_g = CitationGraphDataset("cora")[0]
        max_nid = []

        for epoch in range(2):
            for idx, blocks in zip(range(0, num_nodes_to_sample, batch_size), dataloader):
                block = blocks[-1]
                o_src, o_dst =  block.edges()
                src_nodes_id = block.srcdata[dgl.NID][o_src]
                dst_nodes_id = block.dstdata[dgl.NID][o_dst]
                has_edges = groundtruth_g.has_edges_between(src_nodes_id, dst_nodes_id)
                assert np.all(F.asnumpy(has_edges))
                max_nid.append(np.max(F.asnumpy(dst_nodes_id)))
                # assert np.all(np.unique(np.sort(F.asnumpy(dst_nodes_id))) == np.arange(idx, batch_size))
            if drop_last:
                assert np.max(max_nid) == num_nodes_to_sample - 1 - num_nodes_to_sample % batch_size
            else:
                assert np.max(max_nid) == num_nodes_to_sample - 1
    del dataloader
    dgl.distributed.exit_client() # this is needed since there's two test here in one process

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name != 'pytorch', reason='Only support PyTorch for now')
def test_standalone(tmpdir):
    ip_config = open("mp_ip_config.txt", "w")
    for _ in range(1):
        ip_config.write('{}\n'.format(get_local_usable_addr()))
    ip_config.close()

    g = CitationGraphDataset("cora")[0]
    print(g.idtype)
    num_parts = 1
    num_hops = 1

    partition_graph(g, 'test_sampling', num_parts, tmpdir,
                    num_hops=num_hops, part_method='metis', reshuffle=False)

    os.environ['DGL_DIST_MODE'] = 'standalone'
    try:
        start_dist_dataloader(0, tmpdir, False, 2, True)
    except Exception as e:
        print(e)
    dgl.distributed.exit_client() # this is needed since there's two test here in one process


@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name != 'pytorch', reason='Only support PyTorch for now')
@pytest.mark.parametrize("num_server", [3])
@pytest.mark.parametrize("num_workers", [0, 4])
@pytest.mark.parametrize("drop_last", [True, False])
def test_dist_dataloader(tmpdir, num_server, num_workers, drop_last):
    ip_config = open("mp_ip_config.txt", "w")
    for _ in range(num_server):
        ip_config.write('{}\n'.format(get_local_usable_addr()))
    ip_config.close()

    g = CitationGraphDataset("cora")[0]
    print(g.idtype)
    num_parts = num_server
    num_hops = 1

    partition_graph(g, 'test_sampling', num_parts, tmpdir,
                    num_hops=num_hops, part_method='metis', reshuffle=False)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(
            i, tmpdir, num_server > 1, num_workers+1))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    time.sleep(3)
    os.environ['DGL_DIST_MODE'] = 'distributed'
    ptrainer = ctx.Process(target=start_dist_dataloader, args=(
        0, tmpdir, num_server > 1, num_workers, drop_last))
    ptrainer.start()
    time.sleep(1)

    for p in pserver_list:
        p.join()
    ptrainer.join()

def start_node_dataloader(rank, tmpdir, disable_shared_mem, num_workers):
    import dgl
    import torch as th
    dgl.distributed.initialize("mp_ip_config.txt", 1, num_workers=num_workers)
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _ = load_partition(tmpdir / 'test_sampling.json', rank)
    num_nodes_to_sample = 202
    batch_size = 32
    train_nid = th.arange(num_nodes_to_sample)
    dist_graph = DistGraph("test_mp", gpb=gpb, part_config=tmpdir / 'test_sampling.json')

    # Create sampler
    sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 10])

    # We need to test creating DistDataLoader multiple times.
    for i in range(2):
        # Create DataLoader for constructing blocks
        dataloader = dgl.dataloading.NodeDataLoader(
            dist_graph,
            train_nid,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers)

        groundtruth_g = CitationGraphDataset("cora")[0]
        max_nid = []

        for epoch in range(2):
            for idx, (_, _, blocks) in zip(range(0, num_nodes_to_sample, batch_size), dataloader):
                block = blocks[-1]
                o_src, o_dst =  block.edges()
                src_nodes_id = block.srcdata[dgl.NID][o_src]
                dst_nodes_id = block.dstdata[dgl.NID][o_dst]
                has_edges = groundtruth_g.has_edges_between(src_nodes_id, dst_nodes_id)
                assert np.all(F.asnumpy(has_edges))
                max_nid.append(np.max(F.asnumpy(dst_nodes_id)))
                # assert np.all(np.unique(np.sort(F.asnumpy(dst_nodes_id))) == np.arange(idx, batch_size))
    del dataloader
    dgl.distributed.exit_client() # this is needed since there's two test here in one process


@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name != 'pytorch', reason='Only support PyTorch for now')
@pytest.mark.parametrize("num_server", [3])
@pytest.mark.parametrize("num_workers", [0, 4])
@pytest.mark.parametrize("dataloader_type", ["node"])
def test_dataloader(tmpdir, num_server, num_workers, dataloader_type):
    ip_config = open("mp_ip_config.txt", "w")
    for _ in range(num_server):
        ip_config.write('{}\n'.format(get_local_usable_addr()))
    ip_config.close()

    g = CitationGraphDataset("cora")[0]
    print(g.idtype)
    num_parts = num_server
    num_hops = 1

    partition_graph(g, 'test_sampling', num_parts, tmpdir,
                    num_hops=num_hops, part_method='metis', reshuffle=False)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(
            i, tmpdir, num_server > 1, num_workers+1))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    time.sleep(3)
    os.environ['DGL_DIST_MODE'] = 'distributed'
    ptrainer_list = []
    if dataloader_type == 'node':
        p = ctx.Process(target=start_node_dataloader, args=(
            0, tmpdir, num_server > 1, num_workers))
        p.start()
        time.sleep(1)
        ptrainer_list.append(p)
    for p in pserver_list:
        p.join()
    for p in ptrainer_list:
        p.join()

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        test_dataloader(Path(tmpdirname), 3, 4, 'node')
        test_standalone(Path(tmpdirname))
        test_dist_dataloader(Path(tmpdirname), 3, 0, True)
        test_dist_dataloader(Path(tmpdirname), 3, 4, True)
