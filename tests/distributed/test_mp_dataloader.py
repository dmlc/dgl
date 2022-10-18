import multiprocessing as mp
import os
import tempfile
import time

import backend as F
import dgl
import numpy as np
import pytest
import torch as th
from dgl.data import CitationGraphDataset
from dgl.distributed import (
    DistDataLoader,
    DistGraph,
    DistGraphServer,
    load_partition,
    partition_graph,
)
from scipy import sparse as spsp
from utils import generate_ip_config, reset_envs


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
                self.g, seeds, fanout, replace=True
            )
            # Then we compact the frontier into a bipartite graph for
            # message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return blocks


def start_server(
    rank,
    ip_config,
    part_config,
    disable_shared_mem,
    num_clients,
    keep_alive=False,
):
    print("server: #clients=" + str(num_clients))
    g = DistGraphServer(
        rank,
        ip_config,
        1,
        num_clients,
        part_config,
        disable_shared_mem=disable_shared_mem,
        graph_format=["csc", "coo"],
        keep_alive=keep_alive,
    )
    g.start()


def start_dist_dataloader(
    rank,
    ip_config,
    part_config,
    num_server,
    drop_last,
    orig_nid,
    orig_eid,
    group_id=0,
):
    import dgl
    import torch as th

    os.environ["DGL_GROUP_ID"] = str(group_id)
    dgl.distributed.initialize(ip_config)
    gpb = None
    disable_shared_mem = num_server > 0
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(part_config, rank)
    num_nodes_to_sample = 202
    batch_size = 32
    train_nid = th.arange(num_nodes_to_sample)
    dist_graph = DistGraph("test_mp", gpb=gpb, part_config=part_config)

    for i in range(num_server):
        part, _, _, _, _, _, _ = load_partition(part_config, i)

    # Create sampler
    sampler = NeighborSampler(
        dist_graph, [5, 10], dgl.distributed.sample_neighbors
    )

    # We need to test creating DistDataLoader multiple times.
    for i in range(2):
        # Create DataLoader for constructing blocks
        dataloader = DistDataLoader(
            dataset=train_nid.numpy(),
            batch_size=batch_size,
            collate_fn=sampler.sample_blocks,
            shuffle=False,
            drop_last=drop_last,
        )

        groundtruth_g = CitationGraphDataset("cora")[0]
        max_nid = []

        for epoch in range(2):
            for idx, blocks in zip(
                range(0, num_nodes_to_sample, batch_size), dataloader
            ):
                block = blocks[-1]
                o_src, o_dst = block.edges()
                src_nodes_id = block.srcdata[dgl.NID][o_src]
                dst_nodes_id = block.dstdata[dgl.NID][o_dst]
                max_nid.append(np.max(F.asnumpy(dst_nodes_id)))

                src_nodes_id = orig_nid[src_nodes_id]
                dst_nodes_id = orig_nid[dst_nodes_id]
                has_edges = groundtruth_g.has_edges_between(
                    src_nodes_id, dst_nodes_id
                )
                assert np.all(F.asnumpy(has_edges))
            if drop_last:
                assert (
                    np.max(max_nid)
                    == num_nodes_to_sample
                    - 1
                    - num_nodes_to_sample % batch_size
                )
            else:
                assert np.max(max_nid) == num_nodes_to_sample - 1
    del dataloader
    # this is needed since there's two test here in one process
    dgl.distributed.exit_client()


def test_standalone():
    reset_envs()
    with tempfile.TemporaryDirectory() as test_dir:
        ip_config = os.path.join(test_dir, "ip_config.txt")
        generate_ip_config(ip_config, 1, 1)

        g = CitationGraphDataset("cora")[0]
        print(g.idtype)
        num_parts = 1
        num_hops = 1

        orig_nid, orig_eid = partition_graph(
            g,
            "test_sampling",
            num_parts,
            test_dir,
            num_hops=num_hops,
            part_method="metis",
            reshuffle=True,
            return_mapping=True,
        )
        part_config = os.path.join(test_dir, "test_sampling.json")
        os.environ["DGL_DIST_MODE"] = "standalone"
        try:
            start_dist_dataloader(
                0, ip_config, part_config, 1, True, orig_nid, orig_eid
            )
        except Exception as e:
            print(e)


def start_dist_neg_dataloader(
    rank,
    ip_config,
    part_config,
    num_server,
    num_workers,
    orig_nid,
    groundtruth_g,
):
    import dgl
    import torch as th

    dgl.distributed.initialize(ip_config)
    gpb = None
    disable_shared_mem = num_server > 1
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(part_config, rank)
    num_edges_to_sample = 202
    batch_size = 32
    dist_graph = DistGraph("test_mp", gpb=gpb, part_config=part_config)
    assert len(dist_graph.ntypes) == len(groundtruth_g.ntypes)
    assert len(dist_graph.etypes) == len(groundtruth_g.etypes)
    if len(dist_graph.etypes) == 1:
        train_eid = th.arange(num_edges_to_sample)
    else:
        train_eid = {dist_graph.etypes[0]: th.arange(num_edges_to_sample)}

    for i in range(num_server):
        part, _, _, _, _, _, _ = load_partition(part_config, i)

    num_negs = 5
    sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 10])
    negative_sampler = dgl.dataloading.negative_sampler.Uniform(num_negs)
    dataloader = dgl.dataloading.DistEdgeDataLoader(
        dist_graph,
        train_eid,
        sampler,
        batch_size=batch_size,
        negative_sampler=negative_sampler,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    for _ in range(2):
        for _, (_, pos_graph, neg_graph, blocks) in zip(
            range(0, num_edges_to_sample, batch_size), dataloader
        ):
            block = blocks[-1]
            for src_type, etype, dst_type in block.canonical_etypes:
                o_src, o_dst = block.edges(etype=etype)
                src_nodes_id = block.srcnodes[src_type].data[dgl.NID][o_src]
                dst_nodes_id = block.dstnodes[dst_type].data[dgl.NID][o_dst]
                src_nodes_id = orig_nid[src_type][src_nodes_id]
                dst_nodes_id = orig_nid[dst_type][dst_nodes_id]
                has_edges = groundtruth_g.has_edges_between(
                    src_nodes_id, dst_nodes_id, etype=etype
                )
                assert np.all(F.asnumpy(has_edges))
                assert np.all(
                    F.asnumpy(block.dstnodes[dst_type].data[dgl.NID])
                    == F.asnumpy(pos_graph.nodes[dst_type].data[dgl.NID])
                )
                assert np.all(
                    F.asnumpy(block.dstnodes[dst_type].data[dgl.NID])
                    == F.asnumpy(neg_graph.nodes[dst_type].data[dgl.NID])
                )
                assert pos_graph.num_edges() * num_negs == neg_graph.num_edges()

    del dataloader
    # this is needed since there's two test here in one process
    dgl.distributed.exit_client()


def check_neg_dataloader(g, num_server, num_workers):
    with tempfile.TemporaryDirectory() as test_dir:
        ip_config = "ip_config.txt"
        generate_ip_config(ip_config, num_server, num_server)

        num_parts = num_server
        num_hops = 1
        orig_nid, orig_eid = partition_graph(
            g,
            "test_sampling",
            num_parts,
            test_dir,
            num_hops=num_hops,
            part_method="metis",
            reshuffle=True,
            return_mapping=True,
        )
        part_config = os.path.join(test_dir, "test_sampling.json")
        if not isinstance(orig_nid, dict):
            orig_nid = {g.ntypes[0]: orig_nid}
        if not isinstance(orig_eid, dict):
            orig_eid = {g.etypes[0]: orig_eid}

        pserver_list = []
        ctx = mp.get_context("spawn")
        for i in range(num_server):
            p = ctx.Process(
                target=start_server,
                args=(
                    i,
                    ip_config,
                    part_config,
                    num_server > 1,
                    num_workers + 1,
                ),
            )
            p.start()
            time.sleep(1)
            pserver_list.append(p)
        os.environ["DGL_DIST_MODE"] = "distributed"
        os.environ["DGL_NUM_SAMPLER"] = str(num_workers)
        ptrainer_list = []

        p = ctx.Process(
            target=start_dist_neg_dataloader,
            args=(
                0,
                ip_config,
                part_config,
                num_server,
                num_workers,
                orig_nid,
                g,
            ),
        )
        p.start()
        ptrainer_list.append(p)

        for p in pserver_list:
            p.join()
        for p in ptrainer_list:
            p.join()


@pytest.mark.parametrize("num_server", [3])
@pytest.mark.parametrize("num_workers", [0, 4])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("reshuffle", [True, False])
@pytest.mark.parametrize("num_groups", [1])
def test_dist_dataloader(
    num_server, num_workers, drop_last, reshuffle, num_groups
):
    reset_envs()
    # No multiple partitions on single machine for
    # multiple client groups in case of race condition.
    if num_groups > 1:
        num_server = 1
    with tempfile.TemporaryDirectory() as test_dir:
        ip_config = "ip_config.txt"
        generate_ip_config(ip_config, num_server, num_server)

        g = CitationGraphDataset("cora")[0]
        print(g.idtype)
        num_parts = num_server
        num_hops = 1

        orig_nid, orig_eid = partition_graph(
            g,
            "test_sampling",
            num_parts,
            test_dir,
            num_hops=num_hops,
            part_method="metis",
            reshuffle=reshuffle,
            return_mapping=True,
        )

        part_config = os.path.join(test_dir, "test_sampling.json")
        pserver_list = []
        ctx = mp.get_context("spawn")
        keep_alive = num_groups > 1
        for i in range(num_server):
            p = ctx.Process(
                target=start_server,
                args=(
                    i,
                    ip_config,
                    part_config,
                    num_server > 1,
                    num_workers + 1,
                    keep_alive,
                ),
            )
            p.start()
            time.sleep(1)
            pserver_list.append(p)

        os.environ["DGL_DIST_MODE"] = "distributed"
        os.environ["DGL_NUM_SAMPLER"] = str(num_workers)
        ptrainer_list = []
        num_trainers = 1
        for trainer_id in range(num_trainers):
            for group_id in range(num_groups):
                p = ctx.Process(
                    target=start_dist_dataloader,
                    args=(
                        trainer_id,
                        ip_config,
                        part_config,
                        num_server,
                        drop_last,
                        orig_nid,
                        orig_eid,
                        group_id,
                    ),
                )
                p.start()
                time.sleep(
                    1
                )  # avoid race condition when instantiating DistGraph
                ptrainer_list.append(p)

        for p in ptrainer_list:
            p.join()
        if keep_alive:
            for p in pserver_list:
                assert p.is_alive()
            # force shutdown server
            dgl.distributed.shutdown_servers("mp_ip_config.txt", 1)
        for p in pserver_list:
            p.join()


def start_node_dataloader(
    rank,
    ip_config,
    part_config,
    num_server,
    num_workers,
    orig_nid,
    orig_eid,
    groundtruth_g,
):
    dgl.distributed.initialize(ip_config)
    gpb = None
    disable_shared_mem = num_server > 1
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(part_config, rank)
    num_nodes_to_sample = 202
    batch_size = 32
    dist_graph = DistGraph("test_mp", gpb=gpb, part_config=part_config)
    assert len(dist_graph.ntypes) == len(groundtruth_g.ntypes)
    assert len(dist_graph.etypes) == len(groundtruth_g.etypes)
    if len(dist_graph.etypes) == 1:
        train_nid = th.arange(num_nodes_to_sample)
    else:
        train_nid = {"n3": th.arange(num_nodes_to_sample)}

    for i in range(num_server):
        part, _, _, _, _, _, _ = load_partition(part_config, i)

    # Create sampler
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [
            # test dict for hetero
            {etype: 5 for etype in dist_graph.etypes}
            if len(dist_graph.etypes) > 1
            else 5,
            10,
        ]
    )  # test int for hetero

    # We need to test creating DistDataLoader multiple times.
    for i in range(2):
        # Create DataLoader for constructing blocks
        dataloader = dgl.dataloading.DistNodeDataLoader(
            dist_graph,
            train_nid,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
        )

        for epoch in range(2):
            for idx, (_, _, blocks) in zip(
                range(0, num_nodes_to_sample, batch_size), dataloader
            ):
                block = blocks[-1]
                for src_type, etype, dst_type in block.canonical_etypes:
                    o_src, o_dst = block.edges(etype=etype)
                    src_nodes_id = block.srcnodes[src_type].data[dgl.NID][o_src]
                    dst_nodes_id = block.dstnodes[dst_type].data[dgl.NID][o_dst]
                    src_nodes_id = orig_nid[src_type][src_nodes_id]
                    dst_nodes_id = orig_nid[dst_type][dst_nodes_id]
                    has_edges = groundtruth_g.has_edges_between(
                        src_nodes_id, dst_nodes_id, etype=etype
                    )
                    assert np.all(F.asnumpy(has_edges))
    del dataloader
    # this is needed since there's two test here in one process
    dgl.distributed.exit_client()


def start_edge_dataloader(
    rank,
    ip_config,
    part_config,
    num_server,
    num_workers,
    orig_nid,
    orig_eid,
    groundtruth_g,
):
    dgl.distributed.initialize(ip_config)
    gpb = None
    disable_shared_mem = num_server > 1
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(part_config, rank)
    num_edges_to_sample = 202
    batch_size = 32
    dist_graph = DistGraph("test_mp", gpb=gpb, part_config=part_config)
    assert len(dist_graph.ntypes) == len(groundtruth_g.ntypes)
    assert len(dist_graph.etypes) == len(groundtruth_g.etypes)
    if len(dist_graph.etypes) == 1:
        train_eid = th.arange(num_edges_to_sample)
    else:
        train_eid = {dist_graph.etypes[0]: th.arange(num_edges_to_sample)}

    for i in range(num_server):
        part, _, _, _, _, _, _ = load_partition(part_config, i)

    # Create sampler
    sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 10])

    # We need to test creating DistDataLoader multiple times.
    for i in range(2):
        # Create DataLoader for constructing blocks
        dataloader = dgl.dataloading.DistEdgeDataLoader(
            dist_graph,
            train_eid,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
        )

        for epoch in range(2):
            for idx, (input_nodes, pos_pair_graph, blocks) in zip(
                range(0, num_edges_to_sample, batch_size), dataloader
            ):
                block = blocks[-1]
                for src_type, etype, dst_type in block.canonical_etypes:
                    o_src, o_dst = block.edges(etype=etype)
                    src_nodes_id = block.srcnodes[src_type].data[dgl.NID][o_src]
                    dst_nodes_id = block.dstnodes[dst_type].data[dgl.NID][o_dst]
                    src_nodes_id = orig_nid[src_type][src_nodes_id]
                    dst_nodes_id = orig_nid[dst_type][dst_nodes_id]
                    has_edges = groundtruth_g.has_edges_between(
                        src_nodes_id, dst_nodes_id, etype=etype
                    )
                    assert np.all(F.asnumpy(has_edges))
                    assert np.all(
                        F.asnumpy(block.dstnodes[dst_type].data[dgl.NID])
                        == F.asnumpy(
                            pos_pair_graph.nodes[dst_type].data[dgl.NID]
                        )
                    )
    del dataloader
    dgl.distributed.exit_client()


def check_dataloader(g, num_server, num_workers, dataloader_type):
    with tempfile.TemporaryDirectory() as test_dir:
        ip_config = "ip_config.txt"
        generate_ip_config(ip_config, num_server, num_server)

        num_parts = num_server
        num_hops = 1
        orig_nid, orig_eid = partition_graph(
            g,
            "test_sampling",
            num_parts,
            test_dir,
            num_hops=num_hops,
            part_method="metis",
            reshuffle=True,
            return_mapping=True,
        )
        part_config = os.path.join(test_dir, "test_sampling.json")
        if not isinstance(orig_nid, dict):
            orig_nid = {g.ntypes[0]: orig_nid}
        if not isinstance(orig_eid, dict):
            orig_eid = {g.etypes[0]: orig_eid}

        pserver_list = []
        ctx = mp.get_context("spawn")
        for i in range(num_server):
            p = ctx.Process(
                target=start_server,
                args=(
                    i,
                    ip_config,
                    part_config,
                    num_server > 1,
                    num_workers + 1,
                ),
            )
            p.start()
            time.sleep(1)
            pserver_list.append(p)

        os.environ["DGL_DIST_MODE"] = "distributed"
        os.environ["DGL_NUM_SAMPLER"] = str(num_workers)
        ptrainer_list = []
        if dataloader_type == "node":
            p = ctx.Process(
                target=start_node_dataloader,
                args=(
                    0,
                    ip_config,
                    part_config,
                    num_server,
                    num_workers,
                    orig_nid,
                    orig_eid,
                    g,
                ),
            )
            p.start()
            ptrainer_list.append(p)
        elif dataloader_type == "edge":
            p = ctx.Process(
                target=start_edge_dataloader,
                args=(
                    0,
                    ip_config,
                    part_config,
                    num_server,
                    num_workers,
                    orig_nid,
                    orig_eid,
                    g,
                ),
            )
            p.start()
            ptrainer_list.append(p)
        for p in pserver_list:
            p.join()
        for p in ptrainer_list:
            p.join()


def create_random_hetero():
    num_nodes = {"n1": 10000, "n2": 10010, "n3": 10020}
    etypes = [("n1", "r1", "n2"), ("n1", "r2", "n3"), ("n2", "r3", "n3")]
    edges = {}
    for etype in etypes:
        src_ntype, _, dst_ntype = etype
        arr = spsp.random(
            num_nodes[src_ntype],
            num_nodes[dst_ntype],
            density=0.001,
            format="coo",
            random_state=100,
        )
        edges[etype] = (arr.row, arr.col)
    g = dgl.heterograph(edges, num_nodes)
    g.nodes["n1"].data["feat"] = F.unsqueeze(
        F.arange(0, g.number_of_nodes("n1")), 1
    )
    g.edges["r1"].data["feat"] = F.unsqueeze(
        F.arange(0, g.number_of_edges("r1")), 1
    )
    return g


@pytest.mark.parametrize("num_server", [3])
@pytest.mark.parametrize("num_workers", [0, 4])
@pytest.mark.parametrize("dataloader_type", ["node", "edge"])
def test_dataloader(num_server, num_workers, dataloader_type):
    reset_envs()
    g = CitationGraphDataset("cora")[0]
    check_dataloader(g, num_server, num_workers, dataloader_type)
    g = create_random_hetero()
    check_dataloader(g, num_server, num_workers, dataloader_type)


@pytest.mark.parametrize("num_server", [3])
@pytest.mark.parametrize("num_workers", [0, 4])
def test_neg_dataloader(num_server, num_workers):
    reset_envs()
    g = CitationGraphDataset("cora")[0]
    check_neg_dataloader(g, num_server, num_workers)
    g = create_random_hetero()
    check_neg_dataloader(g, num_server, num_workers)


def start_multiple_dataloaders(
    ip_config, part_config, graph_name, orig_g, num_dataloaders, dataloader_type
):
    dgl.distributed.initialize(ip_config)
    dist_g = dgl.distributed.DistGraph(graph_name, part_config=part_config)
    if dataloader_type == "node":
        train_ids = th.arange(orig_g.num_nodes())
        batch_size = orig_g.num_nodes() // 100
    else:
        train_ids = th.arange(orig_g.num_edges())
        batch_size = orig_g.num_edges() // 100
    sampler = dgl.dataloading.NeighborSampler([-1])
    dataloaders = []
    dl_iters = []
    for _ in range(num_dataloaders):
        if dataloader_type == "node":
            dataloader = dgl.dataloading.DistNodeDataLoader(
                dist_g, train_ids, sampler, batch_size=batch_size
            )
        else:
            dataloader = dgl.dataloading.DistEdgeDataLoader(
                dist_g, train_ids, sampler, batch_size=batch_size
            )
        dataloaders.append(dataloader)
        dl_iters.append(iter(dataloader))

    # iterate on multiple dataloaders randomly
    while len(dl_iters) > 0:
        next_dl = np.random.choice(len(dl_iters), 1)[0]
        try:
            _ = next(dl_iters[next_dl])
        except StopIteration:
            dl_iters.pop(next_dl)
            del dataloaders[next_dl]

    dgl.distributed.exit_client()


@pytest.mark.parametrize("num_dataloaders", [1, 4])
@pytest.mark.parametrize("num_workers", [0, 1, 4])
@pytest.mark.parametrize("dataloader_type", ["node", "edge"])
def test_multiple_dist_dataloaders(
    num_dataloaders, num_workers, dataloader_type
):
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    os.environ["DGL_NUM_SAMPLER"] = str(num_workers)
    num_parts = 1
    num_servers = 1
    with tempfile.TemporaryDirectory() as test_dir:
        ip_config = os.path.join(test_dir, "ip_config.txt")
        generate_ip_config(ip_config, num_parts, num_servers)

        orig_g = dgl.rand_graph(1000, 10000)
        graph_name = "test"
        partition_graph(orig_g, graph_name, num_parts, test_dir)
        part_config = os.path.join(test_dir, f"{graph_name}.json")

        p_servers = []
        ctx = mp.get_context("spawn")
        for i in range(num_servers):
            p = ctx.Process(
                target=start_server,
                args=(
                    i,
                    ip_config,
                    part_config,
                    num_servers > 1,
                    num_workers + 1,
                ),
            )
            p.start()
            time.sleep(1)
            p_servers.append(p)

        p_client = ctx.Process(
            target=start_multiple_dataloaders,
            args=(
                ip_config,
                part_config,
                graph_name,
                orig_g,
                num_dataloaders,
                dataloader_type,
            ),
        )
        p_client.start()

        p_client.join()
        for p in p_servers:
            p.join()
    reset_envs()

