import multiprocessing as mp
import os
import tempfile
import time
import unittest
import uuid

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


def _unique_rand_graph(num_nodes=1000, num_edges=10 * 1000):
    edges_set = set()
    while len(edges_set) < num_edges:
        src = np.random.randint(0, num_nodes - 1)
        dst = np.random.randint(0, num_nodes - 1)
        if (
            src != dst
            and (src, dst) not in edges_set
            and (dst, src) not in edges_set
        ):
            edges_set.add((src, dst))
    src_list, dst_list = zip(*edges_set)

    src = th.tensor(src_list, dtype=th.long)
    dst = th.tensor(dst_list, dtype=th.long)
    g = dgl.graph((th.cat([src, dst]), th.cat([dst, src])))
    E = len(src)
    reverse_eids = th.cat([th.arange(E, 2 * E), th.arange(0, E)])
    return g, reverse_eids


class NeighborSampler(object):
    def __init__(
        self,
        g,
        fanouts,
        sample_neighbors,
        use_graphbolt=False,
        return_eids=False,
    ):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors
        self.use_graphbolt = use_graphbolt
        self.return_eids = return_eids

    def sample_blocks(self, seeds):
        import torch as th

        seeds = th.tensor(np.asarray(seeds), dtype=self.g.idtype)
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = self.sample_neighbors(
                self.g, seeds, fanout, use_graphbolt=self.use_graphbolt
            )
            # Then we compact the frontier into a bipartite graph for
            # message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]
            if frontier.num_edges() > 0:
                if not self.use_graphbolt or self.return_eids:
                    block.edata[dgl.EID] = frontier.edata[dgl.EID]

            blocks.insert(0, block)
        return blocks


def start_server(
    rank,
    ip_config,
    part_config,
    disable_shared_mem,
    num_clients,
    use_graphbolt=False,
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
        use_graphbolt=use_graphbolt,
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
    use_graphbolt=False,
    return_eids=False,
):
    dgl.distributed.initialize(ip_config)
    gpb = None
    disable_shared_mem = num_server > 1
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(part_config, rank)
    num_nodes_to_sample = 202
    batch_size = 32
    train_nid = th.arange(num_nodes_to_sample)
    graph_name = os.path.splitext(os.path.basename(part_config))[0]
    dist_graph = DistGraph(
        graph_name,
        gpb=gpb,
        part_config=part_config,
    )

    # Create sampler
    sampler = NeighborSampler(
        dist_graph,
        [5, 10],
        dgl.distributed.sample_neighbors,
        use_graphbolt=use_graphbolt,
        return_eids=return_eids,
    )

    # Enable santity check in distributed sampling.
    os.environ["DGL_DIST_DEBUG"] = "1"

    # We need to test creating DistDataLoader multiple times.
    for i in range(2):
        # Create DataLoader for constructing blocks
        dataloader = DistDataLoader(
            dataset=train_nid,
            batch_size=batch_size,
            collate_fn=sampler.sample_blocks,
            shuffle=False,
            drop_last=drop_last,
        )

        groundtruth_g = CitationGraphDataset("cora")[0]
        max_nid = []

        for _ in range(2):
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

                if use_graphbolt and not return_eids:
                    continue
                eids = orig_eid[block.edata[dgl.EID]]
                expected_eids = groundtruth_g.edge_ids(
                    src_nodes_id, dst_nodes_id
                )
                assert th.equal(
                    eids, expected_eids
                ), f"{eids} != {expected_eids}"
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


@unittest.skip(reason="Skip due to glitch in CI")
def test_standalone():
    reset_envs()
    with tempfile.TemporaryDirectory() as test_dir:
        ip_config = os.path.join(test_dir, "ip_config.txt")
        generate_ip_config(ip_config, 1, 1)

        g = CitationGraphDataset("cora")[0]
        print(g.idtype)
        num_parts = 1
        num_hops = 1
        graph_name = f"graph_{uuid.uuid4()}"
        orig_nid, orig_eid = partition_graph(
            g,
            graph_name,
            num_parts,
            test_dir,
            num_hops=num_hops,
            part_method="metis",
            return_mapping=True,
        )
        part_config = os.path.join(test_dir, f"{graph_name}.json")
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
    graph_name = os.path.splitext(os.path.basename(part_config))[0]
    dist_graph = DistGraph(graph_name, gpb=gpb, part_config=part_config)
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
    dataloader = dgl.distributed.DistEdgeDataLoader(
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
        graph_name = f"graph_{uuid.uuid4()}"
        orig_nid, orig_eid = partition_graph(
            g,
            graph_name,
            num_parts,
            test_dir,
            num_hops=num_hops,
            part_method="metis",
            return_mapping=True,
        )
        part_config = os.path.join(test_dir, f"{graph_name}.json")
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
            assert p.exitcode == 0
        for p in ptrainer_list:
            p.join()
            assert p.exitcode == 0


@pytest.mark.parametrize("num_server", [1])
@pytest.mark.parametrize("num_workers", [0, 1])
@pytest.mark.parametrize("use_graphbolt", [False, True])
@pytest.mark.parametrize("return_eids", [False, True])
def test_dist_dataloader(num_server, num_workers, use_graphbolt, return_eids):
    if not use_graphbolt and return_eids:
        # return_eids is not supported in non-GraphBolt mode.
        return
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    os.environ["DGL_NUM_SAMPLER"] = str(num_workers)
    with tempfile.TemporaryDirectory() as test_dir:
        ip_config = "ip_config.txt"
        generate_ip_config(ip_config, num_server, num_server)

        g = CitationGraphDataset("cora")[0]
        num_parts = num_server
        num_hops = 1
        graph_name = f"graph_{uuid.uuid4()}"
        orig_nid, orig_eid = partition_graph(
            g,
            graph_name,
            num_parts,
            test_dir,
            num_hops=num_hops,
            part_method="metis",
            return_mapping=True,
            use_graphbolt=use_graphbolt,
            store_eids=return_eids,
        )

        part_config = os.path.join(test_dir, f"{graph_name}.json")
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
                    use_graphbolt,
                ),
            )
            p.start()
            time.sleep(1)
            pserver_list.append(p)

        ptrainer_list = []
        num_trainers = 1
        for trainer_id in range(num_trainers):
            p = ctx.Process(
                target=start_dist_dataloader,
                args=(
                    trainer_id,
                    ip_config,
                    part_config,
                    num_server,
                    False,
                    orig_nid,
                    orig_eid,
                    use_graphbolt,
                    return_eids,
                ),
            )
            p.start()
            time.sleep(1)  # avoid race condition when instantiating DistGraph
            ptrainer_list.append(p)

        for p in ptrainer_list:
            p.join()
            assert p.exitcode == 0
        for p in pserver_list:
            p.join()
            assert p.exitcode == 0


def start_node_dataloader(
    rank,
    ip_config,
    part_config,
    num_server,
    num_workers,
    orig_nid,
    orig_eid,
    groundtruth_g,
    use_graphbolt=False,
    return_eids=False,
    prob_or_mask=None,
    use_deprecated_dataloader=False,
):
    dgl.distributed.initialize(ip_config, use_graphbolt=use_graphbolt)
    gpb = None
    disable_shared_mem = num_server > 1
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(part_config, rank)
    num_nodes_to_sample = 202
    batch_size = 32
    graph_name = os.path.splitext(os.path.basename(part_config))[0]
    dist_graph = DistGraph(
        graph_name,
        gpb=gpb,
        part_config=part_config,
    )
    assert len(dist_graph.ntypes) == len(groundtruth_g.ntypes)
    assert len(dist_graph.etypes) == len(groundtruth_g.etypes)
    if len(dist_graph.etypes) == 1:
        train_nid = th.arange(num_nodes_to_sample, dtype=dist_graph.idtype)
    else:
        train_nid = {
            "n3": th.arange(num_nodes_to_sample, dtype=dist_graph.idtype)
        }

    for i in range(num_server):
        part, _, _, _, _, _, _ = load_partition(part_config, i)

    # Create sampler
    _prob = None
    _mask = None
    if prob_or_mask is None:
        pass
    elif prob_or_mask == "prob":
        _prob = "prob"
    elif prob_or_mask == "mask":
        _mask = "mask"
    else:
        raise ValueError(f"Unsupported prob type: {prob_or_mask}")
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [
            (
                # test dict for hetero
                {etype: 5 for etype in dist_graph.etypes}
                if len(dist_graph.etypes) > 1
                else 5
            ),
            10,
        ],
        prob=_prob,
        mask=_mask,
    )  # test int for hetero

    # Enable santity check in distributed sampling.
    os.environ["DGL_DIST_DEBUG"] = "1"

    # We need to test creating DistDataLoader multiple times.
    for i in range(2):
        # Create DataLoader for constructing blocks
        dataloader_cls = (
            dgl.dataloading.DistNodeDataLoader
            if use_deprecated_dataloader
            else dgl.distributed.DistNodeDataLoader
        )
        dataloader = dataloader_cls(
            dist_graph,
            train_nid,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
        )

        for _ in range(2):
            for idx, (_, _, blocks) in zip(
                range(0, num_nodes_to_sample, batch_size), dataloader
            ):
                block = blocks[-1]
                for c_etype in block.canonical_etypes:
                    src_type, _, dst_type = c_etype
                    o_src, o_dst = block.edges(etype=c_etype)
                    src_nodes_id = block.srcnodes[src_type].data[dgl.NID][o_src]
                    dst_nodes_id = block.dstnodes[dst_type].data[dgl.NID][o_dst]
                    src_nodes_id = orig_nid[src_type][src_nodes_id]
                    dst_nodes_id = orig_nid[dst_type][dst_nodes_id]
                    has_edges = groundtruth_g.has_edges_between(
                        src_nodes_id, dst_nodes_id, etype=c_etype
                    )
                    assert np.all(F.asnumpy(has_edges))

                    if use_graphbolt and not return_eids:
                        assert dgl.EID not in block.edges[c_etype].data
                        continue
                    eids = orig_eid[c_etype][block.edges[c_etype].data[dgl.EID]]
                    expected_eids = groundtruth_g.edge_ids(
                        src_nodes_id, dst_nodes_id, etype=c_etype
                    )
                    assert th.equal(
                        eids, expected_eids
                    ), f"{eids} != {expected_eids}"
                    # Verify the prob/mask functionality.
                    if prob_or_mask is not None:
                        prob_data = groundtruth_g.edges[c_etype].data[
                            prob_or_mask
                        ][eids]
                        assert th.all(prob_data > 0)
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
    use_graphbolt,
    exclude,
    reverse_eids,
    reverse_etypes,
    negative,
    prob_or_mask,
    use_deprecated_dataloader=False,
):
    dgl.distributed.initialize(ip_config, use_graphbolt=use_graphbolt)
    gpb = None
    disable_shared_mem = num_server > 1
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(part_config, rank)
    num_edges_to_sample = 202
    batch_size = 32
    graph_name = os.path.splitext(os.path.basename(part_config))[0]
    dist_graph = DistGraph(graph_name, gpb=gpb, part_config=part_config)
    assert len(dist_graph.ntypes) == len(groundtruth_g.ntypes)
    assert len(dist_graph.etypes) == len(groundtruth_g.etypes)
    if len(dist_graph.etypes) == 1:
        train_eid = th.arange(num_edges_to_sample)
    else:
        train_eid = {
            dist_graph.canonical_etypes[0]: th.arange(num_edges_to_sample)
        }

    for i in range(num_server):
        part, _, _, _, _, _, _ = load_partition(part_config, i)

    # Create sampler
    _prob = None
    _mask = None
    if prob_or_mask is None:
        pass
    elif prob_or_mask == "prob":
        _prob = "prob"
    elif prob_or_mask == "mask":
        _mask = "mask"
    else:
        raise ValueError(f"Unsupported prob type: {prob_or_mask}")
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [5, -1], prob=_prob, mask=_mask
    )

    # Negative sampler.
    negative_sampler = None
    if negative:
        negative_sampler = dgl.dataloading.negative_sampler.Uniform(5)

    # We need to test creating DistDataLoader multiple times.
    for i in range(2):
        # Create DataLoader for constructing blocks
        dataloader_cls = (
            dgl.dataloading.DistEdgeDataLoader
            if use_deprecated_dataloader
            else dgl.distributed.DistEdgeDataLoader
        )
        dataloader = dataloader_cls(
            dist_graph,
            train_eid,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            exclude=exclude,
            reverse_eids=reverse_eids,
            reverse_etypes=reverse_etypes,
            negative_sampler=negative_sampler,
        )

        for _ in range(2):
            for _, minibatch in zip(
                range(0, num_edges_to_sample, batch_size), dataloader
            ):
                if negative:
                    _, pos_pair_graph, neg_pair_graph, blocks = minibatch
                else:
                    _, pos_pair_graph, blocks = minibatch
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
                    if negative:
                        assert np.all(
                            F.asnumpy(block.dstnodes[dst_type].data[dgl.NID])
                            == F.asnumpy(
                                neg_pair_graph.nodes[dst_type].data[dgl.NID]
                            )
                        )
                    if (
                        dgl.EID
                        not in block.edges[(src_type, etype, dst_type)].data
                    ):
                        continue
                    sampled_eids = block.edges[
                        (src_type, etype, dst_type)
                    ].data[dgl.EID]
                    sampled_orig_eids = orig_eid[(src_type, etype, dst_type)][
                        sampled_eids
                    ]
                    raw_src, raw_dst = groundtruth_g.find_edges(
                        sampled_orig_eids, etype=(src_type, etype, dst_type)
                    )
                    sampled_src, sampled_dst = block.edges(
                        etype=(src_type, etype, dst_type)
                    )
                    sampled_orig_src = block.nodes[src_type].data[dgl.NID][
                        sampled_src
                    ]
                    sampled_orig_dst = block.nodes[dst_type].data[dgl.NID][
                        sampled_dst
                    ]
                    assert th.equal(
                        raw_src, orig_nid[src_type][sampled_orig_src]
                    )
                    assert th.equal(
                        raw_dst, orig_nid[dst_type][sampled_orig_dst]
                    )
                    # Verify the prob/mask functionality.
                    if prob_or_mask is not None:
                        prob_data = groundtruth_g.edges[etype].data[
                            prob_or_mask
                        ][sampled_orig_eids]
                        assert th.all(prob_data > 0)
                # Verify the exclude functionality.
                if dgl.EID not in blocks[-1].edata.keys():
                    continue
                for (
                    src_type,
                    etype,
                    dst_type,
                ) in pos_pair_graph.canonical_etypes:
                    for block in blocks:
                        if (
                            src_type,
                            etype,
                            dst_type,
                        ) not in block.canonical_etypes:
                            continue
                        current_eids = block.edges[etype].data[dgl.EID]
                        seed_eids = pos_pair_graph.edges[etype].data[dgl.EID]
                        if exclude is None:
                            # seed_eids are not guaranteed to be sampled.
                            pass
                        elif exclude == "self":
                            assert not th.any(th.isin(current_eids, seed_eids))
                        elif exclude == "reverse_id":
                            src, dst = groundtruth_g.find_edges(seed_eids)
                            reverse_seed_eids = groundtruth_g.edge_ids(dst, src)
                            assert not th.any(
                                th.isin(current_eids, reverse_seed_eids)
                            )
                            assert not th.any(th.isin(current_eids, seed_eids))
                        elif exclude == "reverse_types":
                            assert not th.any(th.isin(current_eids, seed_eids))
                            reverse_etype = reverse_etypes[
                                (src_type, etype, dst_type)
                            ]
                            if reverse_etype in block.canonical_etypes:
                                assert not th.any(
                                    th.isin(
                                        block.edges[reverse_etype].data[
                                            dgl.EID
                                        ],
                                        seed_eids,
                                    )
                                )
                        else:
                            raise ValueError(
                                f"Unsupported exclude type: {exclude}"
                            )
    del dataloader
    dgl.distributed.exit_client()


def check_dataloader(
    g,
    num_server,
    num_workers,
    dataloader_type,
    use_graphbolt=False,
    return_eids=False,
    exclude=None,
    reverse_eids=None,
    reverse_etypes=None,
    negative=False,
    prob_or_mask=None,
    use_deprecated_dataloader=False,
):
    with tempfile.TemporaryDirectory() as test_dir:
        ip_config = "ip_config.txt"
        generate_ip_config(ip_config, num_server, num_server)

        num_parts = num_server
        num_hops = 1
        graph_name = f"graph_{uuid.uuid4()}"
        orig_nid, orig_eid = partition_graph(
            g,
            graph_name,
            num_parts,
            test_dir,
            num_hops=num_hops,
            part_method="metis",
            return_mapping=True,
            use_graphbolt=use_graphbolt,
            store_eids=return_eids,
        )
        part_config = os.path.join(test_dir, f"{graph_name}.json")
        if not isinstance(orig_nid, dict):
            orig_nid = {g.ntypes[0]: orig_nid}
        if not isinstance(orig_eid, dict):
            orig_eid = {g.canonical_etypes[0]: orig_eid}

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
                    use_graphbolt,
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
                    use_graphbolt,
                    return_eids,
                    prob_or_mask,
                    use_deprecated_dataloader,
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
                    use_graphbolt,
                    exclude,
                    reverse_eids,
                    reverse_etypes,
                    negative,
                    prob_or_mask,
                    use_deprecated_dataloader,
                ),
            )
            p.start()
            ptrainer_list.append(p)
        for p in pserver_list:
            p.join()
            assert p.exitcode == 0
        for p in ptrainer_list:
            p.join()
            assert p.exitcode == 0


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
    # Add reverse edges.
    src, dst = edges[("n1", "r1", "n2")]
    edges[("n2", "r21", "n1")] = (dst, src)
    g = dgl.heterograph(edges, num_nodes)
    g.nodes["n1"].data["feat"] = F.unsqueeze(F.arange(0, g.num_nodes("n1")), 1)
    g.edges["r1"].data["feat"] = F.unsqueeze(F.arange(0, g.num_edges("r1")), 1)
    return g


@pytest.mark.parametrize("num_server", [1])
@pytest.mark.parametrize("num_workers", [0, 1])
@pytest.mark.parametrize("dataloader_type", ["node", "edge"])
@pytest.mark.parametrize("use_graphbolt", [False, True])
@pytest.mark.parametrize("return_eids", [False, True])
def test_dataloader_homograph(
    num_server, num_workers, dataloader_type, use_graphbolt, return_eids
):
    if not use_graphbolt and return_eids:
        # return_eids is not supported in non-GraphBolt mode.
        return
    reset_envs()
    g = CitationGraphDataset("cora")[0]
    check_dataloader(
        g,
        num_server,
        num_workers,
        dataloader_type,
        use_graphbolt=use_graphbolt,
        return_eids=return_eids,
    )


@pytest.mark.parametrize("num_workers", [0])
@pytest.mark.parametrize("use_graphbolt", [False, True])
@pytest.mark.parametrize("exclude", [None, "self", "reverse_id"])
@pytest.mark.parametrize("negative", [False, True])
def test_edge_dataloader_homograph(
    num_workers, use_graphbolt, exclude, negative
):
    num_server = 1
    dataloader_type = "edge"
    reset_envs()
    g, reverse_eids = _unique_rand_graph()
    check_dataloader(
        g,
        num_server,
        num_workers,
        dataloader_type,
        use_graphbolt=use_graphbolt,
        return_eids=True,
        exclude=exclude,
        reverse_eids=reverse_eids,
        negative=negative,
    )


@pytest.mark.parametrize("num_server", [1])
@pytest.mark.parametrize("num_workers", [1])
@pytest.mark.parametrize("dataloader_type", ["node", "edge"])
@pytest.mark.parametrize("use_graphbolt", [False, True])
@pytest.mark.parametrize("prob_or_mask", ["prob", "mask"])
def test_dataloader_homograph_prob_or_mask(
    num_server, num_workers, dataloader_type, use_graphbolt, prob_or_mask
):
    reset_envs()
    g = CitationGraphDataset("cora")[0]
    prob = th.rand(g.num_edges())
    mask = prob > 0.2
    g.edata["prob"] = F.tensor(prob)
    g.edata["mask"] = F.tensor(mask)
    check_dataloader(
        g,
        num_server,
        num_workers,
        dataloader_type,
        use_graphbolt=use_graphbolt,
        return_eids=True,
        prob_or_mask=prob_or_mask,
    )


@pytest.mark.parametrize("num_server", [1])
@pytest.mark.parametrize("num_workers", [0, 1])
@pytest.mark.parametrize("dataloader_type", ["node", "edge"])
@pytest.mark.parametrize("use_graphbolt", [False, True])
@pytest.mark.parametrize("return_eids", [False, True])
def test_dataloader_heterograph(
    num_server, num_workers, dataloader_type, use_graphbolt, return_eids
):
    if not use_graphbolt and return_eids:
        # return_eids is not supported in non-GraphBolt mode.
        return
    reset_envs()
    g = create_random_hetero()
    check_dataloader(
        g,
        num_server,
        num_workers,
        dataloader_type,
        use_graphbolt=use_graphbolt,
        return_eids=return_eids,
    )


@pytest.mark.parametrize("num_workers", [0])
@pytest.mark.parametrize("use_graphbolt", [False, True])
@pytest.mark.parametrize("exclude", [None, "self", "reverse_types"])
@pytest.mark.parametrize("negative", [False, True])
def test_edge_dataloader_heterograph(
    num_workers, use_graphbolt, exclude, negative
):
    num_server = 1
    dataloader_type = "edge"
    reset_envs()
    g = create_random_hetero()
    reverse_etypes = {("n1", "r1", "n2"): ("n2", "r21", "n1")}
    check_dataloader(
        g,
        num_server,
        num_workers,
        dataloader_type,
        use_graphbolt=use_graphbolt,
        return_eids=True,
        exclude=exclude,
        reverse_etypes=reverse_etypes,
        negative=negative,
    )


@pytest.mark.parametrize("num_server", [1])
@pytest.mark.parametrize("num_workers", [1])
@pytest.mark.parametrize("dataloader_type", ["node", "edge"])
@pytest.mark.parametrize("use_graphbolt", [False, True])
@pytest.mark.parametrize("prob_or_mask", ["prob", "mask"])
def test_dataloader_heterograph_prob_or_mask(
    num_server, num_workers, dataloader_type, use_graphbolt, prob_or_mask
):
    reset_envs()
    g = create_random_hetero()
    for etype in g.canonical_etypes:
        prob = th.rand(g.num_edges(etype))
        mask = prob > prob.median()
        g.edges[etype].data["prob"] = prob
        g.edges[etype].data["mask"] = mask
    check_dataloader(
        g,
        num_server,
        num_workers,
        dataloader_type,
        use_graphbolt=use_graphbolt,
        return_eids=True,
        prob_or_mask=prob_or_mask,
    )


@unittest.skip(reason="Skip due to glitch in CI")
@pytest.mark.parametrize("num_server", [3])
@pytest.mark.parametrize("num_workers", [0, 4])
def test_neg_dataloader(num_server, num_workers):
    reset_envs()
    g = CitationGraphDataset("cora")[0]
    check_neg_dataloader(g, num_server, num_workers)
    g = create_random_hetero()
    check_neg_dataloader(g, num_server, num_workers)


def start_multiple_dataloaders(
    ip_config,
    part_config,
    graph_name,
    orig_g,
    num_dataloaders,
    dataloader_type,
    use_graphbolt,
):
    dgl.distributed.initialize(ip_config)
    dist_g = dgl.distributed.DistGraph(graph_name, part_config=part_config)
    if dataloader_type == "node":
        train_ids = th.arange(orig_g.num_nodes(), dtype=dist_g.idtype)
        batch_size = orig_g.num_nodes() // 100
    else:
        train_ids = th.arange(orig_g.num_edges())
        batch_size = orig_g.num_edges() // 100
    sampler = dgl.dataloading.NeighborSampler([-1])
    dataloaders = []
    dl_iters = []
    for _ in range(num_dataloaders):
        if dataloader_type == "node":
            dataloader = dgl.distributed.DistNodeDataLoader(
                dist_g, train_ids, sampler, batch_size=batch_size
            )
        else:
            dataloader = dgl.distributed.DistEdgeDataLoader(
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


@pytest.mark.parametrize("num_dataloaders", [4])
@pytest.mark.parametrize("num_workers", [0])
@pytest.mark.parametrize("dataloader_type", ["node", "edge"])
@pytest.mark.parametrize("use_graphbolt", [False, True])
def test_multiple_dist_dataloaders(
    num_dataloaders, num_workers, dataloader_type, use_graphbolt
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
        graph_name = f"graph_{uuid.uuid4()}"
        partition_graph(
            orig_g,
            graph_name,
            num_parts,
            test_dir,
            use_graphbolt=use_graphbolt,
        )
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
                    use_graphbolt,
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
                use_graphbolt,
            ),
        )
        p_client.start()

        p_client.join()
        assert p_client.exitcode == 0
        for p in p_servers:
            p.join()
            assert p.exitcode == 0
    reset_envs()


@pytest.mark.parametrize("dataloader_type", ["node", "edge"])
def test_deprecated_dataloader(dataloader_type):
    reset_envs()
    g = CitationGraphDataset("cora")[0]
    check_dataloader(
        g,
        1,
        0,
        dataloader_type,
        use_deprecated_dataloader=True,
    )
