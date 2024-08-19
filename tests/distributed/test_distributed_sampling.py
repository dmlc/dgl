import multiprocessing as mp
import os
import random
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import backend as F
import dgl
import numpy as np
import pytest
import torch
from dgl.data import CitationGraphDataset, WN18Dataset
from dgl.distributed import (
    DistGraph,
    DistGraphServer,
    load_partition,
    load_partition_book,
    partition_graph,
    sample_etype_neighbors,
    sample_neighbors,
)

from dgl.distributed.graph_partition_book import _etype_tuple_to_str

from scipy import sparse as spsp
from utils import generate_ip_config, reset_envs


def start_server(
    rank,
    tmpdir,
    disable_shared_mem,
    graph_name,
    graph_format=["csc", "coo"],
    use_graphbolt=False,
):
    g = DistGraphServer(
        rank,
        "rpc_ip_config.txt",
        1,
        1,
        tmpdir / (graph_name + ".json"),
        disable_shared_mem=disable_shared_mem,
        graph_format=graph_format,
        use_graphbolt=use_graphbolt,
    )
    g.start()


def start_sample_client(rank, tmpdir, disable_shared_mem):
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(
            tmpdir / "test_sampling.json", rank
        )
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_sampling", gpb=gpb)
    try:
        sampled_graph = sample_neighbors(
            dist_graph,
            torch.tensor([0, 10, 99, 66, 1024, 2008], dtype=dist_graph.idtype),
            3,
        )
    except Exception as e:
        print(traceback.format_exc())
        sampled_graph = None
    dgl.distributed.exit_client()
    return sampled_graph


def start_sample_client_shuffle(
    rank,
    tmpdir,
    disable_shared_mem,
    g,
    num_servers,
    group_id,
    orig_nid,
    orig_eid,
    use_graphbolt=False,
    return_eids=False,
    node_id_dtype=None,
    replace=False,
):
    os.environ["DGL_GROUP_ID"] = str(group_id)
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(
            tmpdir / "test_sampling.json", rank
        )
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_sampling", gpb=gpb)
    sampled_graph = sample_neighbors(
        dist_graph,
        torch.tensor([0, 10, 99, 66, 1024, 2008], dtype=node_id_dtype),
        3,
        replace=replace,
        use_graphbolt=use_graphbolt,
    )
    assert sampled_graph.idtype == dist_graph.idtype
    assert sampled_graph.idtype == torch.int64

    assert (
        dgl.ETYPE not in sampled_graph.edata
    ), "Etype should not be in homogeneous sampled graph."
    src, dst = sampled_graph.edges()
    sampled_in_degrees = sampled_graph.in_degrees(dst)
    src = orig_nid[src]
    dst = orig_nid[dst]
    assert sampled_graph.num_nodes() == g.num_nodes()
    assert np.all(F.asnumpy(g.has_edges_between(src, dst)))
    if use_graphbolt and not return_eids:
        assert (
            dgl.EID not in sampled_graph.edata
        ), "EID should not be in sampled graph if use_graphbolt=True."
    else:
        eids = g.edge_ids(src, dst)
        eids1 = orig_eid[sampled_graph.edata[dgl.EID]]
        assert np.array_equal(F.asnumpy(eids1), F.asnumpy(eids))
    # Verify replace argument.
    orig_in_degrees = g.in_degrees(dst)
    if replace:
        assert torch.all(
            (sampled_in_degrees == 3) | (sampled_in_degrees == orig_in_degrees)
        )
    else:
        assert torch.all(sampled_in_degrees <= 3)


def start_find_edges_client(rank, tmpdir, disable_shared_mem, eids, etype=None):
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(
            tmpdir / "test_find_edges.json", rank
        )
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_find_edges", gpb=gpb)
    try:
        u, v = dist_graph.find_edges(eids, etype=etype)
    except Exception as e:
        print(traceback.format_exc())
        u, v = None, None
    dgl.distributed.exit_client()
    return u, v


def start_get_degrees_client(rank, tmpdir, disable_shared_mem, nids=None):
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(
            tmpdir / "test_get_degrees.json", rank
        )
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_get_degrees", gpb=gpb)
    try:
        in_deg = dist_graph.in_degrees(nids)
        all_in_deg = dist_graph.in_degrees()
        out_deg = dist_graph.out_degrees(nids)
        all_out_deg = dist_graph.out_degrees()
    except Exception as e:
        print(traceback.format_exc())
        in_deg, out_deg, all_in_deg, all_out_deg = None, None, None, None
    dgl.distributed.exit_client()
    return in_deg, out_deg, all_in_deg, all_out_deg


def check_rpc_sampling(tmpdir, num_server):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = CitationGraphDataset("cora")[0]
    print(g.idtype)
    num_parts = num_server
    num_hops = 1

    partition_graph(
        g,
        "test_sampling",
        num_parts,
        tmpdir,
        num_hops=num_hops,
        part_method="metis",
    )

    pserver_list = []
    ctx = mp.get_context("spawn")
    for i in range(num_server):
        p = ctx.Process(
            target=start_server,
            args=(i, tmpdir, num_server > 1, "test_sampling"),
        )
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    sampled_graph = start_sample_client(0, tmpdir, num_server > 1)
    print("Done sampling")
    for p in pserver_list:
        p.join()
        assert p.exitcode == 0

    src, dst = sampled_graph.edges()
    assert sampled_graph.num_nodes() == g.num_nodes()
    assert np.all(F.asnumpy(g.has_edges_between(src, dst)))
    eids = g.edge_ids(src, dst)
    assert np.array_equal(
        F.asnumpy(sampled_graph.edata[dgl.EID]), F.asnumpy(eids)
    )


def check_rpc_find_edges_shuffle(tmpdir, num_server):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = CitationGraphDataset("cora")[0]
    num_parts = num_server

    orig_nid, orig_eid = partition_graph(
        g,
        "test_find_edges",
        num_parts,
        tmpdir,
        num_hops=1,
        part_method="metis",
        return_mapping=True,
    )

    pserver_list = []
    ctx = mp.get_context("spawn")
    for i in range(num_server):
        p = ctx.Process(
            target=start_server,
            args=(i, tmpdir, num_server > 1, "test_find_edges", ["csr", "coo"]),
        )
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    eids = F.tensor(np.random.randint(g.num_edges(), size=100))
    u, v = g.find_edges(orig_eid[eids])
    du, dv = start_find_edges_client(0, tmpdir, num_server > 1, eids)
    du = orig_nid[du]
    dv = orig_nid[dv]
    assert F.array_equal(u, du)
    assert F.array_equal(v, dv)


def create_random_hetero(dense=False, empty=False):
    num_nodes = (
        {"n1": 210, "n2": 200, "n3": 220}
        if dense
        else {"n1": 1010, "n2": 1000, "n3": 1020}
    )
    etypes = [("n1", "r12", "n2"), ("n1", "r13", "n3"), ("n2", "r23", "n3")]
    edges = {}
    random.seed(42)
    for etype in etypes:
        src_ntype, _, dst_ntype = etype
        arr = spsp.random(
            num_nodes[src_ntype] - 10 if empty else num_nodes[src_ntype],
            num_nodes[dst_ntype] - 10 if empty else num_nodes[dst_ntype],
            density=0.1 if dense else 0.001,
            format="coo",
            random_state=100,
        )
        edges[etype] = (arr.row, arr.col)
    g = dgl.heterograph(edges, num_nodes)
    g.nodes["n1"].data["feat"] = F.ones(
        (g.num_nodes("n1"), 10), F.float32, F.cpu()
    )
    return g


def check_rpc_hetero_find_edges_shuffle(tmpdir, num_server):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = create_random_hetero()
    num_parts = num_server

    orig_nid, orig_eid = partition_graph(
        g,
        "test_find_edges",
        num_parts,
        tmpdir,
        num_hops=1,
        part_method="metis",
        return_mapping=True,
    )

    pserver_list = []
    ctx = mp.get_context("spawn")
    for i in range(num_server):
        p = ctx.Process(
            target=start_server,
            args=(i, tmpdir, num_server > 1, "test_find_edges", ["csr", "coo"]),
        )
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    test_etype = g.to_canonical_etype("r12")
    eids = F.tensor(np.random.randint(g.num_edges(test_etype), size=100))
    expect_except = False
    try:
        _, _ = g.find_edges(orig_eid[test_etype][eids], etype=("n1", "r12"))
    except:
        expect_except = True
    assert expect_except
    u, v = g.find_edges(orig_eid[test_etype][eids], etype="r12")
    u1, v1 = g.find_edges(orig_eid[test_etype][eids], etype=("n1", "r12", "n2"))
    assert F.array_equal(u, u1)
    assert F.array_equal(v, v1)
    du, dv = start_find_edges_client(
        0, tmpdir, num_server > 1, eids, etype="r12"
    )
    du = orig_nid["n1"][du]
    dv = orig_nid["n2"][dv]
    assert F.array_equal(u, du)
    assert F.array_equal(v, dv)


# Wait non shared memory graph store
@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow",
    reason="Not support tensorflow for now",
)
@unittest.skipIf(
    dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support"
)
@pytest.mark.parametrize("num_server", [1])
def test_rpc_find_edges_shuffle(num_server):
    reset_envs()
    import tempfile

    os.environ["DGL_DIST_MODE"] = "distributed"
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_hetero_find_edges_shuffle(Path(tmpdirname), num_server)
        check_rpc_find_edges_shuffle(Path(tmpdirname), num_server)


def check_rpc_get_degree_shuffle(tmpdir, num_server):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = CitationGraphDataset("cora")[0]
    num_parts = num_server

    orig_nid, _ = partition_graph(
        g,
        "test_get_degrees",
        num_parts,
        tmpdir,
        num_hops=1,
        part_method="metis",
        return_mapping=True,
    )

    pserver_list = []
    ctx = mp.get_context("spawn")
    for i in range(num_server):
        p = ctx.Process(
            target=start_server,
            args=(i, tmpdir, num_server > 1, "test_get_degrees"),
        )
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    nids = F.tensor(np.random.randint(g.num_nodes(), size=100))
    in_degs, out_degs, all_in_degs, all_out_degs = start_get_degrees_client(
        0, tmpdir, num_server > 1, nids
    )

    print("Done get_degree")
    for p in pserver_list:
        p.join()
        assert p.exitcode == 0

    print("check results")
    assert F.array_equal(g.in_degrees(orig_nid[nids]), in_degs)
    assert F.array_equal(g.in_degrees(orig_nid), all_in_degs)
    assert F.array_equal(g.out_degrees(orig_nid[nids]), out_degs)
    assert F.array_equal(g.out_degrees(orig_nid), all_out_degs)


# Wait non shared memory graph store
@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow",
    reason="Not support tensorflow for now",
)
@unittest.skipIf(
    dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support"
)
@pytest.mark.parametrize("num_server", [1])
def test_rpc_get_degree_shuffle(num_server):
    reset_envs()
    import tempfile

    os.environ["DGL_DIST_MODE"] = "distributed"
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_get_degree_shuffle(Path(tmpdirname), num_server)


# @unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
# @unittest.skipIf(dgl.backend.backend_name == 'tensorflow', reason='Not support tensorflow for now')
@unittest.skip("Only support partition with shuffle")
def test_rpc_sampling():
    reset_envs()
    import tempfile

    os.environ["DGL_DIST_MODE"] = "distributed"
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_sampling(Path(tmpdirname), 1)


def check_rpc_sampling_shuffle(
    tmpdir,
    num_server,
    num_groups=1,
    use_graphbolt=False,
    return_eids=False,
    node_id_dtype=None,
    replace=False,
):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = CitationGraphDataset("cora")[0]
    num_parts = num_server
    num_hops = 1

    orig_nids, orig_eids = partition_graph(
        g,
        "test_sampling",
        num_parts,
        tmpdir,
        num_hops=num_hops,
        part_method="metis",
        return_mapping=True,
        use_graphbolt=use_graphbolt,
        store_eids=return_eids,
    )

    pserver_list = []
    ctx = mp.get_context("spawn")
    for i in range(num_server):
        p = ctx.Process(
            target=start_server,
            args=(
                i,
                tmpdir,
                num_server > 1,
                "test_sampling",
                ["csc", "coo"],
                use_graphbolt,
            ),
        )
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    pclient_list = []
    num_clients = 1
    for client_id in range(num_clients):
        for group_id in range(num_groups):
            p = ctx.Process(
                target=start_sample_client_shuffle,
                args=(
                    client_id,
                    tmpdir,
                    num_server > 1,
                    g,
                    num_server,
                    group_id,
                    orig_nids,
                    orig_eids,
                    use_graphbolt,
                    return_eids,
                    node_id_dtype,
                    replace,
                ),
            )
            p.start()
            time.sleep(1)  # avoid race condition when instantiating DistGraph
            pclient_list.append(p)
    for p in pclient_list:
        p.join()
        assert p.exitcode == 0
    for p in pserver_list:
        p.join()
        assert p.exitcode == 0


def start_hetero_sample_client(
    rank,
    tmpdir,
    disable_shared_mem,
    nodes,
    use_graphbolt=False,
    return_eids=False,
    replace=False,
):
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(
            tmpdir / "test_sampling.json", rank
        )
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_sampling", gpb=gpb)
    assert "feat" in dist_graph.nodes["n1"].data
    assert "feat" not in dist_graph.nodes["n2"].data
    assert "feat" not in dist_graph.nodes["n3"].data
    nodes = {
        k: v.type(dist_graph.idtype).clone().detach() for k, v in nodes.items()
    }
    if gpb is None:
        gpb = dist_graph.get_partition_book()
    try:
        # Enable santity check in distributed sampling.
        os.environ["DGL_DIST_DEBUG"] = "1"
        sampled_graph = sample_neighbors(
            dist_graph, nodes, 3, replace=replace, use_graphbolt=use_graphbolt
        )
        block = dgl.to_block(sampled_graph, nodes)
        if not use_graphbolt or return_eids:
            block.edata[dgl.EID] = sampled_graph.edata[dgl.EID]
    except Exception as e:
        print(traceback.format_exc())
        block = None
    dgl.distributed.exit_client()
    return block, gpb


def start_hetero_etype_sample_client(
    rank,
    tmpdir,
    disable_shared_mem,
    fanout=3,
    nodes=None,
    etype_sorted=False,
    use_graphbolt=False,
    return_eids=False,
):
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(
            tmpdir / "test_sampling.json", rank
        )
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_sampling", gpb=gpb)
    assert "feat" in dist_graph.nodes["n1"].data
    assert "feat" not in dist_graph.nodes["n2"].data
    assert "feat" not in dist_graph.nodes["n3"].data
    nodes = {
        k: v.type(dist_graph.idtype).clone().detach() for k, v in nodes.items()
    }

    if (not use_graphbolt) and dist_graph.local_partition is not None:
        # Check whether etypes are sorted in dist_graph
        local_g = dist_graph.local_partition
        local_nids = np.arange(local_g.num_nodes())
        for lnid in local_nids:
            leids = local_g.in_edges(lnid, form="eid")
            letids = F.asnumpy(local_g.edata[dgl.ETYPE][leids])
            _, idices = np.unique(letids, return_index=True)
            assert np.all(idices[:-1] <= idices[1:])

    if gpb is None:
        gpb = dist_graph.get_partition_book()
    try:
        # Enable santity check in distributed sampling.
        os.environ["DGL_DIST_DEBUG"] = "1"
        sampled_graph = sample_etype_neighbors(
            dist_graph,
            nodes,
            fanout,
            etype_sorted=etype_sorted,
            use_graphbolt=use_graphbolt,
        )
        block = dgl.to_block(sampled_graph, nodes)
        if sampled_graph.num_edges() > 0:
            if not use_graphbolt or return_eids:
                block.edata[dgl.EID] = sampled_graph.edata[dgl.EID]
    except Exception as e:
        print(traceback.format_exc())
        block = None
    dgl.distributed.exit_client()
    return block, gpb


def check_rpc_hetero_sampling_shuffle(
    tmpdir, num_server, use_graphbolt=False, return_eids=False, replace=False
):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = create_random_hetero()
    num_parts = num_server
    num_hops = 1

    orig_nid_map, orig_eid_map = partition_graph(
        g,
        "test_sampling",
        num_parts,
        tmpdir,
        num_hops=num_hops,
        part_method="metis",
        return_mapping=True,
        use_graphbolt=use_graphbolt,
        store_eids=return_eids,
    )

    pserver_list = []
    ctx = mp.get_context("spawn")
    for i in range(num_server):
        p = ctx.Process(
            target=start_server,
            args=(
                i,
                tmpdir,
                num_server > 1,
                "test_sampling",
                ["csc", "coo"],
                use_graphbolt,
            ),
        )
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    nodes = {"n3": torch.tensor([0, 10, 99, 66, 124, 208], dtype=g.idtype)}
    block, gpb = start_hetero_sample_client(
        0,
        tmpdir,
        num_server > 1,
        nodes=nodes,
        use_graphbolt=use_graphbolt,
        return_eids=return_eids,
        replace=replace,
    )
    for p in pserver_list:
        p.join()
        assert p.exitcode == 0

    for c_etype in block.canonical_etypes:
        src_type, etype, dst_type = c_etype
        src, dst = block.edges(etype=etype)
        # These are global Ids after shuffling.
        shuffled_src = F.gather_row(block.srcnodes[src_type].data[dgl.NID], src)
        shuffled_dst = F.gather_row(block.dstnodes[dst_type].data[dgl.NID], dst)
        orig_src = F.asnumpy(F.gather_row(orig_nid_map[src_type], shuffled_src))
        orig_dst = F.asnumpy(F.gather_row(orig_nid_map[dst_type], shuffled_dst))

        assert np.all(
            F.asnumpy(g.has_edges_between(orig_src, orig_dst, etype=etype))
        )

        if use_graphbolt and not return_eids:
            continue

        shuffled_eid = block.edges[etype].data[dgl.EID]
        orig_eid = F.asnumpy(F.gather_row(orig_eid_map[c_etype], shuffled_eid))

        # Check the node Ids and edge Ids.
        orig_src1, orig_dst1 = g.find_edges(orig_eid, etype=etype)
        assert np.all(F.asnumpy(orig_src1) == orig_src)
        assert np.all(F.asnumpy(orig_dst1) == orig_dst)


def get_degrees(g, nids, ntype):
    deg = F.zeros((len(nids),), dtype=F.int64)
    for srctype, etype, dsttype in g.canonical_etypes:
        if srctype == ntype:
            deg += g.out_degrees(u=nids, etype=etype)
        elif dsttype == ntype:
            deg += g.in_degrees(v=nids, etype=etype)
    return deg


def check_rpc_hetero_sampling_empty_shuffle(
    tmpdir, num_server, use_graphbolt=False, return_eids=False
):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = create_random_hetero(empty=True)
    num_parts = num_server
    num_hops = 1

    orig_nids, _ = partition_graph(
        g,
        "test_sampling",
        num_parts,
        tmpdir,
        num_hops=num_hops,
        part_method="metis",
        return_mapping=True,
        use_graphbolt=use_graphbolt,
        store_eids=return_eids,
    )

    pserver_list = []
    ctx = mp.get_context("spawn")
    for i in range(num_server):
        p = ctx.Process(
            target=start_server,
            args=(
                i,
                tmpdir,
                num_server > 1,
                "test_sampling",
                ["csc", "coo"],
                use_graphbolt,
            ),
        )
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    deg = get_degrees(g, orig_nids["n3"], "n3")
    empty_nids = F.nonzero_1d(deg == 0).to(g.idtype)
    block, gpb = start_hetero_sample_client(
        0,
        tmpdir,
        num_server > 1,
        nodes={"n3": empty_nids},
        use_graphbolt=use_graphbolt,
        return_eids=return_eids,
    )
    for p in pserver_list:
        p.join()
        assert p.exitcode == 0

    assert block.num_edges() == 0
    assert len(block.etypes) == len(g.etypes)


def check_rpc_hetero_etype_sampling_shuffle(
    tmpdir,
    num_server,
    graph_formats=None,
    use_graphbolt=False,
    return_eids=False,
):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = create_random_hetero(dense=True)
    num_parts = num_server
    num_hops = 1

    orig_nid_map, orig_eid_map = partition_graph(
        g,
        "test_sampling",
        num_parts,
        tmpdir,
        num_hops=num_hops,
        part_method="metis",
        return_mapping=True,
        graph_formats=graph_formats,
        use_graphbolt=use_graphbolt,
        store_eids=return_eids,
    )

    pserver_list = []
    ctx = mp.get_context("spawn")
    for i in range(num_server):
        p = ctx.Process(
            target=start_server,
            args=(
                i,
                tmpdir,
                num_server > 1,
                "test_sampling",
                ["csc", "coo"],
                use_graphbolt,
            ),
        )
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    fanout = {etype: 3 for etype in g.canonical_etypes}
    etype_sorted = False
    if graph_formats is not None:
        etype_sorted = "csc" in graph_formats or "csr" in graph_formats
    nodes = {"n3": torch.tensor([0, 10, 99, 66, 124, 208], dtype=g.idtype)}
    block, gpb = start_hetero_etype_sample_client(
        0,
        tmpdir,
        num_server > 1,
        fanout,
        nodes=nodes,
        etype_sorted=etype_sorted,
        use_graphbolt=use_graphbolt,
        return_eids=return_eids,
    )
    print("Done sampling")
    for p in pserver_list:
        p.join()
        assert p.exitcode == 0

    src, dst = block.edges(etype=("n1", "r13", "n3"))
    assert len(src) == 18
    src, dst = block.edges(etype=("n2", "r23", "n3"))
    assert len(src) == 18

    for c_etype in block.canonical_etypes:
        src_type, etype, dst_type = c_etype
        src, dst = block.edges(etype=etype)
        # These are global Ids after shuffling.
        shuffled_src = F.gather_row(block.srcnodes[src_type].data[dgl.NID], src)
        shuffled_dst = F.gather_row(block.dstnodes[dst_type].data[dgl.NID], dst)
        orig_src = F.asnumpy(F.gather_row(orig_nid_map[src_type], shuffled_src))
        orig_dst = F.asnumpy(F.gather_row(orig_nid_map[dst_type], shuffled_dst))
        assert np.all(
            F.asnumpy(g.has_edges_between(orig_src, orig_dst, etype=etype))
        )

        if use_graphbolt and not return_eids:
            continue

        # Check the node Ids and edge Ids.
        shuffled_eid = block.edges[etype].data[dgl.EID]
        orig_eid = F.asnumpy(F.gather_row(orig_eid_map[c_etype], shuffled_eid))
        orig_src1, orig_dst1 = g.find_edges(orig_eid, etype=etype)
        assert np.all(F.asnumpy(orig_src1) == orig_src)
        assert np.all(F.asnumpy(orig_dst1) == orig_dst)


def check_rpc_hetero_etype_sampling_empty_shuffle(
    tmpdir, num_server, use_graphbolt=False, return_eids=False
):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = create_random_hetero(dense=True, empty=True)
    num_parts = num_server
    num_hops = 1

    orig_nids, _ = partition_graph(
        g,
        "test_sampling",
        num_parts,
        tmpdir,
        num_hops=num_hops,
        part_method="metis",
        return_mapping=True,
        use_graphbolt=use_graphbolt,
        store_eids=return_eids,
    )

    pserver_list = []
    ctx = mp.get_context("spawn")
    for i in range(num_server):
        p = ctx.Process(
            target=start_server,
            args=(
                i,
                tmpdir,
                num_server > 1,
                "test_sampling",
                ["csc", "coo"],
                use_graphbolt,
            ),
        )
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    fanout = 3
    deg = get_degrees(g, orig_nids["n3"], "n3")
    empty_nids = F.nonzero_1d(deg == 0).to(g.idtype)
    block, gpb = start_hetero_etype_sample_client(
        0,
        tmpdir,
        num_server > 1,
        fanout,
        nodes={"n3": empty_nids},
        use_graphbolt=use_graphbolt,
        return_eids=return_eids,
    )
    print("Done sampling")
    for p in pserver_list:
        p.join()
        assert p.exitcode == 0

    assert block.num_edges() == 0
    assert len(block.etypes) == len(g.etypes)


def create_random_bipartite():
    g = dgl.rand_bipartite("user", "buys", "game", 500, 1000, 1000)
    g.nodes["user"].data["feat"] = F.ones(
        (g.num_nodes("user"), 10), F.float32, F.cpu()
    )
    g.nodes["game"].data["feat"] = F.ones(
        (g.num_nodes("game"), 10), F.float32, F.cpu()
    )
    return g


def start_bipartite_sample_client(
    rank,
    tmpdir,
    disable_shared_mem,
    nodes,
    use_graphbolt=False,
    return_eids=False,
):
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(
            tmpdir / "test_sampling.json", rank
        )
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_sampling", gpb=gpb)
    assert "feat" in dist_graph.nodes["user"].data
    assert "feat" in dist_graph.nodes["game"].data
    nodes = {
        k: v.type(dist_graph.idtype).clone().detach() for k, v in nodes.items()
    }
    if gpb is None:
        gpb = dist_graph.get_partition_book()
    # Enable santity check in distributed sampling.
    os.environ["DGL_DIST_DEBUG"] = "1"
    sampled_graph = sample_neighbors(
        dist_graph, nodes, 3, use_graphbolt=use_graphbolt
    )
    block = dgl.to_block(sampled_graph, nodes)
    if sampled_graph.num_edges() > 0:
        if not use_graphbolt or return_eids:
            block.edata[dgl.EID] = sampled_graph.edata[dgl.EID]
    dgl.distributed.exit_client()
    return block, gpb


def start_bipartite_etype_sample_client(
    rank,
    tmpdir,
    disable_shared_mem,
    fanout=3,
    nodes={},
    use_graphbolt=False,
    return_eids=False,
):
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(
            tmpdir / "test_sampling.json", rank
        )
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_sampling", gpb=gpb)
    assert "feat" in dist_graph.nodes["user"].data
    assert "feat" in dist_graph.nodes["game"].data
    nodes = {
        k: v.type(dist_graph.idtype).clone().detach() for k, v in nodes.items()
    }

    if not use_graphbolt and dist_graph.local_partition is not None:
        # Check whether etypes are sorted in dist_graph
        local_g = dist_graph.local_partition
        local_nids = np.arange(local_g.num_nodes())
        for lnid in local_nids:
            leids = local_g.in_edges(lnid, form="eid")
            letids = F.asnumpy(local_g.edata[dgl.ETYPE][leids])
            _, idices = np.unique(letids, return_index=True)
            assert np.all(idices[:-1] <= idices[1:])

    if gpb is None:
        gpb = dist_graph.get_partition_book()
    sampled_graph = sample_etype_neighbors(
        dist_graph, nodes, fanout, use_graphbolt=use_graphbolt
    )
    block = dgl.to_block(sampled_graph, nodes)
    if sampled_graph.num_edges() > 0:
        if not use_graphbolt or return_eids:
            block.edata[dgl.EID] = sampled_graph.edata[dgl.EID]
    dgl.distributed.exit_client()
    return block, gpb


def check_rpc_bipartite_sampling_empty(
    tmpdir, num_server, use_graphbolt=False, return_eids=False
):
    """sample on bipartite via sample_neighbors() which yields empty sample results"""
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = create_random_bipartite()
    num_parts = num_server
    num_hops = 1

    orig_nids, _ = partition_graph(
        g,
        "test_sampling",
        num_parts,
        tmpdir,
        num_hops=num_hops,
        part_method="metis",
        return_mapping=True,
        use_graphbolt=use_graphbolt,
        store_eids=return_eids,
    )

    pserver_list = []
    ctx = mp.get_context("spawn")
    for i in range(num_server):
        p = ctx.Process(
            target=start_server,
            args=(
                i,
                tmpdir,
                num_server > 1,
                "test_sampling",
                ["csc", "coo"],
                use_graphbolt,
            ),
        )
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    deg = get_degrees(g, orig_nids["game"], "game")
    empty_nids = F.nonzero_1d(deg == 0).to(g.idtype)
    nodes = {"game": empty_nids, "user": torch.tensor([1], dtype=g.idtype)}
    block, _ = start_bipartite_sample_client(
        0,
        tmpdir,
        num_server > 1,
        nodes=nodes,
        use_graphbolt=use_graphbolt,
        return_eids=return_eids,
    )

    print("Done sampling")
    for p in pserver_list:
        p.join()
        assert p.exitcode == 0

    assert block.num_edges() == 0
    assert len(block.etypes) == len(g.etypes)


def check_rpc_bipartite_sampling_shuffle(
    tmpdir, num_server, use_graphbolt=False, return_eids=False
):
    """sample on bipartite via sample_neighbors() which yields non-empty sample results"""
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = create_random_bipartite()
    num_parts = num_server
    num_hops = 1

    orig_nid_map, orig_eid_map = partition_graph(
        g,
        "test_sampling",
        num_parts,
        tmpdir,
        num_hops=num_hops,
        part_method="metis",
        return_mapping=True,
        use_graphbolt=use_graphbolt,
        store_eids=return_eids,
    )

    pserver_list = []
    ctx = mp.get_context("spawn")
    for i in range(num_server):
        p = ctx.Process(
            target=start_server,
            args=(
                i,
                tmpdir,
                num_server > 1,
                "test_sampling",
                ["csc", "coo"],
                use_graphbolt,
            ),
        )
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    deg = get_degrees(g, orig_nid_map["game"], "game")
    nids = F.nonzero_1d(deg > 0)
    nodes = {"game": nids, "user": torch.tensor([0], dtype=g.idtype)}
    block, gpb = start_bipartite_sample_client(
        0,
        tmpdir,
        num_server > 1,
        nodes=nodes,
        use_graphbolt=use_graphbolt,
        return_eids=return_eids,
    )
    print("Done sampling")
    for p in pserver_list:
        p.join()
        assert p.exitcode == 0

    for c_etype in block.canonical_etypes:
        src_type, etype, dst_type = c_etype
        src, dst = block.edges(etype=etype)
        # These are global Ids after shuffling.
        shuffled_src = F.gather_row(block.srcnodes[src_type].data[dgl.NID], src)
        shuffled_dst = F.gather_row(block.dstnodes[dst_type].data[dgl.NID], dst)
        orig_src = F.asnumpy(F.gather_row(orig_nid_map[src_type], shuffled_src))
        orig_dst = F.asnumpy(F.gather_row(orig_nid_map[dst_type], shuffled_dst))
        assert np.all(
            F.asnumpy(g.has_edges_between(orig_src, orig_dst, etype=etype))
        )

        if use_graphbolt and not return_eids:
            continue

        shuffled_eid = block.edges[etype].data[dgl.EID]
        orig_eid = F.asnumpy(F.gather_row(orig_eid_map[c_etype], shuffled_eid))

        # Check the node Ids and edge Ids.
        orig_src1, orig_dst1 = g.find_edges(orig_eid, etype=etype)
        assert np.all(F.asnumpy(orig_src1) == orig_src)
        assert np.all(F.asnumpy(orig_dst1) == orig_dst)


def check_rpc_bipartite_etype_sampling_empty(
    tmpdir, num_server, use_graphbolt=False, return_eids=False
):
    """sample on bipartite via sample_etype_neighbors() which yields empty sample results"""
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = create_random_bipartite()
    num_parts = num_server
    num_hops = 1

    orig_nids, _ = partition_graph(
        g,
        "test_sampling",
        num_parts,
        tmpdir,
        num_hops=num_hops,
        part_method="metis",
        return_mapping=True,
        use_graphbolt=use_graphbolt,
        store_eids=return_eids,
    )

    pserver_list = []
    ctx = mp.get_context("spawn")
    for i in range(num_server):
        p = ctx.Process(
            target=start_server,
            args=(
                i,
                tmpdir,
                num_server > 1,
                "test_sampling",
                ["csc", "coo"],
                use_graphbolt,
            ),
        )
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    deg = get_degrees(g, orig_nids["game"], "game")
    empty_nids = F.nonzero_1d(deg == 0).to(g.idtype)
    nodes = {"game": empty_nids, "user": torch.tensor([1], dtype=g.idtype)}
    block, _ = start_bipartite_etype_sample_client(
        0,
        tmpdir,
        num_server > 1,
        nodes=nodes,
        use_graphbolt=use_graphbolt,
        return_eids=return_eids,
    )

    print("Done sampling")
    for p in pserver_list:
        p.join()
        assert p.exitcode == 0

    assert block is not None
    assert block.num_edges() == 0
    assert len(block.etypes) == len(g.etypes)


def check_rpc_bipartite_etype_sampling_shuffle(
    tmpdir, num_server, use_graphbolt=False, return_eids=False
):
    """sample on bipartite via sample_etype_neighbors() which yields non-empty sample results"""
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = create_random_bipartite()
    num_parts = num_server
    num_hops = 1

    orig_nid_map, orig_eid_map = partition_graph(
        g,
        "test_sampling",
        num_parts,
        tmpdir,
        num_hops=num_hops,
        part_method="metis",
        return_mapping=True,
        use_graphbolt=use_graphbolt,
        store_eids=return_eids,
    )

    pserver_list = []
    ctx = mp.get_context("spawn")
    for i in range(num_server):
        p = ctx.Process(
            target=start_server,
            args=(
                i,
                tmpdir,
                num_server > 1,
                "test_sampling",
                ["csc", "coo"],
                use_graphbolt,
            ),
        )
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    fanout = 3
    deg = get_degrees(g, orig_nid_map["game"], "game")
    nids = F.nonzero_1d(deg > 0)
    nodes = {"game": nids, "user": torch.tensor([0], dtype=g.idtype)}
    block, gpb = start_bipartite_etype_sample_client(
        0,
        tmpdir,
        num_server > 1,
        fanout,
        nodes=nodes,
        use_graphbolt=use_graphbolt,
        return_eids=return_eids,
    )
    print("Done sampling")
    for p in pserver_list:
        p.join()
        assert p.exitcode == 0

    for c_etype in block.canonical_etypes:
        src_type, etype, dst_type = c_etype
        src, dst = block.edges(etype=etype)
        # These are global Ids after shuffling.
        shuffled_src = F.gather_row(block.srcnodes[src_type].data[dgl.NID], src)
        shuffled_dst = F.gather_row(block.dstnodes[dst_type].data[dgl.NID], dst)
        orig_src = F.asnumpy(F.gather_row(orig_nid_map[src_type], shuffled_src))
        orig_dst = F.asnumpy(F.gather_row(orig_nid_map[dst_type], shuffled_dst))
        assert np.all(
            F.asnumpy(g.has_edges_between(orig_src, orig_dst, etype=etype))
        )

        if use_graphbolt and not return_eids:
            continue

        # Check the node Ids and edge Ids.
        shuffled_eid = block.edges[etype].data[dgl.EID]
        orig_eid = F.asnumpy(F.gather_row(orig_eid_map[c_etype], shuffled_eid))
        orig_src1, orig_dst1 = g.find_edges(orig_eid, etype=etype)
        assert np.all(F.asnumpy(orig_src1) == orig_src)
        assert np.all(F.asnumpy(orig_dst1) == orig_dst)


@pytest.mark.parametrize("num_server", [1])
@pytest.mark.parametrize("use_graphbolt", [False, True])
@pytest.mark.parametrize("return_eids", [False, True])
@pytest.mark.parametrize("node_id_dtype", [torch.int64])
@pytest.mark.parametrize("replace", [False, True])
def test_rpc_sampling_shuffle(
    num_server, use_graphbolt, return_eids, node_id_dtype, replace
):
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_sampling_shuffle(
            Path(tmpdirname),
            num_server,
            use_graphbolt=use_graphbolt,
            return_eids=return_eids,
            node_id_dtype=node_id_dtype,
            replace=replace,
        )


@pytest.mark.parametrize("num_server", [1])
@pytest.mark.parametrize("use_graphbolt,", [False, True])
@pytest.mark.parametrize("return_eids", [False, True])
@pytest.mark.parametrize("replace", [False, True])
def test_rpc_hetero_sampling_shuffle(
    num_server, use_graphbolt, return_eids, replace
):
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_hetero_sampling_shuffle(
            Path(tmpdirname),
            num_server,
            use_graphbolt=use_graphbolt,
            return_eids=return_eids,
            replace=replace,
        )


@pytest.mark.parametrize("num_server", [1])
@pytest.mark.parametrize("use_graphbolt", [False, True])
@pytest.mark.parametrize("return_eids", [False, True])
def test_rpc_hetero_sampling_empty_shuffle(
    num_server, use_graphbolt, return_eids
):
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_hetero_sampling_empty_shuffle(
            Path(tmpdirname),
            num_server,
            use_graphbolt=use_graphbolt,
            return_eids=return_eids,
        )


@pytest.mark.parametrize("num_server", [1])
@pytest.mark.parametrize(
    "graph_formats", [None, ["csc"], ["csr"], ["csc", "coo"]]
)
def test_rpc_hetero_etype_sampling_shuffle_dgl(num_server, graph_formats):
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_hetero_etype_sampling_shuffle(
            Path(tmpdirname), num_server, graph_formats=graph_formats
        )


@pytest.mark.parametrize("num_server", [1])
@pytest.mark.parametrize("return_eids", [False, True])
def test_rpc_hetero_etype_sampling_shuffle_graphbolt(num_server, return_eids):
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_hetero_etype_sampling_shuffle(
            Path(tmpdirname),
            num_server,
            use_graphbolt=True,
            return_eids=return_eids,
        )


@pytest.mark.parametrize("num_server", [1])
@pytest.mark.parametrize("use_graphbolt", [False, True])
@pytest.mark.parametrize("return_eids", [False, True])
def test_rpc_hetero_etype_sampling_empty_shuffle(
    num_server, use_graphbolt, return_eids
):
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_hetero_etype_sampling_empty_shuffle(
            Path(tmpdirname),
            num_server,
            use_graphbolt=use_graphbolt,
            return_eids=return_eids,
        )


@pytest.mark.parametrize("num_server", [1])
@pytest.mark.parametrize("use_graphbolt", [False, True])
@pytest.mark.parametrize("return_eids", [False, True])
def test_rpc_bipartite_sampling_empty_shuffle(
    num_server, use_graphbolt, return_eids
):
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_bipartite_sampling_empty(
            Path(tmpdirname), num_server, use_graphbolt, return_eids
        )


@pytest.mark.parametrize("num_server", [1])
@pytest.mark.parametrize("use_graphbolt", [False, True])
@pytest.mark.parametrize("return_eids", [False, True])
def test_rpc_bipartite_sampling_shuffle(num_server, use_graphbolt, return_eids):
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_bipartite_sampling_shuffle(
            Path(tmpdirname), num_server, use_graphbolt, return_eids
        )


@pytest.mark.parametrize("num_server", [1])
@pytest.mark.parametrize("use_graphbolt", [False, True])
@pytest.mark.parametrize("return_eids", [False, True])
def test_rpc_bipartite_etype_sampling_empty_shuffle(
    num_server, use_graphbolt, return_eids
):
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_bipartite_etype_sampling_empty(
            Path(tmpdirname),
            num_server,
            use_graphbolt=use_graphbolt,
            return_eids=return_eids,
        )


@pytest.mark.parametrize("num_server", [1])
@pytest.mark.parametrize("use_graphbolt", [False, True])
@pytest.mark.parametrize("return_eids", [False, True])
def test_rpc_bipartite_etype_sampling_shuffle(
    num_server, use_graphbolt, return_eids
):
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_bipartite_etype_sampling_shuffle(
            Path(tmpdirname),
            num_server,
            use_graphbolt=use_graphbolt,
            return_eids=return_eids,
        )


def check_standalone_sampling(tmpdir):
    g = CitationGraphDataset("cora")[0]
    prob = np.maximum(np.random.randn(g.num_edges()), 0)
    mask = prob > 0
    g.edata["prob"] = F.tensor(prob)
    g.edata["mask"] = F.tensor(mask)
    num_parts = 1
    num_hops = 1
    partition_graph(
        g,
        "test_sampling",
        num_parts,
        tmpdir,
        num_hops=num_hops,
        part_method="metis",
    )

    os.environ["DGL_DIST_MODE"] = "standalone"
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph(
        "test_sampling", part_config=tmpdir / "test_sampling.json"
    )
    sampled_graph = sample_neighbors(
        dist_graph,
        torch.tensor([0, 10, 99, 66, 1024, 2008], dtype=dist_graph.idtype),
        3,
    )

    src, dst = sampled_graph.edges()
    assert sampled_graph.num_nodes() == g.num_nodes()
    assert np.all(F.asnumpy(g.has_edges_between(src, dst)))
    eids = g.edge_ids(src, dst)
    assert np.array_equal(
        F.asnumpy(sampled_graph.edata[dgl.EID]), F.asnumpy(eids)
    )

    sampled_graph = sample_neighbors(
        dist_graph,
        torch.tensor([0, 10, 99, 66, 1024, 2008], dtype=dist_graph.idtype),
        3,
        prob="mask",
    )
    eid = F.asnumpy(sampled_graph.edata[dgl.EID])
    assert mask[eid].all()

    sampled_graph = sample_neighbors(
        dist_graph,
        torch.tensor([0, 10, 99, 66, 1024, 2008], dtype=dist_graph.idtype),
        3,
        prob="prob",
    )
    eid = F.asnumpy(sampled_graph.edata[dgl.EID])
    assert (prob[eid] > 0).all()
    dgl.distributed.exit_client()


def check_standalone_etype_sampling(tmpdir):
    hg = CitationGraphDataset("cora")[0]
    prob = np.maximum(np.random.randn(hg.num_edges()), 0)
    mask = prob > 0
    hg.edata["prob"] = F.tensor(prob)
    hg.edata["mask"] = F.tensor(mask)
    num_parts = 1
    num_hops = 1

    partition_graph(
        hg,
        "test_sampling",
        num_parts,
        tmpdir,
        num_hops=num_hops,
        part_method="metis",
    )
    os.environ["DGL_DIST_MODE"] = "standalone"
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph(
        "test_sampling", part_config=tmpdir / "test_sampling.json"
    )
    sampled_graph = sample_etype_neighbors(
        dist_graph,
        torch.tensor([0, 10, 99, 66, 1023], dtype=dist_graph.idtype),
        3,
    )

    src, dst = sampled_graph.edges()
    assert sampled_graph.num_nodes() == hg.num_nodes()
    assert np.all(F.asnumpy(hg.has_edges_between(src, dst)))
    eids = hg.edge_ids(src, dst)
    assert np.array_equal(
        F.asnumpy(sampled_graph.edata[dgl.EID]), F.asnumpy(eids)
    )

    sampled_graph = sample_etype_neighbors(
        dist_graph,
        torch.tensor([0, 10, 99, 66, 1023], dtype=dist_graph.idtype),
        3,
        prob="mask",
    )
    eid = F.asnumpy(sampled_graph.edata[dgl.EID])
    assert mask[eid].all()

    sampled_graph = sample_etype_neighbors(
        dist_graph,
        torch.tensor([0, 10, 99, 66, 1023], dtype=dist_graph.idtype),
        3,
        prob="prob",
    )
    eid = F.asnumpy(sampled_graph.edata[dgl.EID])
    assert (prob[eid] > 0).all()
    dgl.distributed.exit_client()


def check_standalone_etype_sampling_heterograph(tmpdir):
    hg = CitationGraphDataset("cora")[0]
    num_parts = 1
    num_hops = 1
    src, dst = hg.edges()
    new_hg = dgl.heterograph(
        {
            ("paper", "cite", "paper"): (src, dst),
            ("paper", "cite-by", "paper"): (dst, src),
        },
        {"paper": hg.num_nodes()},
    )
    partition_graph(
        new_hg,
        "test_hetero_sampling",
        num_parts,
        tmpdir,
        num_hops=num_hops,
        part_method="metis",
    )
    os.environ["DGL_DIST_MODE"] = "standalone"
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph(
        "test_hetero_sampling", part_config=tmpdir / "test_hetero_sampling.json"
    )
    sampled_graph = sample_etype_neighbors(
        dist_graph,
        torch.tensor(
            [0, 1, 2, 10, 99, 66, 1023, 1024, 2700, 2701],
            dtype=dist_graph.idtype,
        ),
        1,
    )
    src, dst = sampled_graph.edges(etype=("paper", "cite", "paper"))
    assert len(src) == 10
    src, dst = sampled_graph.edges(etype=("paper", "cite-by", "paper"))
    assert len(src) == 10
    assert sampled_graph.num_nodes() == new_hg.num_nodes()
    dgl.distributed.exit_client()


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow",
    reason="Not support tensorflow for now",
)
def test_standalone_sampling():
    reset_envs()
    import tempfile

    os.environ["DGL_DIST_MODE"] = "standalone"
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_standalone_sampling(Path(tmpdirname))


def start_in_subgraph_client(rank, tmpdir, disable_shared_mem, nodes):
    gpb = None
    dgl.distributed.initialize("rpc_ip_config.txt")
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(
            tmpdir / "test_in_subgraph.json", rank
        )
    dist_graph = DistGraph("test_in_subgraph", gpb=gpb)
    try:
        sampled_graph = dgl.distributed.in_subgraph(dist_graph, nodes)
    except Exception as e:
        print(traceback.format_exc())
        sampled_graph = None
    dgl.distributed.exit_client()
    return sampled_graph


def check_rpc_in_subgraph_shuffle(tmpdir, num_server):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = CitationGraphDataset("cora")[0]
    num_parts = num_server

    orig_nid, orig_eid = partition_graph(
        g,
        "test_in_subgraph",
        num_parts,
        tmpdir,
        num_hops=1,
        part_method="metis",
        return_mapping=True,
    )

    pserver_list = []
    ctx = mp.get_context("spawn")
    for i in range(num_server):
        p = ctx.Process(
            target=start_server,
            args=(i, tmpdir, num_server > 1, "test_in_subgraph"),
        )
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    nodes = torch.tensor([0, 10, 99, 66, 1024, 2008], dtype=g.idtype)
    sampled_graph = start_in_subgraph_client(0, tmpdir, num_server > 1, nodes)
    for p in pserver_list:
        p.join()
        assert p.exitcode == 0

    src, dst = sampled_graph.edges()
    src = orig_nid[src]
    dst = orig_nid[dst]
    assert sampled_graph.num_nodes() == g.num_nodes()
    assert np.all(F.asnumpy(g.has_edges_between(src, dst)))

    subg1 = dgl.in_subgraph(g, orig_nid[nodes])
    src1, dst1 = subg1.edges()
    assert np.all(np.sort(F.asnumpy(src)) == np.sort(F.asnumpy(src1)))
    assert np.all(np.sort(F.asnumpy(dst)) == np.sort(F.asnumpy(dst1)))
    eids = g.edge_ids(src, dst)
    eids1 = orig_eid[sampled_graph.edata[dgl.EID]]
    assert np.array_equal(F.asnumpy(eids1), F.asnumpy(eids))


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow",
    reason="Not support tensorflow for now",
)
def test_rpc_in_subgraph():
    reset_envs()
    import tempfile

    os.environ["DGL_DIST_MODE"] = "distributed"
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_in_subgraph_shuffle(Path(tmpdirname), 1)


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow",
    reason="Not support tensorflow for now",
)
@unittest.skipIf(
    dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support"
)
def test_standalone_etype_sampling():
    reset_envs()
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.environ["DGL_DIST_MODE"] = "standalone"
        check_standalone_etype_sampling_heterograph(Path(tmpdirname))
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.environ["DGL_DIST_MODE"] = "standalone"
        check_standalone_etype_sampling(Path(tmpdirname))


@pytest.mark.parametrize("num_parts", [1, 4])
@pytest.mark.parametrize("use_graphbolt", [False])
@pytest.mark.parametrize("prob_or_mask", ["prob", "mask"])
def test_local_sampling_homograph(num_parts, use_graphbolt, prob_or_mask):
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    with tempfile.TemporaryDirectory() as test_dir:
        g = CitationGraphDataset("cora")[0]
        prob = torch.rand(g.num_edges())
        mask = prob > 0.2
        prob[torch.randperm(len(prob))[: int(len(prob) * 0.5)]] = 0.0
        g.edata["prob"] = prob
        g.edata["mask"] = mask
        graph_name = "test_local_sampling"

        _, orig_eids = partition_graph(
            g,
            graph_name,
            num_parts,
            test_dir,
            num_hops=1,
            part_method="metis",
            return_mapping=True,
            use_graphbolt=use_graphbolt,
            store_eids=True,
            store_inner_node=True,
            store_inner_edge=True,
        )

        part_config = os.path.join(test_dir, f"{graph_name}.json")
        for part_id in range(num_parts):
            local_g, _, edge_feats, gpb, _, _, _ = load_partition(
                part_config,
                part_id,
                load_feats=True,
                use_graphbolt=use_graphbolt,
            )
            inner_global_nids = gpb.partid2nids(part_id)
            inner_global_eids = gpb.partid2eids(part_id)
            inner_node_data = (
                local_g.node_attributes["inner_node"]
                if use_graphbolt
                else local_g.ndata["inner_node"]
            )
            inner_edge_data = (
                local_g.edge_attributes["inner_edge"]
                if use_graphbolt
                else local_g.edata["inner_edge"]
            )
            assert len(inner_global_nids) == inner_node_data.sum()
            assert len(inner_global_eids) == inner_edge_data.sum()

            c_etype = gpb.canonical_etypes[0]
            _prob = []
            prob = edge_feats[_etype_tuple_to_str(c_etype) + "/" + prob_or_mask]
            assert len(prob) == len(inner_global_eids)
            assert len(prob) <= inner_edge_data.shape[0]
            _prob.append(prob)

            sampled_g = dgl.distributed.graph_services._sample_neighbors(
                use_graphbolt,
                local_g,
                gpb,
                inner_global_nids,
                5,
                prob=_prob,
            )
            sampled_homo_eids = sampled_g.global_eids
            sampled_orig_eids = orig_eids[sampled_homo_eids]
            assert torch.all(g.edata[prob_or_mask][sampled_orig_eids] > 0)


@pytest.mark.parametrize("num_parts", [1, 4])
@pytest.mark.parametrize("use_graphbolt", [False])
@pytest.mark.parametrize("prob_or_mask", ["prob", "mask"])
def test_local_sampling_heterograph(num_parts, use_graphbolt, prob_or_mask):
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    with tempfile.TemporaryDirectory() as test_dir:
        g = create_random_hetero()
        for c_etype in g.canonical_etypes:
            prob = torch.rand(g.num_edges(c_etype))
            mask = prob > 0.2
            prob[torch.randperm(len(prob))[: int(len(prob) * 0.5)]] = 0.0
            g.edges[c_etype].data["prob"] = prob
            g.edges[c_etype].data["mask"] = mask
        graph_name = "test_local_sampling"

        _, orig_eids = partition_graph(
            g,
            graph_name,
            num_parts,
            test_dir,
            num_hops=1,
            part_method="metis",
            return_mapping=True,
            use_graphbolt=use_graphbolt,
            store_eids=True,
            store_inner_node=True,
            store_inner_edge=True,
        )

        part_config = os.path.join(test_dir, f"{graph_name}.json")
        for part_id in range(num_parts):
            local_g, _, edge_feats, gpb, _, _, _ = load_partition(
                part_config,
                part_id,
                load_feats=True,
                use_graphbolt=use_graphbolt,
            )
            inner_global_nids = [
                gpb.map_to_homo_nid(gpb.partid2nids(part_id, ntype), ntype)
                for ntype in gpb.ntypes
            ]
            inner_global_nids = torch.cat(inner_global_nids)
            inner_global_eids = {
                c_etype: gpb.partid2eids(part_id, c_etype)
                for c_etype in gpb.canonical_etypes
            }
            inner_node_data = (
                local_g.node_attributes["inner_node"]
                if use_graphbolt
                else local_g.ndata["inner_node"]
            )
            inner_edge_data = (
                local_g.edge_attributes["inner_edge"]
                if use_graphbolt
                else local_g.edata["inner_edge"]
            )
            assert len(inner_global_nids) == inner_node_data.sum()
            num_inner_global_eids = sum(
                [len(eids) for eids in inner_global_eids.values()]
            )
            assert num_inner_global_eids == inner_edge_data.sum()

            _prob = []
            for i, c_etype in enumerate(gpb.canonical_etypes):
                prob = edge_feats[
                    _etype_tuple_to_str(c_etype) + "/" + prob_or_mask
                ]
                assert len(prob) == len(inner_global_eids[c_etype])
                assert (
                    len(prob)
                    == gpb.local_etype_offset[i + 1] - gpb.local_etype_offset[i]
                )
                assert len(prob) <= inner_edge_data.shape[0]
                _prob.append(prob)

            sampled_g = dgl.distributed.graph_services._sample_etype_neighbors(
                use_graphbolt,
                local_g,
                gpb,
                inner_global_nids,
                torch.full((len(g.canonical_etypes),), 5, dtype=torch.int64),
                prob=_prob,
                etype_offset=gpb.local_etype_offset,
            )
            sampled_homo_eids = sampled_g.global_eids
            sampled_etype_ids, sampled_per_etype_eids = gpb.map_to_per_etype(
                sampled_homo_eids
            )
            for etype_id, c_etype in enumerate(gpb.canonical_etypes):
                indices = torch.nonzero(sampled_etype_ids == etype_id).squeeze()
                sampled_eids = sampled_per_etype_eids[indices]
                sampled_orig_eids = orig_eids[c_etype][sampled_eids]
                assert torch.all(
                    g.edges[c_etype].data[prob_or_mask][sampled_orig_eids] > 0
                )


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.environ["DGL_DIST_MODE"] = "standalone"
        check_standalone_etype_sampling_heterograph(Path(tmpdirname))

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.environ["DGL_DIST_MODE"] = "standalone"
        check_standalone_etype_sampling(Path(tmpdirname))
        check_standalone_sampling(Path(tmpdirname))
        os.environ["DGL_DIST_MODE"] = "distributed"
        check_rpc_sampling(Path(tmpdirname), 2)
        check_rpc_sampling(Path(tmpdirname), 1)
        check_rpc_get_degree_shuffle(Path(tmpdirname), 1)
        check_rpc_get_degree_shuffle(Path(tmpdirname), 2)
        check_rpc_find_edges_shuffle(Path(tmpdirname), 2)
        check_rpc_find_edges_shuffle(Path(tmpdirname), 1)
        check_rpc_hetero_find_edges_shuffle(Path(tmpdirname), 1)
        check_rpc_hetero_find_edges_shuffle(Path(tmpdirname), 2)
        check_rpc_in_subgraph_shuffle(Path(tmpdirname), 2)
        check_rpc_sampling_shuffle(Path(tmpdirname), 1)
        check_rpc_hetero_sampling_shuffle(Path(tmpdirname), 1)
        check_rpc_hetero_sampling_shuffle(Path(tmpdirname), 2)
        check_rpc_hetero_sampling_empty_shuffle(Path(tmpdirname), 1)
        check_rpc_hetero_etype_sampling_shuffle(Path(tmpdirname), 1)
        check_rpc_hetero_etype_sampling_shuffle(Path(tmpdirname), 2)
        check_rpc_hetero_etype_sampling_empty_shuffle(Path(tmpdirname), 1)
