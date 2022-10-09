import dgl
import unittest
import os
from dgl.data import CitationGraphDataset
from dgl.data import WN18Dataset
from dgl.distributed import sample_neighbors, sample_etype_neighbors
from dgl.distributed import partition_graph, load_partition, load_partition_book
import sys
import multiprocessing as mp
import numpy as np
import backend as F
import time
from utils import generate_ip_config, reset_envs
from pathlib import Path
import pytest
from scipy import sparse as spsp
import random
from dgl.distributed import DistGraphServer, DistGraph


def start_server(rank, tmpdir, disable_shared_mem, graph_name, graph_format=['csc', 'coo'],
                 keep_alive=False):
    g = DistGraphServer(rank, "rpc_ip_config.txt", 1, 1,
                        tmpdir / (graph_name + '.json'), disable_shared_mem=disable_shared_mem,
                        graph_format=graph_format, keep_alive=keep_alive)
    g.start()


def start_sample_client(rank, tmpdir, disable_shared_mem):
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(tmpdir / 'test_sampling.json', rank)
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_sampling", gpb=gpb)
    try:
        sampled_graph = sample_neighbors(dist_graph, [0, 10, 99, 66, 1024, 2008], 3)
    except Exception as e:
        print(e)
        sampled_graph = None
    dgl.distributed.exit_client()
    return sampled_graph


def start_sample_client_shuffle(rank, tmpdir, disable_shared_mem, g, num_servers, group_id,
        orig_nid, orig_eid):
    os.environ['DGL_GROUP_ID'] = str(group_id)
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(tmpdir / 'test_sampling.json', rank)
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_sampling", gpb=gpb)
    sampled_graph = sample_neighbors(dist_graph, [0, 10, 99, 66, 1024, 2008], 3)

    src, dst = sampled_graph.edges()
    src = orig_nid[src]
    dst = orig_nid[dst]
    assert sampled_graph.number_of_nodes() == g.number_of_nodes()
    assert np.all(F.asnumpy(g.has_edges_between(src, dst)))
    eids = g.edge_ids(src, dst)
    eids1 = orig_eid[sampled_graph.edata[dgl.EID]]
    assert np.array_equal(F.asnumpy(eids1), F.asnumpy(eids))

def start_find_edges_client(rank, tmpdir, disable_shared_mem, eids, etype=None):
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(tmpdir / 'test_find_edges.json', rank)
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_find_edges", gpb=gpb)
    try:
        u, v = dist_graph.find_edges(eids, etype=etype)
    except Exception as e:
        print(e)
        u, v = None, None
    dgl.distributed.exit_client()
    return u, v

def start_get_degrees_client(rank, tmpdir, disable_shared_mem, nids=None):
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(tmpdir / 'test_get_degrees.json', rank)
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_get_degrees", gpb=gpb)
    try:
        in_deg = dist_graph.in_degrees(nids)
        all_in_deg = dist_graph.in_degrees()
        out_deg = dist_graph.out_degrees(nids)
        all_out_deg = dist_graph.out_degrees()
    except Exception as e:
        print(e)
        in_deg, out_deg, all_in_deg, all_out_deg = None, None, None, None
    dgl.distributed.exit_client()
    return in_deg, out_deg, all_in_deg, all_out_deg

def check_rpc_sampling(tmpdir, num_server):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = CitationGraphDataset("cora")[0]
    g.readonly()
    print(g.idtype)
    num_parts = num_server
    num_hops = 1

    partition_graph(g, 'test_sampling', num_parts, tmpdir,
                    num_hops=num_hops, part_method='metis', reshuffle=False)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(i, tmpdir, num_server > 1, 'test_sampling'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    sampled_graph = start_sample_client(0, tmpdir, num_server > 1)
    print("Done sampling")
    for p in pserver_list:
        p.join()

    src, dst = sampled_graph.edges()
    assert sampled_graph.number_of_nodes() == g.number_of_nodes()
    assert np.all(F.asnumpy(g.has_edges_between(src, dst)))
    eids = g.edge_ids(src, dst)
    assert np.array_equal(
        F.asnumpy(sampled_graph.edata[dgl.EID]), F.asnumpy(eids))

def check_rpc_find_edges_shuffle(tmpdir, num_server):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = CitationGraphDataset("cora")[0]
    g.readonly()
    num_parts = num_server

    orig_nid, orig_eid = partition_graph(g, 'test_find_edges', num_parts, tmpdir,
                                         num_hops=1, part_method='metis',
                                         reshuffle=True, return_mapping=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(i, tmpdir, num_server > 1,
                                                   'test_find_edges', ['csr', 'coo']))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    eids = F.tensor(np.random.randint(g.number_of_edges(), size=100))
    u, v = g.find_edges(orig_eid[eids])
    du, dv = start_find_edges_client(0, tmpdir, num_server > 1, eids)
    du = orig_nid[du]
    dv = orig_nid[dv]
    assert F.array_equal(u, du)
    assert F.array_equal(v, dv)

def create_random_hetero(dense=False, empty=False):
    num_nodes = {'n1': 210, 'n2': 200, 'n3': 220} if dense else \
        {'n1': 1010, 'n2': 1000, 'n3': 1020}
    etypes = [('n1', 'r12', 'n2'),
              ('n1', 'r13', 'n3'),
              ('n2', 'r23', 'n3')]
    edges = {}
    random.seed(42)
    for etype in etypes:
        src_ntype, _, dst_ntype = etype
        arr = spsp.random(num_nodes[src_ntype] - 10 if empty else num_nodes[src_ntype],
                          num_nodes[dst_ntype] - 10 if empty else num_nodes[dst_ntype],
                          density=0.1 if dense else 0.001,
                          format='coo', random_state=100)
        edges[etype] = (arr.row, arr.col)
    g = dgl.heterograph(edges, num_nodes)
    g.nodes['n1'].data['feat'] = F.ones((g.number_of_nodes('n1'), 10), F.float32, F.cpu())
    return g

def check_rpc_hetero_find_edges_shuffle(tmpdir, num_server):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = create_random_hetero()
    num_parts = num_server

    orig_nid, orig_eid = partition_graph(g, 'test_find_edges', num_parts, tmpdir,
                                         num_hops=1, part_method='metis',
                                         reshuffle=True, return_mapping=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(i, tmpdir, num_server > 1,
                                                   'test_find_edges', ['csr', 'coo']))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    eids = F.tensor(np.random.randint(g.num_edges('r12'), size=100))
    expect_except = False
    try:
        _, _ = g.find_edges(orig_eid['r12'][eids], etype=('n1', 'r12'))
    except:
        expect_except = True
    assert expect_except
    u, v = g.find_edges(orig_eid['r12'][eids], etype='r12')
    u1, v1 = g.find_edges(orig_eid['r12'][eids], etype=('n1', 'r12', 'n2'))
    assert F.array_equal(u, u1)
    assert F.array_equal(v, v1)
    du, dv = start_find_edges_client(0, tmpdir, num_server > 1, eids, etype='r12')
    du = orig_nid['n1'][du]
    dv = orig_nid['n2'][dv]
    assert F.array_equal(u, du)
    assert F.array_equal(v, dv)

# Wait non shared memory graph store
@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name == 'tensorflow', reason='Not support tensorflow for now')
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support")
@pytest.mark.parametrize("num_server", [1, 2])
def test_rpc_find_edges_shuffle(num_server):
    reset_envs()
    import tempfile
    os.environ['DGL_DIST_MODE'] = 'distributed'
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_hetero_find_edges_shuffle(Path(tmpdirname), num_server)
        check_rpc_find_edges_shuffle(Path(tmpdirname), num_server)

def check_rpc_get_degree_shuffle(tmpdir, num_server):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = CitationGraphDataset("cora")[0]
    g.readonly()
    num_parts = num_server

    orig_nid, _ = partition_graph(g, 'test_get_degrees', num_parts, tmpdir,
        num_hops=1, part_method='metis', reshuffle=True, return_mapping=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(i, tmpdir, num_server > 1, 'test_get_degrees'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    nids = F.tensor(np.random.randint(g.number_of_nodes(), size=100))
    in_degs, out_degs, all_in_degs, all_out_degs = start_get_degrees_client(0, tmpdir, num_server > 1, nids)

    print("Done get_degree")
    for p in pserver_list:
        p.join()

    print('check results')
    assert F.array_equal(g.in_degrees(orig_nid[nids]), in_degs)
    assert F.array_equal(g.in_degrees(orig_nid), all_in_degs)
    assert F.array_equal(g.out_degrees(orig_nid[nids]), out_degs)
    assert F.array_equal(g.out_degrees(orig_nid), all_out_degs)

# Wait non shared memory graph store
@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name == 'tensorflow', reason='Not support tensorflow for now')
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support")
@pytest.mark.parametrize("num_server", [1, 2])
def test_rpc_get_degree_shuffle(num_server):
    reset_envs()
    import tempfile
    os.environ['DGL_DIST_MODE'] = 'distributed'
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_get_degree_shuffle(Path(tmpdirname), num_server)

#@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
#@unittest.skipIf(dgl.backend.backend_name == 'tensorflow', reason='Not support tensorflow for now')
@unittest.skip('Only support partition with shuffle')
def test_rpc_sampling():
    reset_envs()
    import tempfile
    os.environ['DGL_DIST_MODE'] = 'distributed'
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_sampling(Path(tmpdirname), 2)

def check_rpc_sampling_shuffle(tmpdir, num_server, num_groups=1):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = CitationGraphDataset("cora")[0]
    g.readonly()
    num_parts = num_server
    num_hops = 1

    orig_nids, orig_eids = partition_graph(g, 'test_sampling', num_parts, tmpdir,
        num_hops=num_hops, part_method='metis', reshuffle=True, return_mapping=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    keep_alive = num_groups > 1
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(
            i, tmpdir, num_server > 1, 'test_sampling', ['csc', 'coo'], keep_alive))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    pclient_list = []
    num_clients = 1
    for client_id in range(num_clients):
        for group_id in range(num_groups):
            p = ctx.Process(target=start_sample_client_shuffle,
                args=(client_id, tmpdir, num_server > 1, g, num_server,
                    group_id, orig_nids, orig_eids))
            p.start()
            time.sleep(1) # avoid race condition when instantiating DistGraph
            pclient_list.append(p)
    for p in pclient_list:
        p.join()
    if keep_alive:
        for p in pserver_list:
            assert p.is_alive()
        # force shutdown server
        dgl.distributed.shutdown_servers("rpc_ip_config.txt", 1)
    for p in pserver_list:
        p.join()

def start_hetero_sample_client(rank, tmpdir, disable_shared_mem, nodes):
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(tmpdir / 'test_sampling.json', rank)
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_sampling", gpb=gpb)
    assert 'feat' in dist_graph.nodes['n1'].data
    assert 'feat' not in dist_graph.nodes['n2'].data
    assert 'feat' not in dist_graph.nodes['n3'].data
    if gpb is None:
        gpb = dist_graph.get_partition_book()
    try:
        sampled_graph = sample_neighbors(dist_graph, nodes, 3)
        block = dgl.to_block(sampled_graph, nodes)
        block.edata[dgl.EID] = sampled_graph.edata[dgl.EID]
    except Exception as e:
        print(e)
        block = None
    dgl.distributed.exit_client()
    return block, gpb

def start_hetero_etype_sample_client(rank, tmpdir, disable_shared_mem, fanout=3,
                                     nodes={'n3': [0, 10, 99, 66, 124, 208]},
                                     etype_sorted=False):
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(tmpdir / 'test_sampling.json', rank)
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_sampling", gpb=gpb)
    assert 'feat' in dist_graph.nodes['n1'].data
    assert 'feat' not in dist_graph.nodes['n2'].data
    assert 'feat' not in dist_graph.nodes['n3'].data

    if dist_graph.local_partition is not None:
        # Check whether etypes are sorted in dist_graph
        local_g = dist_graph.local_partition
        local_nids = np.arange(local_g.num_nodes())
        for lnid in local_nids:
            leids = local_g.in_edges(lnid, form='eid')
            letids = F.asnumpy(local_g.edata[dgl.ETYPE][leids])
            _, idices = np.unique(letids, return_index=True)
            assert np.all(idices[:-1] <= idices[1:])

    if gpb is None:
        gpb = dist_graph.get_partition_book()
    try:
        sampled_graph = sample_etype_neighbors(dist_graph, nodes, dgl.ETYPE, fanout, etype_sorted=etype_sorted)
        block = dgl.to_block(sampled_graph, nodes)
        block.edata[dgl.EID] = sampled_graph.edata[dgl.EID]
    except Exception as e:
        print(e)
        block = None
    dgl.distributed.exit_client()
    return block, gpb

def check_rpc_hetero_sampling_shuffle(tmpdir, num_server):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = create_random_hetero()
    num_parts = num_server
    num_hops = 1

    orig_nid_map, orig_eid_map = partition_graph(g, 'test_sampling', num_parts, tmpdir,
        num_hops=num_hops, part_method='metis', reshuffle=True, return_mapping=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(i, tmpdir, num_server > 1, 'test_sampling'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    block, gpb = start_hetero_sample_client(0, tmpdir, num_server > 1,
                                            nodes = {'n3': [0, 10, 99, 66, 124, 208]})
    print("Done sampling")
    for p in pserver_list:
        p.join()

    for src_type, etype, dst_type in block.canonical_etypes:
        src, dst = block.edges(etype=etype)
        # These are global Ids after shuffling.
        shuffled_src = F.gather_row(block.srcnodes[src_type].data[dgl.NID], src)
        shuffled_dst = F.gather_row(block.dstnodes[dst_type].data[dgl.NID], dst)
        shuffled_eid = block.edges[etype].data[dgl.EID]

        orig_src = F.asnumpy(F.gather_row(orig_nid_map[src_type], shuffled_src))
        orig_dst = F.asnumpy(F.gather_row(orig_nid_map[dst_type], shuffled_dst))
        orig_eid = F.asnumpy(F.gather_row(orig_eid_map[etype], shuffled_eid))

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

def check_rpc_hetero_sampling_empty_shuffle(tmpdir, num_server):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = create_random_hetero(empty=True)
    num_parts = num_server
    num_hops = 1

    orig_nids, _ = partition_graph(g, 'test_sampling', num_parts, tmpdir,
                                   num_hops=num_hops, part_method='metis',
                                   reshuffle=True, return_mapping=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(i, tmpdir, num_server > 1, 'test_sampling'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    deg = get_degrees(g, orig_nids['n3'], 'n3')
    empty_nids = F.nonzero_1d(deg == 0)
    block, gpb = start_hetero_sample_client(0, tmpdir, num_server > 1,
                                            nodes = {'n3': empty_nids})
    print("Done sampling")
    for p in pserver_list:
        p.join()

    assert block.number_of_edges() == 0
    assert len(block.etypes) == len(g.etypes)

def check_rpc_hetero_etype_sampling_shuffle(tmpdir, num_server, etype_sorted=False):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = create_random_hetero(dense=True)
    num_parts = num_server
    num_hops = 1

    orig_nid_map, orig_eid_map = partition_graph(g, 'test_sampling', num_parts, tmpdir,
        num_hops=num_hops, part_method='metis', reshuffle=True, return_mapping=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(i, tmpdir, num_server > 1, 'test_sampling', ['csc', 'coo']))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    fanout = 3
    block, gpb = start_hetero_etype_sample_client(0, tmpdir, num_server > 1, fanout,
                                                  nodes={'n3': [0, 10, 99, 66, 124, 208]},
                                                  etype_sorted=etype_sorted)
    print("Done sampling")
    for p in pserver_list:
        p.join()

    src, dst = block.edges(etype=('n1', 'r13', 'n3'))
    assert len(src) == 18
    src, dst = block.edges(etype=('n2', 'r23', 'n3'))
    assert len(src) == 18

    for src_type, etype, dst_type in block.canonical_etypes:
        src, dst = block.edges(etype=etype)
        # These are global Ids after shuffling.
        shuffled_src = F.gather_row(block.srcnodes[src_type].data[dgl.NID], src)
        shuffled_dst = F.gather_row(block.dstnodes[dst_type].data[dgl.NID], dst)
        shuffled_eid = block.edges[etype].data[dgl.EID]

        orig_src = F.asnumpy(F.gather_row(orig_nid_map[src_type], shuffled_src))
        orig_dst = F.asnumpy(F.gather_row(orig_nid_map[dst_type], shuffled_dst))
        orig_eid = F.asnumpy(F.gather_row(orig_eid_map[etype], shuffled_eid))

        # Check the node Ids and edge Ids.
        orig_src1, orig_dst1 = g.find_edges(orig_eid, etype=etype)
        assert np.all(F.asnumpy(orig_src1) == orig_src)
        assert np.all(F.asnumpy(orig_dst1) == orig_dst)

def check_rpc_hetero_etype_sampling_empty_shuffle(tmpdir, num_server):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = create_random_hetero(dense=True, empty=True)
    num_parts = num_server
    num_hops = 1

    orig_nids, _ = partition_graph(g, 'test_sampling', num_parts, tmpdir,
                                   num_hops=num_hops, part_method='metis',
                                   reshuffle=True, return_mapping=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(i, tmpdir, num_server > 1, 'test_sampling'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    fanout = 3
    deg = get_degrees(g, orig_nids['n3'], 'n3')
    empty_nids = F.nonzero_1d(deg == 0)
    block, gpb = start_hetero_etype_sample_client(0, tmpdir, num_server > 1, fanout,
                                                  nodes={'n3': empty_nids})
    print("Done sampling")
    for p in pserver_list:
        p.join()

    assert block.number_of_edges() == 0
    assert len(block.etypes) == len(g.etypes)


def create_random_bipartite():
    g = dgl.rand_bipartite('user', 'buys', 'game', 500, 1000, 1000)
    g.nodes['user'].data['feat'] = F.ones(
        (g.num_nodes('user'), 10), F.float32, F.cpu())
    g.nodes['game'].data['feat'] = F.ones(
        (g.num_nodes('game'), 10), F.float32, F.cpu())
    return g


def start_bipartite_sample_client(rank, tmpdir, disable_shared_mem, nodes):
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(
            tmpdir / 'test_sampling.json', rank)
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_sampling", gpb=gpb)
    assert 'feat' in dist_graph.nodes['user'].data
    assert 'feat' in dist_graph.nodes['game'].data
    if gpb is None:
        gpb = dist_graph.get_partition_book()
    sampled_graph = sample_neighbors(dist_graph, nodes, 3)
    block = dgl.to_block(sampled_graph, nodes)
    if sampled_graph.num_edges() > 0:
        block.edata[dgl.EID] = sampled_graph.edata[dgl.EID]
    dgl.distributed.exit_client()
    return block, gpb


def start_bipartite_etype_sample_client(rank, tmpdir, disable_shared_mem, fanout=3,
                                        nodes={}):
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(
            tmpdir / 'test_sampling.json', rank)
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_sampling", gpb=gpb)
    assert 'feat' in dist_graph.nodes['user'].data
    assert 'feat' in dist_graph.nodes['game'].data

    if dist_graph.local_partition is not None:
        # Check whether etypes are sorted in dist_graph
        local_g = dist_graph.local_partition
        local_nids = np.arange(local_g.num_nodes())
        for lnid in local_nids:
            leids = local_g.in_edges(lnid, form='eid')
            letids = F.asnumpy(local_g.edata[dgl.ETYPE][leids])
            _, idices = np.unique(letids, return_index=True)
            assert np.all(idices[:-1] <= idices[1:])

    if gpb is None:
        gpb = dist_graph.get_partition_book()
    sampled_graph = sample_etype_neighbors(
        dist_graph, nodes, dgl.ETYPE, fanout)
    block = dgl.to_block(sampled_graph, nodes)
    if sampled_graph.num_edges() > 0:
        block.edata[dgl.EID] = sampled_graph.edata[dgl.EID]
    dgl.distributed.exit_client()
    return block, gpb


def check_rpc_bipartite_sampling_empty(tmpdir, num_server):
    """sample on bipartite via sample_neighbors() which yields empty sample results"""
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = create_random_bipartite()
    num_parts = num_server
    num_hops = 1

    orig_nids, _ = partition_graph(g, 'test_sampling', num_parts, tmpdir,
                                   num_hops=num_hops, part_method='metis', reshuffle=True, return_mapping=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(
            i, tmpdir, num_server > 1, 'test_sampling'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    deg = get_degrees(g, orig_nids['game'], 'game')
    empty_nids = F.nonzero_1d(deg == 0)
    block, _ = start_bipartite_sample_client(0, tmpdir, num_server > 1,
                                             nodes={'game': empty_nids, 'user': [1]})

    print("Done sampling")
    for p in pserver_list:
        p.join()

    assert block.number_of_edges() == 0
    assert len(block.etypes) == len(g.etypes)


def check_rpc_bipartite_sampling_shuffle(tmpdir, num_server):
    """sample on bipartite via sample_neighbors() which yields non-empty sample results"""
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = create_random_bipartite()
    num_parts = num_server
    num_hops = 1

    orig_nid_map, orig_eid_map = partition_graph(g, 'test_sampling', num_parts, tmpdir,
        num_hops=num_hops, part_method='metis', reshuffle=True, return_mapping=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(
            i, tmpdir, num_server > 1, 'test_sampling'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    deg = get_degrees(g, orig_nid_map['game'], 'game')
    nids = F.nonzero_1d(deg > 0)
    block, gpb = start_bipartite_sample_client(0, tmpdir, num_server > 1,
                                               nodes={'game': nids, 'user': [0]})
    print("Done sampling")
    for p in pserver_list:
        p.join()

    for src_type, etype, dst_type in block.canonical_etypes:
        src, dst = block.edges(etype=etype)
        # These are global Ids after shuffling.
        shuffled_src = F.gather_row(
            block.srcnodes[src_type].data[dgl.NID], src)
        shuffled_dst = F.gather_row(
            block.dstnodes[dst_type].data[dgl.NID], dst)
        shuffled_eid = block.edges[etype].data[dgl.EID]

        orig_src = F.asnumpy(F.gather_row(
            orig_nid_map[src_type], shuffled_src))
        orig_dst = F.asnumpy(F.gather_row(
            orig_nid_map[dst_type], shuffled_dst))
        orig_eid = F.asnumpy(F.gather_row(orig_eid_map[etype], shuffled_eid))

        # Check the node Ids and edge Ids.
        orig_src1, orig_dst1 = g.find_edges(orig_eid, etype=etype)
        assert np.all(F.asnumpy(orig_src1) == orig_src)
        assert np.all(F.asnumpy(orig_dst1) == orig_dst)


def check_rpc_bipartite_etype_sampling_empty(tmpdir, num_server):
    """sample on bipartite via sample_etype_neighbors() which yields empty sample results"""
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = create_random_bipartite()
    num_parts = num_server
    num_hops = 1

    orig_nids, _ = partition_graph(g, 'test_sampling', num_parts, tmpdir,
                                   num_hops=num_hops, part_method='metis', reshuffle=True, return_mapping=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(
            i, tmpdir, num_server > 1, 'test_sampling'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    deg = get_degrees(g, orig_nids['game'], 'game')
    empty_nids = F.nonzero_1d(deg == 0)
    block, gpb = start_bipartite_etype_sample_client(0, tmpdir, num_server > 1,
                                                     nodes={'game': empty_nids, 'user': [1]})

    print("Done sampling")
    for p in pserver_list:
        p.join()

    assert block is not None
    assert block.number_of_edges() == 0
    assert len(block.etypes) == len(g.etypes)


def check_rpc_bipartite_etype_sampling_shuffle(tmpdir, num_server):
    """sample on bipartite via sample_etype_neighbors() which yields non-empty sample results"""
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = create_random_bipartite()
    num_parts = num_server
    num_hops = 1

    orig_nid_map, orig_eid_map = partition_graph(g, 'test_sampling', num_parts, tmpdir,
        num_hops=num_hops, part_method='metis', reshuffle=True, return_mapping=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(
            i, tmpdir, num_server > 1, 'test_sampling'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    fanout = 3
    deg = get_degrees(g, orig_nid_map['game'], 'game')
    nids = F.nonzero_1d(deg > 0)
    block, gpb = start_bipartite_etype_sample_client(0, tmpdir, num_server > 1, fanout,
                                                     nodes={'game': nids, 'user': [0]})
    print("Done sampling")
    for p in pserver_list:
        p.join()

    for src_type, etype, dst_type in block.canonical_etypes:
        src, dst = block.edges(etype=etype)
        # These are global Ids after shuffling.
        shuffled_src = F.gather_row(
            block.srcnodes[src_type].data[dgl.NID], src)
        shuffled_dst = F.gather_row(
            block.dstnodes[dst_type].data[dgl.NID], dst)
        shuffled_eid = block.edges[etype].data[dgl.EID]

        orig_src = F.asnumpy(F.gather_row(
            orig_nid_map[src_type], shuffled_src))
        orig_dst = F.asnumpy(F.gather_row(
            orig_nid_map[dst_type], shuffled_dst))
        orig_eid = F.asnumpy(F.gather_row(orig_eid_map[etype], shuffled_eid))

        # Check the node Ids and edge Ids.
        orig_src1, orig_dst1 = g.find_edges(orig_eid, etype=etype)
        assert np.all(F.asnumpy(orig_src1) == orig_src)
        assert np.all(F.asnumpy(orig_dst1) == orig_dst)

# Wait non shared memory graph store
@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name == 'tensorflow', reason='Not support tensorflow for now')
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support")
@pytest.mark.parametrize("num_server", [1, 2])
def test_rpc_sampling_shuffle(num_server):
    reset_envs()
    import tempfile
    os.environ['DGL_DIST_MODE'] = 'distributed'
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_sampling_shuffle(Path(tmpdirname), num_server)
        # [TODO][Rhett] Tests for multiple groups may fail sometimes and
        # root cause is unknown. Let's disable them for now.
        #check_rpc_sampling_shuffle(Path(tmpdirname), num_server, num_groups=2)
        check_rpc_hetero_sampling_shuffle(Path(tmpdirname), num_server)
        check_rpc_hetero_sampling_empty_shuffle(Path(tmpdirname), num_server)
        check_rpc_hetero_etype_sampling_shuffle(Path(tmpdirname), num_server)
        check_rpc_hetero_etype_sampling_shuffle(Path(tmpdirname), num_server, etype_sorted=True)
        check_rpc_hetero_etype_sampling_empty_shuffle(Path(tmpdirname), num_server)
        check_rpc_bipartite_sampling_empty(Path(tmpdirname), num_server)
        check_rpc_bipartite_sampling_shuffle(Path(tmpdirname), num_server)
        check_rpc_bipartite_etype_sampling_empty(Path(tmpdirname), num_server)
        check_rpc_bipartite_etype_sampling_shuffle(Path(tmpdirname), num_server)

def check_standalone_sampling(tmpdir, reshuffle):
    g = CitationGraphDataset("cora")[0]
    num_parts = 1
    num_hops = 1
    partition_graph(g, 'test_sampling', num_parts, tmpdir,
                    num_hops=num_hops, part_method='metis', reshuffle=reshuffle)

    os.environ['DGL_DIST_MODE'] = 'standalone'
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_sampling", part_config=tmpdir / 'test_sampling.json')
    sampled_graph = sample_neighbors(dist_graph, [0, 10, 99, 66, 1024, 2008], 3)

    src, dst = sampled_graph.edges()
    assert sampled_graph.number_of_nodes() == g.number_of_nodes()
    assert np.all(F.asnumpy(g.has_edges_between(src, dst)))
    eids = g.edge_ids(src, dst)
    assert np.array_equal(
        F.asnumpy(sampled_graph.edata[dgl.EID]), F.asnumpy(eids))
    dgl.distributed.exit_client()

def check_standalone_etype_sampling(tmpdir, reshuffle):
    hg = CitationGraphDataset('cora')[0]
    num_parts = 1
    num_hops = 1

    partition_graph(hg, 'test_sampling', num_parts, tmpdir,
                    num_hops=num_hops, part_method='metis', reshuffle=reshuffle)
    os.environ['DGL_DIST_MODE'] = 'standalone'
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_sampling", part_config=tmpdir / 'test_sampling.json')
    sampled_graph = sample_etype_neighbors(dist_graph, [0, 10, 99, 66, 1023], dgl.ETYPE, 3)

    src, dst = sampled_graph.edges()
    assert sampled_graph.number_of_nodes() == hg.number_of_nodes()
    assert np.all(F.asnumpy(hg.has_edges_between(src, dst)))
    eids = hg.edge_ids(src, dst)
    assert np.array_equal(
        F.asnumpy(sampled_graph.edata[dgl.EID]), F.asnumpy(eids))
    dgl.distributed.exit_client()

def check_standalone_etype_sampling_heterograph(tmpdir, reshuffle):
    hg = CitationGraphDataset('cora')[0]
    num_parts = 1
    num_hops = 1
    src, dst = hg.edges()
    new_hg = dgl.heterograph({('paper', 'cite', 'paper'): (src, dst),
                              ('paper', 'cite-by', 'paper'): (dst, src)},
                              {'paper': hg.number_of_nodes()})
    partition_graph(new_hg, 'test_hetero_sampling', num_parts, tmpdir,
                    num_hops=num_hops, part_method='metis', reshuffle=reshuffle)
    os.environ['DGL_DIST_MODE'] = 'standalone'
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_hetero_sampling", part_config=tmpdir / 'test_hetero_sampling.json')
    sampled_graph = sample_etype_neighbors(dist_graph, [0, 1, 2, 10, 99, 66, 1023, 1024, 2700, 2701], dgl.ETYPE, 1)
    src, dst = sampled_graph.edges(etype=('paper', 'cite', 'paper'))
    assert len(src) == 10
    src, dst = sampled_graph.edges(etype=('paper', 'cite-by', 'paper'))
    assert len(src) == 10
    assert sampled_graph.number_of_nodes() == new_hg.number_of_nodes()
    dgl.distributed.exit_client()

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name == 'tensorflow', reason='Not support tensorflow for now')
def test_standalone_sampling():
    reset_envs()
    import tempfile
    os.environ['DGL_DIST_MODE'] = 'standalone'
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_standalone_sampling(Path(tmpdirname), False)
        check_standalone_sampling(Path(tmpdirname), True)

def start_in_subgraph_client(rank, tmpdir, disable_shared_mem, nodes):
    gpb = None
    dgl.distributed.initialize("rpc_ip_config.txt")
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(tmpdir / 'test_in_subgraph.json', rank)
    dist_graph = DistGraph("test_in_subgraph", gpb=gpb)
    try:
        sampled_graph = dgl.distributed.in_subgraph(dist_graph, nodes)
    except Exception as e:
        print(e)
        sampled_graph = None
    dgl.distributed.exit_client()
    return sampled_graph


def check_rpc_in_subgraph_shuffle(tmpdir, num_server):
    generate_ip_config("rpc_ip_config.txt", num_server, num_server)

    g = CitationGraphDataset("cora")[0]
    g.readonly()
    num_parts = num_server

    orig_nid, orig_eid = partition_graph(g, 'test_in_subgraph', num_parts, tmpdir,
        num_hops=1, part_method='metis', reshuffle=True, return_mapping=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(i, tmpdir, num_server > 1, 'test_in_subgraph'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    nodes = [0, 10, 99, 66, 1024, 2008]
    sampled_graph = start_in_subgraph_client(0, tmpdir, num_server > 1, nodes)
    for p in pserver_list:
        p.join()

    src, dst = sampled_graph.edges()
    src = orig_nid[src]
    dst = orig_nid[dst]
    assert sampled_graph.number_of_nodes() == g.number_of_nodes()
    assert np.all(F.asnumpy(g.has_edges_between(src, dst)))

    subg1 = dgl.in_subgraph(g, orig_nid[nodes])
    src1, dst1 = subg1.edges()
    assert np.all(np.sort(F.asnumpy(src)) == np.sort(F.asnumpy(src1)))
    assert np.all(np.sort(F.asnumpy(dst)) == np.sort(F.asnumpy(dst1)))
    eids = g.edge_ids(src, dst)
    eids1 = orig_eid[sampled_graph.edata[dgl.EID]]
    assert np.array_equal(F.asnumpy(eids1), F.asnumpy(eids))

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name == 'tensorflow', reason='Not support tensorflow for now')
def test_rpc_in_subgraph():
    reset_envs()
    import tempfile
    os.environ['DGL_DIST_MODE'] = 'distributed'
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_in_subgraph_shuffle(Path(tmpdirname), 2)

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name == 'tensorflow', reason='Not support tensorflow for now')
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support")
def test_standalone_etype_sampling():
    reset_envs()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.environ['DGL_DIST_MODE'] = 'standalone'
        check_standalone_etype_sampling_heterograph(Path(tmpdirname), True)
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.environ['DGL_DIST_MODE'] = 'standalone'
        check_standalone_etype_sampling(Path(tmpdirname), True)
        check_standalone_etype_sampling(Path(tmpdirname), False)

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.environ['DGL_DIST_MODE'] = 'standalone'
        check_standalone_etype_sampling_heterograph(Path(tmpdirname), True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.environ['DGL_DIST_MODE'] = 'standalone'
        check_standalone_etype_sampling(Path(tmpdirname), True)
        check_standalone_etype_sampling(Path(tmpdirname), False)
        check_standalone_sampling(Path(tmpdirname), True)
        check_standalone_sampling(Path(tmpdirname), False)
        os.environ['DGL_DIST_MODE'] = 'distributed'
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
