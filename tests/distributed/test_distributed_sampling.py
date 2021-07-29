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
from utils import get_local_usable_addr
from pathlib import Path
import pytest
from scipy import sparse as spsp
import random
from dgl.distributed import DistGraphServer, DistGraph


def start_server(rank, tmpdir, disable_shared_mem, graph_name, graph_format=['csc', 'coo']):
    g = DistGraphServer(rank, "rpc_ip_config.txt", 1, 1,
                        tmpdir / (graph_name + '.json'), disable_shared_mem=disable_shared_mem,
                        graph_format=graph_format)
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
    dgl.distributed.initialize("rpc_ip_config.txt", 1)
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
    ip_config = open("rpc_ip_config.txt", "w")
    for _ in range(num_server):
        ip_config.write('{}\n'.format(get_local_usable_addr()))
    ip_config.close()

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

    time.sleep(3)
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
    ip_config = open("rpc_ip_config.txt", "w")
    for _ in range(num_server):
        ip_config.write('{}\n'.format(get_local_usable_addr()))
    ip_config.close()

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

    time.sleep(3)
    eids = F.tensor(np.random.randint(g.number_of_edges(), size=100))
    u, v = g.find_edges(orig_eid[eids])
    du, dv = start_find_edges_client(0, tmpdir, num_server > 1, eids)
    du = orig_nid[du]
    dv = orig_nid[dv]
    assert F.array_equal(u, du)
    assert F.array_equal(v, dv)

def create_random_hetero():
    num_nodes = {'n1': 10000, 'n2': 10010, 'n3': 10020}
    etypes = [('n1', 'r1', 'n2'),
              ('n1', 'r2', 'n3'),
              ('n2', 'r3', 'n3')]
    edges = {}
    for etype in etypes:
        src_ntype, _, dst_ntype = etype
        arr = spsp.random(num_nodes[src_ntype], num_nodes[dst_ntype], density=0.001, format='coo',
                          random_state=100)
        edges[etype] = (arr.row, arr.col)
    return dgl.heterograph(edges, num_nodes)

def check_rpc_hetero_find_edges_shuffle(tmpdir, num_server):
    ip_config = open("rpc_ip_config.txt", "w")
    for _ in range(num_server):
        ip_config.write('{}\n'.format(get_local_usable_addr()))
    ip_config.close()

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

    time.sleep(3)
    eids = F.tensor(np.random.randint(g.number_of_edges('r1'), size=100))
    u, v = g.find_edges(orig_eid['r1'][eids], etype='r1')
    du, dv = start_find_edges_client(0, tmpdir, num_server > 1, eids, etype='r1')
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
    import tempfile
    os.environ['DGL_DIST_MODE'] = 'distributed'
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_hetero_find_edges_shuffle(Path(tmpdirname), num_server)
        check_rpc_find_edges_shuffle(Path(tmpdirname), num_server)

def check_rpc_get_degree_shuffle(tmpdir, num_server):
    ip_config = open("rpc_ip_config.txt", "w")
    for _ in range(num_server):
        ip_config.write('{}\n'.format(get_local_usable_addr()))
    ip_config.close()

    g = CitationGraphDataset("cora")[0]
    g.readonly()
    num_parts = num_server

    partition_graph(g, 'test_get_degrees', num_parts, tmpdir,
                    num_hops=1, part_method='metis', reshuffle=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(i, tmpdir, num_server > 1, 'test_get_degrees'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    orig_nid = F.zeros((g.number_of_nodes(),), dtype=F.int64, ctx=F.cpu())
    for i in range(num_server):
        part, _, _, _, _, _, _ = load_partition(tmpdir / 'test_get_degrees.json', i)
        orig_nid[part.ndata[dgl.NID]] = part.ndata['orig_id']
    time.sleep(3)

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
    import tempfile
    os.environ['DGL_DIST_MODE'] = 'distributed'
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_get_degree_shuffle(Path(tmpdirname), num_server)

#@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
#@unittest.skipIf(dgl.backend.backend_name == 'tensorflow', reason='Not support tensorflow for now')
@unittest.skip('Only support partition with shuffle')
def test_rpc_sampling():
    import tempfile
    os.environ['DGL_DIST_MODE'] = 'distributed'
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_sampling(Path(tmpdirname), 2)

def check_rpc_sampling_shuffle(tmpdir, num_server):
    ip_config = open("rpc_ip_config.txt", "w")
    for _ in range(num_server):
        ip_config.write('{}\n'.format(get_local_usable_addr()))
    ip_config.close()

    g = CitationGraphDataset("cora")[0]
    g.readonly()
    num_parts = num_server
    num_hops = 1

    partition_graph(g, 'test_sampling', num_parts, tmpdir,
                    num_hops=num_hops, part_method='metis', reshuffle=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(i, tmpdir, num_server > 1, 'test_sampling'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    time.sleep(3)
    sampled_graph = start_sample_client(0, tmpdir, num_server > 1)
    print("Done sampling")
    for p in pserver_list:
        p.join()

    orig_nid = F.zeros((g.number_of_nodes(),), dtype=F.int64, ctx=F.cpu())
    orig_eid = F.zeros((g.number_of_edges(),), dtype=F.int64, ctx=F.cpu())
    for i in range(num_server):
        part, _, _, _, _, _, _ = load_partition(tmpdir / 'test_sampling.json', i)
        orig_nid[part.ndata[dgl.NID]] = part.ndata['orig_id']
        orig_eid[part.edata[dgl.EID]] = part.edata['orig_id']

    src, dst = sampled_graph.edges()
    src = orig_nid[src]
    dst = orig_nid[dst]
    assert sampled_graph.number_of_nodes() == g.number_of_nodes()
    assert np.all(F.asnumpy(g.has_edges_between(src, dst)))
    eids = g.edge_ids(src, dst)
    eids1 = orig_eid[sampled_graph.edata[dgl.EID]]
    assert np.array_equal(F.asnumpy(eids1), F.asnumpy(eids))

def create_random_hetero(dense=False):
    num_nodes = {'n1': 210, 'n2': 200, 'n3': 220} if dense else \
        {'n1': 1010, 'n2': 1000, 'n3': 1020}
    etypes = [('n1', 'r1', 'n2'),
              ('n1', 'r2', 'n3'),
              ('n2', 'r3', 'n3')]
    edges = {}
    random.seed(42)
    for etype in etypes:
        src_ntype, _, dst_ntype = etype
        arr = spsp.random(num_nodes[src_ntype], num_nodes[dst_ntype], density=0.1 if dense else 0.001, format='coo',
                          random_state=100)
        edges[etype] = (arr.row, arr.col)
    g = dgl.heterograph(edges, num_nodes)
    g.nodes['n1'].data['feat'] = F.ones((g.number_of_nodes('n1'), 10), F.float32, F.cpu())
    return g

def start_hetero_sample_client(rank, tmpdir, disable_shared_mem):
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
        nodes = {'n3': [0, 10, 99, 66, 124, 208]}
        sampled_graph = sample_neighbors(dist_graph, nodes, 3)
        block = dgl.to_block(sampled_graph, nodes)
        block.edata[dgl.EID] = sampled_graph.edata[dgl.EID]
    except Exception as e:
        print(e)
        block = None
    dgl.distributed.exit_client()
    return block, gpb

def start_hetero_etype_sample_client(rank, tmpdir, disable_shared_mem, fanout=3):
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
        nodes = {'n3': [0, 10, 99, 66, 124, 208]}
        sampled_graph = sample_etype_neighbors(dist_graph, nodes, dgl.ETYPE, fanout)
        block = dgl.to_block(sampled_graph, nodes)
        block.edata[dgl.EID] = sampled_graph.edata[dgl.EID]
    except Exception as e:
        print(e)
        block = None
    dgl.distributed.exit_client()
    return block, gpb

def check_rpc_hetero_sampling_shuffle(tmpdir, num_server):
    ip_config = open("rpc_ip_config.txt", "w")
    for _ in range(num_server):
        ip_config.write('{}\n'.format(get_local_usable_addr()))
    ip_config.close()

    g = create_random_hetero()
    num_parts = num_server
    num_hops = 1

    partition_graph(g, 'test_sampling', num_parts, tmpdir,
                    num_hops=num_hops, part_method='metis', reshuffle=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(i, tmpdir, num_server > 1, 'test_sampling'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    time.sleep(3)
    block, gpb = start_hetero_sample_client(0, tmpdir, num_server > 1)
    print("Done sampling")
    for p in pserver_list:
        p.join()

    orig_nid_map = {ntype: F.zeros((g.number_of_nodes(ntype),), dtype=F.int64) for ntype in g.ntypes}
    orig_eid_map = {etype: F.zeros((g.number_of_edges(etype),), dtype=F.int64) for etype in g.etypes}
    for i in range(num_server):
        part, _, _, _, _, _, _ = load_partition(tmpdir / 'test_sampling.json', i)
        ntype_ids, type_nids = gpb.map_to_per_ntype(part.ndata[dgl.NID])
        for ntype_id, ntype in enumerate(g.ntypes):
            idx = ntype_ids == ntype_id
            F.scatter_row_inplace(orig_nid_map[ntype], F.boolean_mask(type_nids, idx),
                                  F.boolean_mask(part.ndata['orig_id'], idx))
        etype_ids, type_eids = gpb.map_to_per_etype(part.edata[dgl.EID])
        for etype_id, etype in enumerate(g.etypes):
            idx = etype_ids == etype_id
            F.scatter_row_inplace(orig_eid_map[etype], F.boolean_mask(type_eids, idx),
                                  F.boolean_mask(part.edata['orig_id'], idx))

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

def check_rpc_hetero_etype_sampling_shuffle(tmpdir, num_server):
    ip_config = open("rpc_ip_config.txt", "w")
    for _ in range(num_server):
        ip_config.write('{}\n'.format(get_local_usable_addr()))
    ip_config.close()
    g = create_random_hetero(dense=True)
    num_parts = num_server
    num_hops = 1

    partition_graph(g, 'test_sampling', num_parts, tmpdir,
                    num_hops=num_hops, part_method='metis', reshuffle=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(i, tmpdir, num_server > 1, 'test_sampling'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    time.sleep(3)
    fanout = 3
    block, gpb = start_hetero_etype_sample_client(0, tmpdir, num_server > 1, fanout)
    print("Done sampling")
    for p in pserver_list:
        p.join()

    src, dst = block.edges(etype=('n1', 'r2', 'n3'))
    assert len(src) == 18
    src, dst = block.edges(etype=('n2', 'r3', 'n3'))
    assert len(src) == 18

    orig_nid_map = {ntype: F.zeros((g.number_of_nodes(ntype),), dtype=F.int64) for ntype in g.ntypes}
    orig_eid_map = {etype: F.zeros((g.number_of_edges(etype),), dtype=F.int64) for etype in g.etypes}
    for i in range(num_server):
        part, _, _, _, _, _, _ = load_partition(tmpdir / 'test_sampling.json', i)
        ntype_ids, type_nids = gpb.map_to_per_ntype(part.ndata[dgl.NID])
        for ntype_id, ntype in enumerate(g.ntypes):
            idx = ntype_ids == ntype_id
            F.scatter_row_inplace(orig_nid_map[ntype], F.boolean_mask(type_nids, idx),
                                  F.boolean_mask(part.ndata['orig_id'], idx))
        etype_ids, type_eids = gpb.map_to_per_etype(part.edata[dgl.EID])
        for etype_id, etype in enumerate(g.etypes):
            idx = etype_ids == etype_id
            F.scatter_row_inplace(orig_eid_map[etype], F.boolean_mask(type_eids, idx),
                                  F.boolean_mask(part.edata['orig_id'], idx))

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

# Wait non shared memory graph store
@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name == 'tensorflow', reason='Not support tensorflow for now')
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support")
@pytest.mark.parametrize("num_server", [1, 2])
def test_rpc_sampling_shuffle(num_server):
    import tempfile
    os.environ['DGL_DIST_MODE'] = 'distributed'
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_sampling_shuffle(Path(tmpdirname), num_server)
        check_rpc_hetero_sampling_shuffle(Path(tmpdirname), num_server)
        check_rpc_hetero_etype_sampling_shuffle(Path(tmpdirname), num_server)

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
    ip_config = open("rpc_ip_config.txt", "w")
    for _ in range(num_server):
        ip_config.write('{}\n'.format(get_local_usable_addr()))
    ip_config.close()

    g = CitationGraphDataset("cora")[0]
    g.readonly()
    num_parts = num_server

    partition_graph(g, 'test_in_subgraph', num_parts, tmpdir,
                    num_hops=1, part_method='metis', reshuffle=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(i, tmpdir, num_server > 1, 'test_in_subgraph'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    nodes = [0, 10, 99, 66, 1024, 2008]
    time.sleep(3)
    sampled_graph = start_in_subgraph_client(0, tmpdir, num_server > 1, nodes)
    for p in pserver_list:
        p.join()


    orig_nid = F.zeros((g.number_of_nodes(),), dtype=F.int64, ctx=F.cpu())
    orig_eid = F.zeros((g.number_of_edges(),), dtype=F.int64, ctx=F.cpu())
    for i in range(num_server):
        part, _, _, _, _, _, _ = load_partition(tmpdir / 'test_in_subgraph.json', i)
        orig_nid[part.ndata[dgl.NID]] = part.ndata['orig_id']
        orig_eid[part.edata[dgl.EID]] = part.edata['orig_id']

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
    import tempfile
    os.environ['DGL_DIST_MODE'] = 'distributed'
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_in_subgraph_shuffle(Path(tmpdirname), 2)

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name == 'tensorflow', reason='Not support tensorflow for now')
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support")
def test_standalone_etype_sampling():
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

    test_rpc_sampling_shuffle(1)
    test_rpc_sampling_shuffle(2)
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
        check_rpc_sampling_shuffle(Path(tmpdirname), 2)
        check_rpc_hetero_sampling_shuffle(Path(tmpdirname), 1)
        check_rpc_hetero_sampling_shuffle(Path(tmpdirname), 2)
