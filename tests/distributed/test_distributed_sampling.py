import dgl
import unittest
import os
from dgl.data import CitationGraphDataset
from dgl.distributed import sample_neighbors, find_edges
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
from dgl.distributed import DistGraphServer, DistGraph


def start_server(rank, tmpdir, disable_shared_mem, graph_name):
    g = DistGraphServer(rank, "rpc_ip_config.txt", 1, 1,
                        tmpdir / (graph_name + '.json'), disable_shared_mem=disable_shared_mem)
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

def start_find_edges_client(rank, tmpdir, disable_shared_mem, eids):
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(tmpdir / 'test_find_edges.json', rank)
    dgl.distributed.initialize("rpc_ip_config.txt")
    dist_graph = DistGraph("test_find_edges", gpb=gpb)
    try:
        u, v = find_edges(dist_graph, eids)
    except Exception as e:
        print(e)
        u, v = None, None
    dgl.distributed.exit_client()
    return u, v

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

    partition_graph(g, 'test_find_edges', num_parts, tmpdir,
                    num_hops=1, part_method='metis', reshuffle=True)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(i, tmpdir, num_server > 1, 'test_find_edges'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    orig_nid = F.zeros((g.number_of_nodes(),), dtype=F.int64)
    orig_eid = F.zeros((g.number_of_edges(),), dtype=F.int64)
    for i in range(num_server):
        part, _, _, _, _, _, _ = load_partition(tmpdir / 'test_find_edges.json', i)
        orig_nid[part.ndata[dgl.NID]] = part.ndata['orig_id']
        orig_eid[part.edata[dgl.EID]] = part.edata['orig_id']

    time.sleep(3)
    eids = F.tensor(np.random.randint(g.number_of_edges(), size=100))
    u, v = g.find_edges(orig_eid[eids])
    du, dv = start_find_edges_client(0, tmpdir, num_server > 1, eids)
    du = orig_nid[du]
    dv = orig_nid[dv]
    assert F.array_equal(u, du)
    assert F.array_equal(v, dv)

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

    orig_nid = F.zeros((g.number_of_nodes(),), dtype=F.int64)
    orig_eid = F.zeros((g.number_of_edges(),), dtype=F.int64)
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

def create_random_hetero():
    num_nodes = {'n1': 1010, 'n2': 1000, 'n3': 1020}
    etypes = [('n1', 'r1', 'n2'),
              ('n1', 'r2', 'n3'),
              ('n2', 'r3', 'n3')]
    edges = {}
    for etype in etypes:
        src_ntype, _, dst_ntype = etype
        arr = spsp.random(num_nodes[src_ntype], num_nodes[dst_ntype], density=0.001, format='coo',
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
        nodes = gpb.map_to_homo_nid(nodes['n3'], 'n3')
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

    orig_nid_map = F.zeros((g.number_of_nodes(),), dtype=F.int64)
    orig_eid_map = F.zeros((g.number_of_edges(),), dtype=F.int64)
    for i in range(num_server):
        part, _, _, _, _, _, _ = load_partition(tmpdir / 'test_sampling.json', i)
        F.scatter_row_inplace(orig_nid_map, part.ndata[dgl.NID], part.ndata['orig_id'])
        F.scatter_row_inplace(orig_eid_map, part.edata[dgl.EID], part.edata['orig_id'])

    src, dst = block.edges()
    # These are global Ids after shuffling.
    shuffled_src = F.gather_row(block.srcdata[dgl.NID], src)
    shuffled_dst = F.gather_row(block.dstdata[dgl.NID], dst)
    shuffled_eid = block.edata[dgl.EID]
    # Get node/edge types.
    etype, _ = gpb.map_to_per_etype(shuffled_eid)
    src_type, _ = gpb.map_to_per_ntype(shuffled_src)
    dst_type, _ = gpb.map_to_per_ntype(shuffled_dst)
    etype = F.asnumpy(etype)
    src_type = F.asnumpy(src_type)
    dst_type = F.asnumpy(dst_type)
    # These are global Ids in the original graph.
    orig_src = F.asnumpy(F.gather_row(orig_nid_map, shuffled_src))
    orig_dst = F.asnumpy(F.gather_row(orig_nid_map, shuffled_dst))
    orig_eid = F.asnumpy(F.gather_row(orig_eid_map, shuffled_eid))

    etype_map = {g.get_etype_id(etype):etype for etype in g.etypes}
    etype_to_eptype = {g.get_etype_id(etype):(src_ntype, dst_ntype) for src_ntype, etype, dst_ntype in g.canonical_etypes}
    for e in np.unique(etype):
        src_t = src_type[etype == e]
        dst_t = dst_type[etype == e]
        assert np.all(src_t == src_t[0])
        assert np.all(dst_t == dst_t[0])

        # Check the node Ids and edge Ids.
        orig_src1, orig_dst1 = g.find_edges(orig_eid[etype == e], etype=etype_map[e])
        assert np.all(F.asnumpy(orig_src1) == orig_src[etype == e])
        assert np.all(F.asnumpy(orig_dst1) == orig_dst[etype == e])

        # Check the node types.
        src_ntype, dst_ntype = etype_to_eptype[e]
        assert np.all(src_t == g.get_ntype_id(src_ntype))
        assert np.all(dst_t == g.get_ntype_id(dst_ntype))

# Wait non shared memory graph store
@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name == 'tensorflow', reason='Not support tensorflow for now')
@pytest.mark.parametrize("num_server", [1, 2])
def test_rpc_sampling_shuffle(num_server):
    import tempfile
    os.environ['DGL_DIST_MODE'] = 'distributed'
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_sampling_shuffle(Path(tmpdirname), num_server)
        check_rpc_hetero_sampling_shuffle(Path(tmpdirname), num_server)

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


    orig_nid = F.zeros((g.number_of_nodes(),), dtype=F.int64)
    orig_eid = F.zeros((g.number_of_edges(),), dtype=F.int64)
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

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.environ['DGL_DIST_MODE'] = 'standalone'
        check_standalone_sampling(Path(tmpdirname), True)
        check_standalone_sampling(Path(tmpdirname), False)
        os.environ['DGL_DIST_MODE'] = 'distributed'
        check_rpc_sampling(Path(tmpdirname), 2)
        check_rpc_sampling(Path(tmpdirname), 1)
        check_rpc_find_edges_shuffle(Path(tmpdirname), 2)
        check_rpc_find_edges_shuffle(Path(tmpdirname), 1)
        check_rpc_in_subgraph_shuffle(Path(tmpdirname), 2)
        check_rpc_sampling_shuffle(Path(tmpdirname), 1)
        check_rpc_sampling_shuffle(Path(tmpdirname), 2)
        check_rpc_hetero_sampling_shuffle(Path(tmpdirname), 1)
        check_rpc_hetero_sampling_shuffle(Path(tmpdirname), 2)
