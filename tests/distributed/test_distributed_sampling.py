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
from dgl.distributed import DistGraphServer, DistGraph


def start_server(rank, tmpdir, disable_shared_mem, graph_name):
    g = DistGraphServer(rank, "rpc_ip_config.txt", 1, 1,
                        tmpdir / (graph_name + '.json'), disable_shared_mem=disable_shared_mem)
    g.start()


def start_sample_client(rank, tmpdir, disable_shared_mem):
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _ = load_partition(tmpdir / 'test_sampling.json', rank)
    dgl.distributed.initialize("rpc_ip_config.txt", 1)
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
        _, _, _, gpb, _ = load_partition(tmpdir / 'test_find_edges.json', rank)
    dgl.distributed.initialize("rpc_ip_config.txt", 1)
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

def check_rpc_find_edges(tmpdir, num_server):
    ip_config = open("rpc_ip_config.txt", "w")
    for _ in range(num_server):
        ip_config.write('{}\n'.format(get_local_usable_addr()))
    ip_config.close()

    g = CitationGraphDataset("cora")[0]
    g.readonly()
    num_parts = num_server

    partition_graph(g, 'test_find_edges', num_parts, tmpdir,
                    num_hops=1, part_method='metis', reshuffle=False)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(i, tmpdir, num_server > 1, 'test_find_edges'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    time.sleep(3)
    eids = F.tensor(np.random.randint(g.number_of_edges(), size=100))
    u, v = g.find_edges(eids)
    du, dv = start_find_edges_client(0, tmpdir, num_server > 1, eids)
    assert F.array_equal(u, du)
    assert F.array_equal(v, dv)

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name == 'tensorflow', reason='Not support tensorflow for now')
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
        part, _, _, _, _ = load_partition(tmpdir / 'test_sampling.json', i)
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

# Wait non shared memory graph store
@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name == 'tensorflow', reason='Not support tensorflow for now')
@pytest.mark.parametrize("num_server", [1, 2])
def test_rpc_sampling_shuffle(num_server):
    import tempfile
    os.environ['DGL_DIST_MODE'] = 'distributed'
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_sampling_shuffle(Path(tmpdirname), num_server)

def check_standalone_sampling(tmpdir):
    g = CitationGraphDataset("cora")[0]
    num_parts = 1
    num_hops = 1
    partition_graph(g, 'test_sampling', num_parts, tmpdir,
                    num_hops=num_hops, part_method='metis', reshuffle=False)

    os.environ['DGL_DIST_MODE'] = 'standalone'
    dgl.distributed.initialize("rpc_ip_config.txt", 1)
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
        check_standalone_sampling(Path(tmpdirname))

def start_in_subgraph_client(rank, tmpdir, disable_shared_mem, nodes):
    gpb = None
    dgl.distributed.initialize("rpc_ip_config.txt", 1)
    if disable_shared_mem:
        _, _, _, gpb, _ = load_partition(tmpdir / 'test_in_subgraph.json', rank)
    dist_graph = DistGraph("test_in_subgraph", gpb=gpb)
    try:
        sampled_graph = dgl.distributed.in_subgraph(dist_graph, nodes)
    except Exception as e:
        print(e)
        sampled_graph = None
    dgl.distributed.exit_client()
    return sampled_graph


def check_rpc_in_subgraph(tmpdir, num_server):
    ip_config = open("rpc_ip_config.txt", "w")
    for _ in range(num_server):
        ip_config.write('{}\n'.format(get_local_usable_addr()))
    ip_config.close()

    g = CitationGraphDataset("cora")[0]
    g.readonly()
    num_parts = num_server

    partition_graph(g, 'test_in_subgraph', num_parts, tmpdir,
                    num_hops=1, part_method='metis', reshuffle=False)

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

    src, dst = sampled_graph.edges()
    assert sampled_graph.number_of_nodes() == g.number_of_nodes()
    subg1 = dgl.in_subgraph(g, nodes)
    src1, dst1 = subg1.edges()
    assert np.all(np.sort(F.asnumpy(src)) == np.sort(F.asnumpy(src1)))
    assert np.all(np.sort(F.asnumpy(dst)) == np.sort(F.asnumpy(dst1)))
    eids = g.edge_ids(src, dst)
    assert np.array_equal(
        F.asnumpy(sampled_graph.edata[dgl.EID]), F.asnumpy(eids))

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name == 'tensorflow', reason='Not support tensorflow for now')
def test_rpc_in_subgraph():
    import tempfile
    os.environ['DGL_DIST_MODE'] = 'distributed'
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_rpc_in_subgraph(Path(tmpdirname), 2)

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.environ['DGL_DIST_MODE'] = 'standalone'
        check_standalone_sampling(Path(tmpdirname))
        os.environ['DGL_DIST_MODE'] = 'distributed'
        check_rpc_in_subgraph(Path(tmpdirname), 2)
        check_rpc_sampling_shuffle(Path(tmpdirname), 1)
        check_rpc_sampling_shuffle(Path(tmpdirname), 2)
        check_rpc_sampling(Path(tmpdirname), 2)
        check_rpc_sampling(Path(tmpdirname), 1)
        check_rpc_find_edges(Path(tmpdirname), 2)
        check_rpc_find_edges(Path(tmpdirname), 1)
