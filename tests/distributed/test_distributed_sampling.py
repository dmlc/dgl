import dgl
import unittest
import os
from dgl.data import CitationGraphDataset
from dgl.distributed.sampling import sample_neighbors
from dgl.distributed import partition_graph, load_partition, load_partition_book
import sys
import multiprocessing as mp
import numpy as np
import backend as F
import time
from utils import get_local_usable_addr
from pathlib import Path

from dgl.distributed import DistGraphServer, DistGraph


def start_server(rank, tmpdir):
    import dgl
    g = DistGraphServer(rank, "rpc_sampling_ip_config.txt", 1, "test_sampling",
                        tmpdir / 'test_sampling.json', disable_shared_mem=True)
    g.start()


def start_client(rank, tmpdir):
    import dgl
    _, _, _, gpb = load_partition(tmpdir / 'test_sampling.json', rank)
    dist_graph = DistGraph("rpc_sampling_ip_config.txt", "test_sampling", gpb=gpb)
    sampled_graph = sample_neighbors(dist_graph, [0, 10, 99, 66, 1024, 2008], 3)
    dgl.distributed.shutdown_servers()
    dgl.distributed.finalize_client()
    return sampled_graph


def check_rpc_sampling(tmpdir):
    num_server = 2
    ip_config = open("rpc_sampling_ip_config.txt", "w")
    for _ in range(num_server):
        ip_config.write('{} 1\n'.format(get_local_usable_addr()))
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
        p = ctx.Process(target=start_server, args=(i, tmpdir))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    time.sleep(3)
    sampled_graph = start_client(0, tmpdir)
    print("Done sampling")
    for p in pserver_list:
        p.join()

    src, dst = sampled_graph.edges()
    assert sampled_graph.number_of_nodes() == g.number_of_nodes()
    assert np.all(F.asnumpy(g.has_edges_between(src, dst)))
    eids = g.edge_ids(src, dst)
    assert np.array_equal(
        F.asnumpy(sampled_graph.edata[dgl.EID]), F.asnumpy(eids))

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name == 'tensorflow', reason='Not support tensorflow for now')
def test_rpc_sampling():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = "/tmp/sampling"
        check_rpc_sampling(Path(tmpdirname))

def check_rpc_sampling_shuffle(tmpdir):
    num_server = 2
    ip_config = open("rpc_sampling_ip_config.txt", "w")
    for _ in range(num_server):
        ip_config.write('{} 1\n'.format(get_local_usable_addr()))
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
        p = ctx.Process(target=start_server, args=(i, tmpdir))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    time.sleep(3)
    sampled_graph = start_client(0, tmpdir)
    print("Done sampling")
    for p in pserver_list:
        p.join()

    orig_nid = F.zeros((g.number_of_nodes(),), dtype=F.int64)
    orig_eid = F.zeros((g.number_of_edges(),), dtype=F.int64)
    for i in range(num_server):
        part, _, _, _ = load_partition(tmpdir / 'test_sampling.json', i)
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
def test_rpc_sampling_shuffle():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = "/tmp/sampling"
        check_rpc_sampling_shuffle(Path(tmpdirname))

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = "/tmp/sampling"
        check_rpc_sampling(Path(tmpdirname))
        check_rpc_sampling_shuffle(Path(tmpdirname))
