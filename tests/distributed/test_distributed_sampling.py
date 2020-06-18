import dgl
import unittest
import os
from dgl.data import CitationGraphDataset
from dgl.distributed.sampling import sample_neighbors
from dgl.distributed import partition_graph, load_partition, GraphPartitionBook
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
                        tmpdir / 'test_sampling.json')
    g.start()


def start_client(rank, tmpdir, shuffle):
    import dgl
    dist_graph = DistGraph("rpc_sampling_ip_config.txt", "test_sampling")
    sampled_graph = sample_neighbors(dist_graph, [0, 10, 99, 66, 1024, 2008], 3)
    if shuffle:
        origin_id = dist_graph.ndata['orig_id']
        sampled_graph.ndata['orig_id'] = dist_graph.ndata['orig_id'][sampled_graph.ndata[dgl.NID]]
    dgl.distributed.shutdown_servers()
    dgl.distributed.finalize_client()
    return sampled_graph


@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name == 'tensorflow', reason='Not support tensorflow for now')
def test_rpc_sampling(tmpdir):
    num_server = 3
    ip_config = open("rpc_sampling_ip_config.txt", "w")
    ip_addr = get_local_usable_addr()
    ip_config.write('{} 1\n'.format(get_local_usable_addr()))
    ip_config.write('{} 1\n'.format(get_local_usable_addr()))
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
        pserver_list.append(p)

    time.sleep(3)
    sampled_graph = start_client(0, tmpdir, shuffle=False)
    print("Done sampling")
    for p in pserver_list:
        p.join()

    src, dst = sampled_graph.edges()
    assert sampled_graph.number_of_nodes() == g.number_of_nodes()
    assert np.all(F.asnumpy(g.has_edges_between(src, dst)))
    eids = g.edge_ids(src, dst)
    assert np.array_equal(
        F.asnumpy(sampled_graph.edata[dgl.EID]), F.asnumpy(eids))

# Wait non shared memory graph store
# @unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
# @unittest.skipIf(dgl.backend.backend_name == 'tensorflow', reason='Not support tensorflow for now')
# def test_rpc_sampling_shuffle(tmpdir):
#     num_server = 3
#     ip_config = open("rpc_sampling_ip_config.txt", "w")
#     ip_addr = get_local_usable_addr()
#     ip_config.write('{} 1\n'.format(get_local_usable_addr()))
#     ip_config.write('{} 1\n'.format(get_local_usable_addr()))
#     ip_config.write('{} 1\n'.format(get_local_usable_addr()))
#     ip_config.close()
    
#     g = CitationGraphDataset("cora")[0]
#     g.readonly()
#     print(g.idtype)
#     num_parts = num_server
#     num_hops = 1

#     partition_graph(g, 'test_sampling', num_parts, tmpdir,
#                     num_hops=num_hops, part_method='metis', reshuffle=True)

#     pserver_list = []
#     ctx = mp.get_context('spawn')
#     for i in range(num_server):
#         p = ctx.Process(target=start_server, args=(i, tmpdir))
#         p.start()
#         pserver_list.append(p)

#     time.sleep(3)
#     sampled_graph = start_client(0, tmpdir, shuffle=True)
#     print("Done sampling")
#     for p in pserver_list:
#         p.join()

#     src, dst = sampled_graph.edges()
#     src = sampled_graph.ndata['orig_id'][src]
#     dst = sampled_graph.ndata['orig_id'][dst]
#     assert sampled_graph.number_of_nodes() == g.number_of_nodes()
#     assert np.all(F.asnumpy(g.has_edges_between(src, dst)))
#     eids = g.edge_ids(src, dst)
#     assert np.array_equal(
#         F.asnumpy(sampled_graph.edata[dgl.EID]), F.asnumpy(eids))

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = "/tmp/sampling"
        test_rpc_sampling(Path(tmpdirname))
        # test_rpc_sampling_shuffle(Path(tmpdirname))
