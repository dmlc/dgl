import numpy as np
import os
from scipy import sparse as spsp
import dgl
import backend as F
import unittest
from pathlib import Path
from utils import get_local_usable_addr

def create_random_graph(n):
    arr = (spsp.random(n, n, density=0.001, format='coo', random_state=100) != 0).astype(np.int64)
    return dgl.graph(arr)

def init_emb(shape, dtype):
    arr = F.ones(shape, dtype, F.cpu())
    return arr

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(dgl.backend.backend_name != 'pytorch', reason='Test with Pytorch Adagrad')
def test_adagrad(tmpdir):
    import torch as th
    import torch.nn as nn
    import torch.nn.functional as F

    g = create_random_graph(100)
    print('test1')

    # Partition the graph
    num_parts = 1
    graph_name = 'sparse_adagrad_graph'
    dgl.distributed.partition_graph(g, graph_name, num_parts, tmpdir)
    print('test2')

    # Prepare ip config
    ip_config = open("rpc_ip_config.txt", "w")
    ip_config.write('{}\n'.format(get_local_usable_addr()))
    ip_config.close()
    print('test3')

    dgl.distributed.initialize("rpc_ip_config.txt")
    print('test4')
    g = dgl.distributed.DistGraph(graph_name,
                                  part_config=tmpdir / 'sparse_adagrad_graph.json')
    print('test5')
    embed_size=10

    w1 = nn.Parameter(th.Tensor(10, 10))
    w2 = nn.Parameter(th.Tensor(10, 10))
    nn.init.xavier_uniform_(w1)
    w2.data[:] = w1.data
    print('test6')

    dgl_emb = dgl.distributed.DistEmbedding(
            g.number_of_nodes(),
            embed_size,
            'test',
            init_emb)
    print('test7')
    torch_embeds = th.nn.Embedding(g.number_of_nodes(), embed_size, sparse=True)
    nn.init.ones_(torch_embeds.weight)
    print('test8')

    emb_optimizer = dgl.distributed.SparseAdagrad([dgl_emb], lr=0.01)
    th_emb_optimizer = th.optim.Adagrad(torch_embeds.parameters(), lr=0.01)
    print('test9')

    for _ in range(10):
        # We only test on the first 50 nodes.
        idx = th.tensor(np.random.choice(50, 10))
        truth = th.ones((len(idx),)).long()

        # different embs
        th_emb_optimizer.zero_grad()
        dgl_res = dgl_emb(idx)
        th_res = torch_embeds(idx)
        result1 = dgl_res @ w1
        result2 = th_res @ w2

        loss1 = F.cross_entropy(result1, truth)
        loss2 = F.cross_entropy(result2, truth)
        print('test10')
        loss1.backward()
        loss2.backward()
        print('test11')
        emb_optimizer.step()
        th_emb_optimizer.step()
        print('test12')

        with th.no_grad():
            np.testing.assert_almost_equal(dgl_res.grad.numpy(),
                                           torch_embeds.weight.grad._values().numpy())
            np.testing.assert_almost_equal(dgl_emb(idx).numpy(),
                                           torch_embeds(idx).numpy())

if __name__ == '__main__':
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        test_adagrad(Path(tmpdirname))
