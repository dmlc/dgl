import torch as th
import backend as F

from dgl.nn import NodeEmbedding
from dgl.optim import SparseAdam, SparseAdagrad

import unittest, os

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
def test_sparse_adam():
    num_embs = 10
    emb_dim = 4
    device=F.ctx()
    dgl_emb = NodeEmbedding(num_embs, emb_dim, 'test')
    torch_emb = th.nn.Embedding(num_embs, emb_dim, sparse=True)
    th.manual_seed(0)
    th.nn.init.uniform_(torch_emb.weight, 0, 1.0)
    th.manual_seed(0)
    th.nn.init.uniform_(dgl_emb.emb_tensor, 0, 1.0)

    dgl_adam = SparseAdam(params=[dgl_emb], lr=0.01)
    torch_adam = th.optim.SparseAdam(list(torch_emb.parameters()), lr=0.01)

    # first step
    idx = th.randint(0, num_embs, size=(4,))
    dgl_value = dgl_emb(idx, device).to(th.device('cpu'))
    torch_value = torch_emb(idx)
    labels = th.ones((4,)).long()

    dgl_adam.zero_grad()
    torch_adam.zero_grad()
    dgl_loss = th.nn.functional.cross_entropy(dgl_value, labels)
    torch_loss = th.nn.functional.cross_entropy(torch_value, labels)
    dgl_loss.backward()
    torch_loss.backward()

    dgl_adam.step()
    torch_adam.step()
    assert F.allclose(dgl_emb.emb_tensor, torch_emb.weight)

    # Can not test second step
    # Pytorch sparseAdam maintains a global step
    # DGL sparseAdam use a per embedding step

if __name__ == '__main__':
    test_sparse_adam()
