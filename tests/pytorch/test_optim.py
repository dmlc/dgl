import time
import multiprocessing as mp
import unittest, os
import pytest

import torch as th
import backend as F

from dgl.nn import NodeEmbedding
from dgl.optim import SparseAdam, SparseAdagrad

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
    th.nn.init.uniform_(dgl_emb.weight, 0, 1.0)

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
    assert F.allclose(dgl_emb.weight, torch_emb.weight)

    # Can not test second step
    # Pytorch sparseAdam maintains a global step
    # DGL sparseAdam use a per embedding step

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
def test_sparse_adam_zero_step():
    num_embs = 10
    emb_dim = 4
    device=F.ctx()
    dgl_emb = NodeEmbedding(num_embs, emb_dim, 'test')
    torch_emb = th.nn.Embedding(num_embs, emb_dim, sparse=True)
    dgl_emb_zero = NodeEmbedding(num_embs, emb_dim, 'test2')
    torch_emb_zero = th.nn.Embedding(num_embs, emb_dim, sparse=True)
    th.manual_seed(0)
    th.nn.init.uniform_(torch_emb.weight, 0, 1.0)
    th.nn.init.uniform_(torch_emb_zero.weight, 0, 1.0)
    th.manual_seed(0)
    th.nn.init.uniform_(dgl_emb.weight, 0, 1.0)
    th.nn.init.uniform_(dgl_emb_zero.weight, 0, 1.0)

    dgl_adam = SparseAdam(params=[dgl_emb, dgl_emb_zero], lr=0.01)
    torch_adam = th.optim.SparseAdam(
        list(torch_emb.parameters()) + list(torch_emb_zero.parameters()), lr=0.01)

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
    assert F.allclose(dgl_emb.weight, torch_emb.weight)

def initializer(emb):
    th.manual_seed(0)
    emb.uniform_(-1.0, 1.0)
    return emb

def start_sparse_adam_worker(rank, world_size, has_zero_grad=False, num_embs=128, emb_dim=10):
    print('start sparse worker for adam {}'.format(rank))
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    backend = 'gloo'
    device=F.ctx()

    th.distributed.init_process_group(backend=backend,
                                      init_method=dist_init_method,
                                      world_size=world_size,
                                      rank=rank)

    dgl_emb = NodeEmbedding(num_embs, emb_dim, 'test', init_func=initializer)
    torch_emb = th.nn.Embedding(num_embs, emb_dim, sparse=True)
    th.manual_seed(0)
    th.nn.init.uniform_(torch_emb.weight, -1.0, 1.0)
    torch_emb = th.nn.parallel.DistributedDataParallel(torch_emb)

    if has_zero_grad:
        dgl_emb_zero = NodeEmbedding(num_embs, emb_dim, 'zero', init_func=initializer)
        torch_emb_zero = th.nn.Embedding(num_embs, emb_dim, sparse=True)
        th.manual_seed(0)
        th.nn.init.uniform_(torch_emb_zero.weight, -1.0, 1.0)
        torch_emb_zero = th.nn.parallel.DistributedDataParallel(torch_emb_zero)

        dgl_adam = SparseAdam(params=[dgl_emb, dgl_emb_zero], lr=0.01)
        torch_adam = th.optim.SparseAdam(
            list(torch_emb.module.parameters()) + list(torch_emb_zero.module.parameters()),
            lr=0.01)
    else:
        dgl_adam = SparseAdam(params=[dgl_emb], lr=0.01)
        torch_adam = th.optim.SparseAdam(list(torch_emb.module.parameters()), lr=0.01)

    start = (num_embs // world_size) * rank
    end = (num_embs // world_size) * (rank + 1)
    idx = th.randint(start, end, size=(4,))
    dgl_value = dgl_emb(idx, device).to(th.device('cpu'))
    torch_value = torch_emb(idx)
    labels = th.ones((4,)).long()

    dgl_adam.zero_grad()
    dgl_loss = th.nn.functional.cross_entropy(dgl_value, labels)
    dgl_loss.backward()
    dgl_adam.step()
    torch_loss = th.nn.functional.cross_entropy(torch_value, labels)
    torch_adam.zero_grad()
    torch_loss.backward()
    torch_adam.step()
    if rank == 0:
        after_step = dgl_emb(idx, device)
        assert F.allclose(dgl_emb.weight, torch_emb.module.weight)
        assert F.allclose(dgl_value, after_step) is False
    th.distributed.barrier()

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@pytest.mark.parametrize("num_workers", [2, 4, 8])
def test_multiprocess_sparse_adam(num_workers):
    worker_list = []

    ctx = mp.get_context('spawn')
    for i in range(num_workers):
        p = ctx.Process(target=start_sparse_adam_worker,
                        args=(i, num_workers))
        p.start()
        worker_list.append(p)

    for p in worker_list:
        p.join()

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@pytest.mark.parametrize("num_workers", [2, 4, 8])
def test_multiprocess_sparse_adam_zero_step(num_workers):
    worker_list = []

    ctx = mp.get_context('spawn')
    for i in range(num_workers):
        p = ctx.Process(target=start_sparse_adam_worker,
                        args=(i, num_workers, True))
        p.start()
        worker_list.append(p)

    for p in worker_list:
        p.join()

if __name__ == '__main__':
    test_sparse_adam()
    test_sparse_adam_zero_step()

    test_multiprocess_sparse_adam(2)
    test_multiprocess_sparse_adam(4)

    test_multiprocess_sparse_adam_zero_step(2)
    test_multiprocess_sparse_adam_zero_step(4)
