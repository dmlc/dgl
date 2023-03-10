import multiprocessing as mp
import os
import unittest

import backend as F
import pytest
import torch as th

from dgl.nn import NodeEmbedding
from dgl.optim import SparseAdam


def initializer(emb):
    th.manual_seed(0)
    emb.uniform_(-1.0, 1.0)
    return emb


def check_all_set_all_get_emb(device, init_emb):
    num_embs = init_emb.shape[0]
    emb_dim = init_emb.shape[1]
    dgl_emb = NodeEmbedding(num_embs, emb_dim, "test", device=device)
    dgl_emb.all_set_embedding(init_emb)

    out_emb = dgl_emb.all_get_embedding()
    assert F.allclose(init_emb, out_emb)


def check_all_set_all_get_optm_state(
    device, state_step, state_mem, state_power
):
    num_embs = state_mem.shape[0]
    emb_dim = state_mem.shape[1]
    dgl_emb = NodeEmbedding(num_embs, emb_dim, "test", device=device)
    optm = SparseAdam(params=[dgl_emb], lr=0.01)

    dgl_emb._all_set_optm_state((state_step, state_mem, state_power))

    out_step, out_mem, out_power = dgl_emb._all_get_optm_state()

    assert F.allclose(state_step, out_step)
    assert F.allclose(state_mem, out_mem)
    assert F.allclose(state_power, out_power)


def start_sparse_worker(rank, world_size, test, args):
    print("start sparse worker {}".format(rank))
    dist_init_method = "tcp://{master_ip}:{master_port}".format(
        master_ip="127.0.0.1", master_port="12345"
    )
    backend = "gloo"
    device = F.ctx()
    if device.type == "cuda":
        device = th.device(rank)
        th.cuda.set_device(device)
    th.distributed.init_process_group(
        backend=backend,
        init_method=dist_init_method,
        world_size=world_size,
        rank=rank,
    )

    test(device, *args)
    th.distributed.barrier()
    th.distributed.destroy_process_group()


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
@pytest.mark.parametrize("num_workers", [1, 2, 3])
def test_multiprocess_sparse_emb_get_set(num_workers):
    if F.ctx().type == "cuda" and th.cuda.device_count() < num_workers:
        pytest.skip("Not enough GPUs to run test.")

    worker_list = []

    init_emb = th.rand([1000, 8])

    ctx = mp.get_context("spawn")
    for i in range(num_workers):
        p = ctx.Process(
            target=start_sparse_worker,
            args=(i, num_workers, check_all_set_all_get_emb, (init_emb,)),
        )
        p.start()
        worker_list.append(p)

    for p in worker_list:
        p.join()
    for p in worker_list:
        assert p.exitcode == 0


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
@pytest.mark.parametrize("num_workers", [1, 2, 3])
def test_multiprocess_sparse_emb_get_set_optm_state(num_workers):
    if F.ctx().type == "cuda" and th.cuda.device_count() < num_workers:
        pytest.skip("Not enough GPUs to run test.")

    worker_list = []

    num_embs, emb_dim = 1000, 8
    state_step = th.randint(1000, (num_embs,))
    state_mem = th.rand((num_embs, emb_dim))
    state_power = th.rand((num_embs, emb_dim))

    ctx = mp.get_context("spawn")
    for i in range(num_workers):
        p = ctx.Process(
            target=start_sparse_worker,
            args=(
                i,
                num_workers,
                check_all_set_all_get_optm_state,
                (state_step, state_mem, state_power),
            ),
        )
        p.start()
        worker_list.append(p)

    for p in worker_list:
        p.join()
    for p in worker_list:
        assert p.exitcode == 0


if __name__ == "__main__":
    # test_multiprocess_sparse_emb_get_set(1)
    # test_multiprocess_sparse_emb_get_set(2)
    # test_multiprocess_sparse_emb_get_set(3)

    test_multiprocess_sparse_emb_get_set_optm_state(1)
    # test_multiprocess_sparse_emb_get_set_optm_state(2)
    # test_multiprocess_sparse_emb_get_set_optm_state(3)
