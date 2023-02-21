import multiprocessing as mp
import os
import unittest

import backend as F
import pytest
import torch as th

from dgl.nn import NodeEmbedding


def initializer(emb):
    th.manual_seed(0)
    emb.uniform_(-1.0, 1.0)
    return emb


def check_all_set_all_get_func(device, init_emb):
    num_embs = init_emb.shape[0]
    emb_dim = init_emb.shape[1]
    dgl_emb = NodeEmbedding(num_embs, emb_dim, "test", device=device)
    dgl_emb.all_set_embedding(init_emb)

    out_emb = dgl_emb.all_get_embedding()
    assert F.allclose(init_emb, out_emb)


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
            args=(i, num_workers, check_all_set_all_get_func, (init_emb,)),
        )
        p.start()
        worker_list.append(p)

    for p in worker_list:
        p.join()
    for p in worker_list:
        assert p.exitcode == 0


if __name__ == "__main__":
    test_sparse_emb_get_set(1)
    test_sparse_emb_get_set(2)
    test_sparse_emb_get_set(3)
