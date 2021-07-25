import dgl.multiprocessing as mp
import unittest, os
import pytest

import torch as th
import dgl
import backend as F

def start_unified_tensor_worker(dev_id, input, seq_idx, rand_idx, output_seq_cpu, output_rand_cpu):
    device = th.device('cuda:'+str(dev_id))
    th.cuda.set_device(device)
    input_unified = dgl.contrib.UnifiedTensor(input, device=device)

    seq_idx = seq_idx.to(device)
    assert th.all(th.eq(output_seq_cpu, input_unified[seq_idx]))

    rand_idx = rand_idx.to(device)
    assert th.all(th.eq(output_rand_cpu, input_unified[rand_idx]))


@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(F.ctx().type == 'cpu', reason='gpu only test')
def test_unified_tensor():
    test_row_size = 65536
    test_col_size = 128

    rand_test_size = 8192

    input = th.rand((test_row_size, test_col_size))
    input_unified = dgl.contrib.UnifiedTensor(input, device=th.device('cuda'))

    seq_idx = th.arange(0, test_row_size)
    assert th.all(th.eq(input[seq_idx], input_unified[seq_idx]))

    seq_idx = seq_idx.to(th.device('cuda'))
    assert th.all(th.eq(input[seq_idx].to(th.device('cuda')), input_unified[seq_idx]))

    rand_idx = th.randint(0, test_row_size, (rand_test_size,))
    assert th.all(th.eq(input[rand_idx], input_unified[rand_idx]))

    rand_idx = rand_idx.to(th.device('cuda'))
    assert th.all(th.eq(input[rand_idx].to(th.device('cuda')), input_unified[rand_idx]))

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(F.ctx().type == 'cpu', reason='gpu only test')
def test_multi_gpu_unified_tensor():
    if F.ctx().type == 'cuda' and th.cuda.device_count() > 1:
        pytest.skip("Only one GPU detected, skip multi-gpu test.")

    num_workers = th.cuda.device_count()

    test_row_size = 65536
    test_col_size = 128

    rand_test_size = 8192

    input = th.rand((test_row_size, test_col_size)).share_memory_()
    seq_idx = th.arange(0, test_row_size)
    rand_idx = th.randint(0, test_row_size, (rand_test_size,))

    output_seq = []
    output_rand = []

    output_seq_cpu = input[seq_idx]
    output_rand_cpu = input[rand_idx]

    worker_list = []

    ctx = mp.get_context('spawn')
    for i in range(num_workers):
        p = ctx.Process(target=start_unified_tensor_worker,
                        args=(i, input, seq_idx, rand_idx, output_seq_cpu, output_rand_cpu,))
        p.start()
        worker_list.append(p)

    for p in worker_list:
        p.join()
    for p in worker_list:
        assert p.exitcode == 0


if __name__ == '__main__':
    test_unified_tensor()
    test_multi_gpu_unified_tensor()
