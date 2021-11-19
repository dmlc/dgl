import dgl.multiprocessing as mp
import unittest, os
import pytest

import torch as th
import dgl
import backend as F

def start_unified_tensor_worker(dev_id, input, seq_idx, rand_idx, output_seq, output_rand):
    device = th.device('cuda:'+str(dev_id))
    th.cuda.set_device(device)
    input_unified = dgl.contrib.UnifiedTensor(input, device=device)
    output_seq.copy_(input_unified[seq_idx.to(device)].cpu())
    output_rand.copy_(input_unified[rand_idx.to(device)].cpu())

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
@pytest.mark.parametrize("num_workers", [1, 2])
def test_multi_gpu_unified_tensor(num_workers):
    if F.ctx().type == 'cuda' and th.cuda.device_count() < num_workers:
        pytest.skip("Not enough number of GPUs to do this test, skip multi-gpu test.")

    test_row_size = 65536
    test_col_size = 128

    rand_test_size = 8192

    input = th.rand((test_row_size, test_col_size)).share_memory_()
    seq_idx = th.arange(0, test_row_size).share_memory_()
    rand_idx = th.randint(0, test_row_size, (rand_test_size,)).share_memory_()

    output_seq = []
    output_rand = []

    output_seq_cpu = input[seq_idx]
    output_rand_cpu = input[rand_idx]

    worker_list = []

    ctx = mp.get_context('spawn')
    for i in range(num_workers):
        output_seq.append(th.zeros((test_row_size, test_col_size)).share_memory_())
        output_rand.append(th.zeros((rand_test_size, test_col_size)).share_memory_())
        p = ctx.Process(target=start_unified_tensor_worker,
                        args=(i, input, seq_idx, rand_idx, output_seq[i], output_rand[i],))
        p.start()
        worker_list.append(p)

    for p in worker_list:
        p.join()
    for p in worker_list:
        assert p.exitcode == 0
    for i in range(num_workers):
        assert th.all(th.eq(output_seq_cpu, output_seq[i]))
        assert th.all(th.eq(output_rand_cpu, output_rand[i]))


if __name__ == '__main__':
    test_unified_tensor()
    test_multi_gpu_unified_tensor(1)
    test_multi_gpu_unified_tensor(2)
