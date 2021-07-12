import unittest, os

import torch as th
import dgl
import backend as F

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

if __name__ == '__main__':
    test_unified_tensor()
