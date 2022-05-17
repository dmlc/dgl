import backend as F
import dgl
import gc
import unittest
import torch


@unittest.skipIf(F.ctx().type == 'cpu', reason='Pinning memory tests require GPU.')
def test_unpin_tensoradapater():
    # run a sufficient number of iterations such that the memory pool should be
    # re-used
    for j in range(3):
        t = F.zerocopy_from_dlpack(dgl.ndarray.empty(
            [10000, 10],
            F.reverse_data_type_dict[F.float32],
            ctx=dgl.utils.to_dgl_context(torch.device('cpu'))).to_dlpack()).zero_()
        assert not F.is_pinned(t)
        dgl.utils.pin_memory_inplace(t)
        assert F.is_pinned(t)
        del t
