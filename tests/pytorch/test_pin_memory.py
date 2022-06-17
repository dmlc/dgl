import backend as F
import dgl
import pytest
import torch

@pytest.mark.skipif(F._default_context_str == 'cpu', reason="Need gpu for this test.")
def test_pin_noncontiguous():
    t = torch.empty([10, 100]).transpose(0, 1)

    assert not t.is_contiguous()
    assert not F.is_pinned(t)

    with pytest.raises(dgl.DGLError):
        dgl.utils.pin_memory_inplace(t)

@pytest.mark.skipif(F._default_context_str == 'cpu', reason="Need gpu for this test.")
def test_pin_view():
    t = torch.empty([100, 10])
    v = t[10:20]

    assert v.is_contiguous()
    assert not F.is_pinned(t)

    with pytest.raises(dgl.DGLError):
        dgl.utils.pin_memory_inplace(v)

@pytest.mark.skipif(F._default_context_str == 'cpu', reason='Need gpu for this test.')
def test_unpin_tensoradapater():
    # run a sufficient number of iterations such that the memory pool should be
    # re-used
    for j in range(5):
        nd = dgl.ndarray.empty(
            [10000, 10],
            F.reverse_data_type_dict[F.float32],
            ctx=dgl.utils.to_dgl_context(torch.device('cpu')))
        t = F.zerocopy_from_dlpack(nd.to_dlpack()).zero_()
        assert not F.is_pinned(t)
        nd.pin_memory_()
        assert F.is_pinned(t)
        del t

if __name__ == "__main__":
    test_pin_noncontiguous()
    test_pin_view()
    test_unpin_tensoradapater()
