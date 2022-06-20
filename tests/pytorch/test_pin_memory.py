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
def test_unpin_automatically():
    # run a sufficient number of iterations such that the memory pool should be
    # re-used
    for j in range(10):
        t = torch.ones(10000, 10)
        assert not F.is_pinned(t)
        nd = dgl.utils.pin_memory_inplace(t)
        assert F.is_pinned(t)
        del nd
        # dgl.ndarray will unpin its data upon destruction
        assert not F.is_pinned(t)
        del t

if __name__ == "__main__":
    test_pin_noncontiguous()
    test_pin_view()
    test_unpin_automatically()
