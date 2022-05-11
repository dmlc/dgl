import backend as F
import dgl
import unittest
import torch

@unittest.skipIf(F._default_context_str == 'cpu', reason="Need gpu for this test")
def test_pin_noncontiguous():
    t = torch.empty([10, 100]).transpose(0, 1)

    assert not t.is_contiguous()
    assert not F.is_pinned(t)

    try:
        dgl.utils.pin_memory_inplace(t) 
    except dgl.DGLError as e:
        # passed
        return
    assert False, "Expected an exception"

@unittest.skipIf(F._default_context_str == 'cpu', reason="Need gpu for this test")
def test_pin_view():
    t = torch.empty([100, 10])
    v = t[10:20]

    assert v.is_contiguous()
    assert not F.is_pinned(t)

    try:
        dgl.utils.pin_memory_inplace(v) 
    except dgl.DGLError as e:
        # passed
        return
    assert False, "Expected an exception"


if __name__ == "__main__":
    test_pin_noncontiguous()
    test_pin_view()
