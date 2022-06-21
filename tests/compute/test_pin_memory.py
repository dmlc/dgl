import backend as F
import dgl
import pytest

@pytest.mark.skipif(F._default_context_str == 'cpu', reason="Need gpu for this test")
def test_pin_unpin():
    t = F.arange(0, 100, dtype=F.int64, ctx=F.cpu())

    assert not F.is_pinned(t)

    if F.backend_name == 'pytorch':
        dgl.utils.pin_memory_inplace(t)
        assert F.is_pinned(t)
        dgl.utils.unpin_memory_inplace(t)
        assert not F.is_pinned(t)
    else:
        with pytest.raises(dgl.DGLError):
            # tensorflow and mxnet should throw an erro
            dgl.utils.pin_memory_inplace(t)

if __name__ == "__main__":
    test_pin_unpin()
