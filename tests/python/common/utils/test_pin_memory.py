import backend as F

import dgl
import pytest


@pytest.mark.skipif(
    F._default_context_str == "cpu", reason="Need gpu for this test"
)
def test_pin_unpin():
    t = F.arange(0, 100, dtype=F.int64, ctx=F.cpu())

    assert not F.is_pinned(t)

    if F.backend_name == "pytorch":
        nd = dgl.utils.pin_memory_inplace(t)
        assert F.is_pinned(t)
        nd.unpin_memory_()
        assert not F.is_pinned(t)
        del nd

        # tensor will be unpinned immediately if the returned ndarray is not saved
        dgl.utils.pin_memory_inplace(t)
        assert not F.is_pinned(t)

        t_pin = t.pin_memory()
        # cannot unpin a tensor that is pinned outside of DGL
        with pytest.raises(dgl.DGLError):
            F.to_dgl_nd(t_pin).unpin_memory_()
    else:
        with pytest.raises(dgl.DGLError):
            # tensorflow and mxnet should throw an error
            dgl.utils.pin_memory_inplace(t)


if __name__ == "__main__":
    test_pin_unpin()
