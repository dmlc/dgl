import backend as F

import dgl
import pytest
import torch


@pytest.mark.skipif(
    F._default_context_str == "cpu", reason="Need gpu for this test."
)
def test_pin_noncontiguous():
    t = torch.empty([10, 100]).transpose(0, 1)

    assert not t.is_contiguous()
    assert not F.is_pinned(t)

    with pytest.raises(dgl.DGLError):
        dgl.utils.pin_memory_inplace(t)


@pytest.mark.skipif(
    F._default_context_str == "cpu", reason="Need gpu for this test."
)
def test_pin_view():
    t = torch.empty([100, 10])
    v = t[10:20]

    assert v.is_contiguous()
    assert not F.is_pinned(t)

    with pytest.raises(dgl.DGLError):
        dgl.utils.pin_memory_inplace(v)

    # make sure an empty view does not generate an error
    u = t[10:10]
    u = dgl.utils.pin_memory_inplace(u)


@pytest.mark.skipif(
    F._default_context_str == "cpu", reason="Need gpu for this test."
)
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


@pytest.mark.skipif(
    F._default_context_str == "cpu", reason="Need gpu for this test."
)
def test_pin_unpin_column():
    g = dgl.graph(([1, 2, 3, 4], [0, 0, 0, 0]))

    g.ndata["x"] = torch.randn(g.num_nodes())
    g.pin_memory_()
    assert g.is_pinned()
    assert g.ndata["x"].is_pinned()
    for col in g._node_frames[0].values():
        assert col.pinned_by_dgl
        assert col._data_nd is not None

    g.ndata["x"] = torch.randn(g.num_nodes())  # unpin the old ndata['x']
    assert g.is_pinned()
    for col in g._node_frames[0].values():
        assert not col.pinned_by_dgl
        assert col._data_nd is None
    assert not g.ndata["x"].is_pinned()


@pytest.mark.skipif(
    F._default_context_str == "cpu", reason="Need gpu for this test."
)
def test_pin_empty():
    t = torch.tensor([])
    assert not t.is_pinned()

    # Empty tensors will not be pinned or unpinned. It's a no-op.
    # This is also the default behavior in PyTorch.
    # We just check that it won't raise an error.
    nd = dgl.utils.pin_memory_inplace(t)
    assert not t.is_pinned()


if __name__ == "__main__":
    test_pin_noncontiguous()
    test_pin_view()
    test_unpin_automatically()
    test_pin_unpin_column()
