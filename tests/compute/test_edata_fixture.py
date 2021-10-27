import dgl
from dgl.edata_fixture import use_edata_for_update
import dgl.function as fn
import dgl.nn
from dgl.data import CoraGraphDataset
import pytest

import backend as F


def test_edata_nnmodule():
    g = CoraGraphDataset()[0]
    g.edata['ee'] = F.attach_grad(F.randn((g.num_edges(),)))
    conv = dgl.nn.GraphConv(
        g.ndata["feat"].shape[-1], 4, norm='none', bias=True)
    with use_edata_for_update("ee"):
        out = conv(g, g.ndata["feat"])
    F.backward(F.reduce_sum(out))
    assert F.grad(g.edata['ee']) is not None


@pytest.mark.parametrize('reduce_func', [fn.sum('h', 'out'), fn.max('h', 'out')])
def test_edata_reduce_func(reduce_func):
    g = CoraGraphDataset()[0]
    with F.record_grad():
        g.edata['ee'] = F.attach_grad(F.randn((g.num_edges(),)))
        with use_edata_for_update("ee"):
            g.update_all(fn.copy_u('feat', 'h'), reduce_func)
        F.backward(F.reduce_sum(g.ndata['out']))
        assert F.grad(g.edata['ee']) is not None


@pytest.mark.parametrize('msg_func', [fn.copy_e('e', 'h')])
@pytest.mark.parametrize('reduce_func', [fn.sum('h', 'out'), fn.max('h', 'out')])
def test_edata_func_should_not_change(msg_func, reduce_func):
    g = CoraGraphDataset()[0]
    with F.record_grad():
        g.edata['ee'] = F.attach_grad(F.randn((g.num_edges(),)))
        g.edata['e'] = F.attach_grad(F.randn((g.num_edges(),)))
        with use_edata_for_update("ee"):
            g.update_all(msg_func, reduce_func)
        F.backward(F.reduce_sum(g.ndata['out']))
        assert F.grad(g.edata['ee']) is None


def test_edata_exit_ctx_manager():
    g = CoraGraphDataset()[0]
    with F.record_grad():
        g.edata['ee'] = F.attach_grad(F.randn((g.num_edges(),)))
        conv = dgl.nn.GraphConv(
            g.ndata["feat"].shape[-1], 4, norm='none', bias=True)
        with use_edata_for_update("ee"):
            out = conv(g, g.ndata["feat"])
        F.backward(F.reduce_sum(out))
        assert F.grad(g.edata['ee']) is not None

        # Exit context manager
        g.edata['ee'] = F.attach_grad(F.randn((g.num_edges(),)))
        out = conv(g, g.ndata["feat"])
        F.backward(F.reduce_sum(out))
        assert F.grad(g.edata['ee']) is None


if __name__ == "__main__":
    test_edata_nnmodule()
    test_edata_exit_ctx_manager()
