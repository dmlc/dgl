"""Built-in message function."""
from __future__ import absolute_import

import operator
import dgl.backend as F

__all__ = ["src_mul_edge", "copy_src", "copy_edge"]


class MessageFunction(object):
    """Base builtin message function class."""

    def __call__(self, src, edge):
        """Regular computation of this builtin.

        This will be used when optimization is not available.
        """
        raise NotImplementedError

    def name(self):
        """Return the name of this builtin function."""
        raise NotImplementedError

    def is_spmv_supported(self, g):
        """Return whether the SPMV optimization is supported."""
        raise NotImplementedError


class BundledMessageFunction(MessageFunction):
    def __init__(self, fn_list):
        if not isinstance(fn_list, (list, tuple)):
            fn_list = [fn_list]
        self.fn_list = fn_list

    def is_spmv_supported(self, g):
        for fn in self.fn_list:
            if not isinstance(fn, MessageFunction) or not fn.is_spmv_supported(g):
                return False
        return True

    def __call__(self, src, edge):
        ret = None
        for fn in self.fn_list:
            msg = fn(src, edge)
            if ret is None:
                ret = msg
            else:
                # ret and msg must be dict
                ret.update(msg)
        return ret

    def name(self):
        return "bundled"


def _is_spmv_supported_node_feat(g, field):
    """Return whether the node feature shape supports SPMV optimization.

    Only scalar and vector features are supported currently.
    """
    feat = g.get_n_repr()[field]
    shape = F.shape(feat)
    return len(shape) == 1 or len(shape) == 2

def _is_spmv_supported_edge_feat(g, field):
    """Return whether the edge feature shape supports SPMV optimization.

    Only scalar feature is supported currently.
    """
    feat = g.get_e_repr()[field]
    shape = F.shape(feat)
    return len(shape) == 1 or (len(shape) == 2 and shape[1] == 1)


class SrcMulEdgeMessageFunction(MessageFunction):
    def __init__(self, mul_op, src_field, edge_field, out_field):
        self.mul_op = mul_op
        self.src_field = src_field
        self.edge_field = edge_field
        self.out_field = out_field

    def is_spmv_supported(self, g):
        return _is_spmv_supported_node_feat(g, self.src_field) \
                and _is_spmv_supported_edge_feat(g, self.edge_field)

    def __call__(self, src, edge):
        ret = self.mul_op(src[self.src_field], edge[self.edge_field])
        return {self.out_field : ret}

    def name(self):
        return "src_mul_edge"

class CopySrcMessageFunction(MessageFunction):
    def __init__(self, src_field, out_field):
        self.src_field = src_field
        self.out_field = out_field

    def is_spmv_supported(self, g):
        return _is_spmv_supported_node_feat(g, self.src_field)

    def __call__(self, src, edge):
        return {self.out_field : src[self.src_field]}

    def name(self):
        return "copy_src"

class CopyEdgeMessageFunction(MessageFunction):
    def __init__(self, edge_field=None, out_field=None):
        self.edge_field = edge_field
        self.out_field = out_field

    def is_spmv_supported(self, g):
        # TODO: support this with g-spmv
        return False
        # return _is_spmv_supported_edge_feat(g, self.edge_field)

    def __call__(self, src, edge):
        if self.edge_field is not None:
            ret = edge[self.edge_field]
        else:
            ret = edge
        if self.out_field is None:
            return ret
        else:
            return {self.out_field : ret}

    def name(self):
        return "copy_edge"


def src_mul_edge(src, edge, out):
    """Builtin message function that computes message by multiplying source node features
    with edge features.

    Parameters
    ----------
    src : str
        The source feature name.
    edge : str
        The edge feature name.
    out : str
        The output message name.
    """
    return SrcMulEdgeMessageFunction(operator.mul, src, edge, out)

def copy_src(src, out):
    """Builtin message function that computes message using source node feature.

    Parameters
    ----------
    src : str
        The source feature name.
    out : str
        The output message name.
    """
    return CopySrcMessageFunction(src, out)

def copy_edge(edge, out):
    """Builtin message function that computes message using edge feature.

    Parameters
    ----------
    edge : str
        The edge feature name.
    out : str
        The output message name.
    """
    return CopyEdgeMessageFunction(edge, out)
