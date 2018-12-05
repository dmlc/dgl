"""Built-in message function."""
from __future__ import absolute_import

from .base import BuiltinFunction
import operator
import dgl.backend as F

__all__ = ["src_mul_edge", "copy_src", "copy_edge"]


class MessageFunction(BuiltinFunction):
    """Base builtin message function class."""

    def __call__(self, edges):
        """Regular computation of this builtin.

        This will be used when optimization is not available.
        """
        raise NotImplementedError

    @property
    def name(self):
        """Return the name of this builtin function."""
        raise NotImplementedError

    def is_spmv_supported(self, g):
        """Return whether the SPMV optimization is supported."""
        raise NotImplementedError

    @property
    def use_edge_feature(self):
        raise NotImplementedError


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
        return _is_spmv_supported_edge_feat(g, self.edge_field)

    def __call__(self, edges):
        sdata = edges.src[self.src_field]
        edata = edges.data[self.edge_field]
        # Due to the different broadcasting semantics of different backends,
        #   we need to broadcast the sdata and edata to be of the same rank.
        rank = max(F.ndim(sdata), F.ndim(edata))
        sshape = F.shape(sdata)
        eshape = F.shape(edata)
        sdata = F.reshape(sdata, sshape + (1,) * (rank - F.ndim(sdata)))
        edata = F.reshape(edata, eshape + (1,) * (rank - F.ndim(edata)))
        ret = self.mul_op(sdata, edata)
        return {self.out_field : ret}

    @property
    def name(self):
        return "src_mul_edge"

    @property
    def use_edge_feature(self):
        return True

class CopySrcMessageFunction(MessageFunction):
    def __init__(self, src_field, out_field):
        self.src_field = src_field
        self.out_field = out_field

    def is_spmv_supported(self, g):
        return True

    def __call__(self, edges):
        return {self.out_field : edges.src[self.src_field]}

    @property
    def name(self):
        return "copy_src"

    @property
    def use_edge_feature(self):
        return False

class CopyEdgeMessageFunction(MessageFunction):
    def __init__(self, edge_field=None, out_field=None):
        self.edge_field = edge_field
        self.out_field = out_field

    def is_spmv_supported(self, g):
        # TODO: support this with e2v spmv
        return False
        # return _is_spmv_supported_edge_feat(g, self.edge_field)

    def __call__(self, edges):
        return {self.out_field : edges.data[self.edge_field]}

    @property
    def name(self):
        return "copy_edge"

    @property
    def use_edge_feature(self):
        return True


def src_mul_edge(src, edge, out):
    """Builtin message function that computes message by multiplying source
    node features with edge features.

    Parameters
    ----------
    src : str
        The source feature field.
    edge : str
        The edge feature field.
    out : str
        The output message field.

    Examples
    --------
    >>> import dgl
    >>> message_func = dgl.function.src_mul_edge(src='h', edge='w', out='m')

    The above example is equivalent to the following user defined function:

    >>> def message_func(edges):
    >>>   return {'m': edges.src['h'] * edges.data['w']}
    """
    return SrcMulEdgeMessageFunction(operator.mul, src, edge, out)

def copy_src(src, out):
    """Builtin message function that computes message using source node feature.

    Parameters
    ----------
    src : str
        The source feature field.
    out : str
        The output message field.

    Examples
    --------
    >>> import dgl
    >>> message_func = dgl.function.copy_src(src='h', out='m')

    The above example is equivalent to the following user defined function:

    >>> def message_func(edges):
    >>>     return {'m': edges.src['h']}
    """
    return CopySrcMessageFunction(src, out)

def copy_edge(edge, out):
    """Builtin message function that computes message using edge feature.

    Parameters
    ----------
    edge : str
        The edge feature field.
    out : str
        The output message field.

    Examples
    --------
    >>> import dgl
    >>> message_func = dgl.function.copy_edge(edge='h', out='m')

    The above example is equivalent to the following user defined function:

    >>> def message_func(edges):
    >>>     return {'m': edges.data['h']}
    """
    return CopyEdgeMessageFunction(edge, out)
