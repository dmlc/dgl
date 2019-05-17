"""Built-in message function."""
from __future__ import absolute_import

from .base import BuiltinFunction
from ..runtime import ir
from ..runtime.ir import var
from ..utils import create_empty_mapping as empty_map

__all__ = ["src_mul_edge", "src_mul_dst", "copy_src", "copy_edge"]


class MessageFunction(BuiltinFunction):
    """Base builtin message function class."""

    def __call__(self):
        """Symbolic computation of this builtin function to create
        runtime.executor
        """
        raise NotImplementedError

    @property
    def name(self):
        """Return the name of this builtin function."""
        raise NotImplementedError


class SrcOpEdgeMessageFunction(MessageFunction):
    """Class for the src_op_edge builtin message function.

    See Also
    --------
    src_mul_edge
    """
    def __init__(self, binary_op, src_field, edge_field, out_field):
        self.binary_op = binary_op
        self.src_field = src_field
        self.edge_field = edge_field
        self.out_field = out_field

    def __call__(self, spmat, src_frame, dst_frame, edge_frame, out_size,
                 reducer="none", src_map=empty_map, dst_map=empty_map,
                 edge_map=empty_map, out_map=empty_map):
        """Symbolic computation of this builtin function to create
        runtime.executor
        """
        src_map = var.MAP(src_map)
        edge_map = var.MAP(edge_map)
        out_map = var.MAP(out_map)
        src_data = ir.READ_COL(src_frame, var.STR(self.src_field))
        edge_data = ir.READ_COL(edge_frame, var.STR(self.edge_field))
        return ir.SRC_OP_EDGE_REDUCE(reducer, self.binary_op, spmat, src_data,
                                     edge_data, out_size, src_map, edge_map,
                                     out_map)

    @property
    def name(self):
        return "src_{}_edge".format(self.binary_op)


class SrcOpDstMessageFunction(MessageFunction):
    """Class for the src_op_dst builtin message function.

    See Also
    --------
    src_mul_dst
    """
    def __init__(self, binary_op, src_field, dst_field, out_field):
        self.binary_op = binary_op
        self.src_field = src_field
        self.dst_field = dst_field
        self.out_field = out_field

    def __call__(self, spmat, src_frame, dst_frame, edge_frame, out_size,
                 reducer="none", src_map=empty_map, dst_map=empty_map,
                 edge_map=empty_map, out_map=empty_map):
        """Symbolic computation of this builtin function to create
        runtime.executor
        """
        src_map = var.MAP(src_map)
        dst_map = var.MAP(dst_map)
        out_map = var.MAP(out_map)
        src_data = ir.READ_COL(src_frame, var.STR(self.src_field))
        dst_data = ir.READ_COL(src_frame, var.STR(self.dst_field))
        return ir.SRC_OP_DST_REDUCE(reducer, self.binary_op, spmat, src_data,
                                    dst_data, out_size, src_map, dst_map,
                                    out_map)

    @property
    def name(self):
        return "src_{}_dst".format(self.binary_op)


class CopySrcMessageFunction(MessageFunction):
    """Class for the copy_src builtin message function.

    See Also
    --------
    copy_src
    """
    def __init__(self, src_field, out_field):
        self.src_field = src_field
        self.out_field = out_field

    def __call__(self, spmat, src_frame, dst_frame, edge_frame, out_size,
                 reducer="none", src_map=empty_map, dst_map=empty_map,
                 edge_map=empty_map, out_map=empty_map):
        """Symbolic computation of this builtin function to create
        runtime.executor
        """
        src_map = var.MAP(src_map)
        out_map = var.MAP(out_map)
        src_data = ir.READ_COL(src_frame, var.STR(self.src_field))
        return ir.COPY_SRC_REDUCE(reducer, spmat, src_data, out_size,
                                  src_map, out_map)

    @property
    def name(self):
        return "copy_src"


class CopyEdgeMessageFunction(MessageFunction):
    """Class for the copy_edge builtin message function.

    See Also
    --------
    copy_edge
    """
    def __init__(self, edge_field=None, out_field=None):
        self.edge_field = edge_field
        self.out_field = out_field

    def __call__(self, spmat, src_frame, dst_frame, edge_frame, out_size,
                 reducer="none", src_map=empty_map, dst_map=empty_map,
                 edge_map=empty_map, out_map=empty_map):
        """Symbolic computation of this builtin function to create
        runtime.executor
        """
        edge_map = var.MAP(edge_map)
        out_map = var.MAP(out_map)
        edge_data = ir.READ_COL(edge_frame, var.STR(self.edge_field))
        return ir.COPY_EDGE_REDUCE(reducer, spmat, edge_data, out_size,
                                   edge_map, out_map)

    @property
    def name(self):
        return "copy_edge"


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
    return SrcOpEdgeMessageFunction("mul", src, edge, out)


def src_mul_dst(src, dst, out):
    """Builtin message function that computes message by multiplying source
    node features with destination node features.

    Parameters
    ----------
    src : str
        The source feature field.
    dst : str
        The destination feature field.
    out : str
        The output message field.

    Examples
    --------
    >>> import dgl
    >>> message_func = dgl.function.src_mul_dst(src='h1', dst='h2', out='m')

    The above example is equivalent to the following user defined function:

    >>> def message_func(edges):
    >>>   return {'m': edges.src['h1'] * edges.dst['h2']}
    """
    return SrcOpDstMessageFunction("mul", src, dst, out)


def copy_src(src, out):
    """Builtin message function that computes message using source node
    feature.

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
