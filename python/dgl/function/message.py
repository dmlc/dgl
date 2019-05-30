"""Built-in message function."""
from __future__ import absolute_import

from .base import BuiltinFunction, TargetCode
from ..runtime import ir
from ..runtime.ir import var

__all__ = ["src_mul_edge", "src_mul_dst", "copy_src", "copy_edge"]


class MessageFunction(BuiltinFunction):
    """Base builtin message function class."""

    def __call__(self, graph, src_frame, dst_frame, edge_frame, out_size,
                 src_map, dst_map, edge_map, out_map, reducer="none"):
        """Symbolic computation of this builtin function to create
        runtime.executor
        """
        raise NotImplementedError

    @property
    def name(self):
        """Return the name of this builtin function."""
        raise NotImplementedError


class BinaryMessageFunction(MessageFunction):
    """Class for the lhs_op_rhs builtin message function.

    See Also
    --------
    src_mul_edge
    """
    def __init__(self, binary_op, lhs, rhs, lhs_field, rhs_field, out_field):
        self.binary_op = binary_op
        self.lhs = lhs
        self.rhs = rhs
        self.lhs_field = lhs_field
        self.rhs_field = rhs_field
        self.out_field = out_field

    def __call__(self, graph, src_frame, dst_frame, edge_frame, out_size,
                 src_map, dst_map, edge_map, out_map, reducer="none"):
        """Symbolic computation of this builtin function to create
        runtime.executor
        """
        graph = var.GRAPH(graph)
        in_frames = (src_frame, dst_frame, edge_frame)
        in_maps = (src_map, dst_map, edge_map)
        lhs_data = ir.READ_COL(in_frames[self.lhs], var.STR(self.lhs_field))
        rhs_data = ir.READ_COL(in_frames[self.rhs], var.STR(self.rhs_field))
        lhs_map = var.MAP(in_maps[self.lhs])
        rhs_map = var.MAP(in_maps[self.rhs])
        out_map = var.MAP(out_map)
        return ir.BINARY_REDUCE(reducer, self.binary_op, graph, self.lhs,
                                self.rhs, lhs_data, rhs_data, out_size,
                                lhs_map, rhs_map, out_map)

    @property
    def name(self):
        lhs = TargetCode.CODE2STR[self.lhs]
        rhs = TargetCode.CODE2STR[self.rhs]
        return "{}_{}_{}".format(lhs, self.binary_op, rhs)


class CopyMessageFunction(MessageFunction):
    """Class for the copy_src builtin message function.

    See Also
    --------
    copy_src
    """
    def __init__(self, target, in_field, out_field):
        self.target = target
        self.in_field = in_field
        self.out_field = out_field

    def __call__(self, graph, src_frame, dst_frame, edge_frame, out_size,
                 src_map, dst_map, edge_map, out_map, reducer="none"):
        """Symbolic computation of this builtin function to create
        runtime.executor
        """
        graph = var.GRAPH(graph)
        in_frames = (src_frame, dst_frame, edge_frame)
        in_maps = (src_map, dst_map, edge_map)
        in_data = ir.READ_COL(in_frames[self.target], var.STR(self.in_field))
        in_map = var.MAP(in_maps[self.target])
        out_map = var.MAP(out_map)
        return ir.COPY_REDUCE(reducer, graph, self.target, in_data, out_size,
                              in_map, out_map)

    @property
    def name(self):
        return "copy_{}".format(TargetCode.CODE2STR[self.target])


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
    return BinaryMessageFunction(
        "mul", TargetCode.SRC, TargetCode.EDGE, src, edge, out)


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
    return BinaryMessageFunction(
        "mul", TargetCode.SRC, TargetCode.DST, src, dst, out)


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
    return CopyMessageFunction(TargetCode.SRC, src, out)


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
    return CopyMessageFunction(TargetCode.EDGE, edge, out)
