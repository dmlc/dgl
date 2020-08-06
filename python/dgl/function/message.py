"""Built-in message function."""
from __future__ import absolute_import

import sys
from itertools import product

from .base import BuiltinFunction, TargetCode
from .._deprecate.runtime import ir
from .._deprecate.runtime.ir import var


__all__ = ["src_mul_edge", "copy_src", "copy_edge", "copy_u", "copy_e",
           "BinaryMessageFunction", "CopyMessageFunction"]


class MessageFunction(BuiltinFunction):
    """Base builtin message function class."""

    def _invoke(self, graph, src_frame, dst_frame, edge_frame, out_size,
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

    def _invoke(self, graph, src_frame, dst_frame, edge_frame, out_size,
                src_map, dst_map, edge_map, out_map, reducer="none"):
        """Symbolic computation of builtin binary message function to create
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
    """Class for the copy builtin message function.

    See Also
    --------
    copy_src
    """
    def __init__(self, target, in_field, out_field):
        self.target = target
        self.in_field = in_field
        self.out_field = out_field

    def _invoke(self, graph, src_frame, dst_frame, edge_frame, out_size,
                src_map, dst_map, edge_map, out_map, reducer="none"):
        """Symbolic computation of builtin message function to create
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


def copy_u(u, out):
    """Builtin message function that computes message using source node
    feature.

    Parameters
    ----------
    u : str
        The source feature field.
    out : str
        The output message field.

    Examples
    --------
    >>> import dgl
    >>> message_func = dgl.function.copy_u('h', 'm')

    The above example is equivalent to the following user defined function:

    >>> def message_func(edges):
    >>>     return {'m': edges.src['h']}
    """
    return CopyMessageFunction(TargetCode.SRC, u, out)


def copy_e(e, out):
    """Builtin message function that computes message using edge feature.

    Parameters
    ----------
    e : str
        The edge feature field.
    out : str
        The output message field.

    Examples
    --------
    >>> import dgl
    >>> message_func = dgl.function.copy_e('h', 'm')

    The above example is equivalent to the following user defined function:

    >>> def message_func(edges):
    >>>     return {'m': edges.data['h']}
    """
    return CopyMessageFunction(TargetCode.EDGE, e, out)


###############################################################################
# Generate all following  builtin message functions:
# element-wise message functions:
# u_add_v, u_sub_v, u_mul_v, u_div_v
# u_add_e, u_sub_e, u_mul_e, u_div_e
# v_add_u, v_sub_u, v_mul_u, v_div_u
# v_add_e, v_sub_e, v_mul_e, v_div_e
# e_add_u, e_sub_u, e_mul_u, e_div_u
# e_add_v, e_sub_v, e_mul_v, e_div_v
#
# dot message functions:
# u_dot_v, u_dot_e, v_dot_e
# v_dot_u, e_dot_u, e_dot_v

_TARGET_MAP = {
    "u": TargetCode.SRC,
    "v": TargetCode.DST,
    "e": TargetCode.EDGE,
}


def _gen_message_builtin(lhs, rhs, binary_op):
    name = "{}_{}_{}".format(lhs, binary_op, rhs)
    docstring = """Builtin message function that computes a message on an edge
    by performing element-wise {} between features of {} and {}
    if the features have the same shape; otherwise, it first broadcasts the features
    to a new shape and performs the element-wise operation.

    Broadcasting follows NumPy semantics. Please see
    https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    for more details about the NumPy broadcasting semantics.

    Parameters
    ----------
    lhs_field : str
        The feature field of {}.
    rhs_field : str
        The feature field of {}.
    out : str
        The output message field.

    Examples
    --------
    >>> import dgl
    >>> message_func = dgl.function.{}('h', 'h', 'm')
    """.format(binary_op,
               TargetCode.CODE2STR[_TARGET_MAP[lhs]],
               TargetCode.CODE2STR[_TARGET_MAP[rhs]],
               TargetCode.CODE2STR[_TARGET_MAP[lhs]],
               TargetCode.CODE2STR[_TARGET_MAP[rhs]],
               name)

    def func(lhs_field, rhs_field, out):
        return BinaryMessageFunction(
            binary_op, _TARGET_MAP[lhs],
            _TARGET_MAP[rhs], lhs_field, rhs_field, out)
    func.__name__ = name
    func.__doc__ = docstring
    return func


def _register_builtin_message_func():
    """Register builtin message functions"""
    target = ["u", "v", "e"]
    for lhs, rhs in product(target, target):
        if lhs != rhs:
            for binary_op in ["add", "sub", "mul", "div", "dot"]:
                func = _gen_message_builtin(lhs, rhs, binary_op)
                setattr(sys.modules[__name__], func.__name__, func)
                __all__.append(func.__name__)

_register_builtin_message_func()


##############################################################################
# For backward compatibility

def src_mul_edge(src, edge, out):
    """Builtin message function that computes message by performing
    binary operation mul between src feature and edge feature.

    Notes
    -----
    This function is deprecated. Please use u_mul_e instead.

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
    >>> message_func = dgl.function.src_mul_edge('h', 'h', 'm')
    """
    return getattr(sys.modules[__name__], "u_mul_e")(src, edge, out)


def copy_src(src, out):
    """Builtin message function that computes message using source node
    feature.

    Notes
    -----
    This function is deprecated. Please use copy_u instead.

    Parameters
    ----------
    src : str
        The source feature field.
    out : str
        The output message field.

    Examples
    --------
    >>> import dgl
    >>> message_func = dgl.function.copy_src('h', 'm')

    The above example is equivalent to the following user defined function:

    >>> def message_func(edges):
    >>>     return {'m': edges.src['h']}
    """
    return copy_u(src, out)


def copy_edge(edge, out):
    """Builtin message function that computes message using edge feature.

    Notes
    -----
    This function is deprecated. Please use copy_e instead.

    Parameters
    ----------
    edge : str
        The edge feature field.
    out : str
        The output message field.

    Examples
    --------
    >>> import dgl
    >>> message_func = dgl.function.copy_edge('h', 'm')

    The above example is equivalent to the following user defined function:

    >>> def message_func(edges):
    >>>     return {'m': edges.data['h']}
    """
    return copy_e(edge, out)
