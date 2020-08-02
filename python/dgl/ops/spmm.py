"""dgl spmm operator module."""
import sys
from ..backend import gspmm as gspmm_internal

__all__ = ['gspmm']


def gspmm(g, op, reduce_op, lhs_data, rhs_data):
    r""" Generalized Sparse Matrix Multiplication interface.
    It fuses two steps into one kernel.

    1. Computes messages by :attr:`op` source node and edge features.
    2. Aggregate the messages by :attr:`reduce_op` as the features on destination nodes.

    .. math::
        x_v = \psi_{(u, v, e)\in \mathcal{G}}(\rho(x_u, x_e))

    where :math:`x_v` is the returned feature on destination nodes, and :math:`x_u`,
    :math:`x_e` refers to :attr:`u`, :attr:`e` respectively. :math:`\rho` means binary
    operator :attr:`op` and :math:`\psi` means reduce operator :attr:`reduce_op`,
    :math:`\mathcal{G}` is the graph we apply gspmm on: :attr:`g`.

    Note that this function does not handle gradients.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    op : str
        The binary op's name, could be ``add``, ``sub``, ``mul``, ``div``,
        ``copy_lhs``, ``copy_rhs``.
    reduce_op : str
        Reduce operator, could be ``sum``, ``max``, ``min``.
    lhs_data : tensor or None
        The left operand, could be None if it's not required by the op.
    rhs_data : tensor or None
        The right operand, could be None if it's not required by the op.

    Returns
    -------
    tensor
        The result tensor.
    """
    return gspmm_internal(g._graph, op, reduce_op, lhs_data, rhs_data)


def _gen_spmm_func(binary_op, reduce_op):
    name = "u_{}_e_{}".format(binary_op, reduce_op)
    docstring = """Generalized SpMM function.
    It fuses two steps into one kernel.

    1. Computes messages by {} source node and edge features.
    2. Aggregate the messages by {} as the features on destination nodes.

    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph
    x : tensor
        The source node features.
    y : tensor
        The edge features.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient). If the
    feature shape of two input operands do not match, we first broadcasts the features to a unified
    shape (note that the memory usage will not increase accordingly) and then performs the operation.

    Broadcasting follows NumPy semantics. Please see
    https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    for more details about the NumPy broadcasting semantics.
    """.format(binary_op, reduce_op)

    def func(g, x, y):
        return gspmm(g, binary_op, reduce_op, x, y)
    func.__name__ = name
    func.__doc__ = docstring
    return func


def _gen_copy_reduce_func(binary_op, reduce_op):

    name = "{}_{}".format(binary_op, reduce_op)
    binary_str = {
        "copy_u": "It copies node feature to edge as the message.",
        'copy_e': "It regards edge feature as message."
    }
    x_str = {
        "copy_u": "source node",
        "copy_e": "edge"
    }
    docstring = lambda binary_op: """Generalized SpMM function. {}
    Then aggregates the message by {} on destination nodes.

    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph
    x : tensor
        The {} features.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient).
    """.format(
        binary_str[binary_op],
        reduce_op,
        x_str[binary_op])

    def func(g, x):
        if binary_op == 'copy_u':
            return gspmm(g, 'copy_lhs', reduce_op, x, None)
        else:
            return gspmm(g, 'copy_rhs', reduce_op, None, x)

    func.__name__ = name
    func.__doc__ = docstring(binary_op)
    return func


def _register_spmm_func():
    """Register spmm functions"""
    for binary_op in ["add", "sub", "mul", "div", "copy_u", "copy_e"]:
        for reduce_op in ["sum", "max", "min"]:
            if binary_op.startswith("copy"):
                func = _gen_copy_reduce_func(binary_op, reduce_op)
            else:
                func = _gen_spmm_func(binary_op, reduce_op)
            setattr(sys.modules[__name__], func.__name__, func)
            __all__.append(func.__name__)


_register_spmm_func()
