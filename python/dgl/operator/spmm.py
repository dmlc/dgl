from ..backend import gspmm
from .utils import _notes_docstring
import sys

__all__ = ['gspmm']


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
    {}""".format(binary_op, reduce_op,
                 _notes_docstring)

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
        x_str[binary_op],
        _notes_docstring)

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
