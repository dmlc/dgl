from itertools import product
import sys

from ..backend import gsddmm
from .utils import _notes_docstring

__all__ = ['gsddmm', 'copy_u', 'copy_v']


def _gen_sddmm_func(lhs_target, rhs_target, binary_op):
    name = "{}_{}_{}".format(lhs_target, binary_op, rhs_target)
    target_dict = {
        'u': "source node",
        'e': "edge",
        'v': "destination node"
    }
    lhs_str = target_dict[lhs_target]
    rhs_str = target_dict[rhs_target]
    docstring = r"""Generalized SDDMM function.
    It computes edge features by {} {} features and {} features.

    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph
    x : tensor
        The {} features.
    y : tensor
        The {} features.

    Returns
    -------
    tensor
        The result tensor.
    {}""".format(binary_op, lhs_str, rhs_str,
                 lhs_str, rhs_str,
                 _notes_docstring)

    def func(g, x, y):
        return gsddmm(g, binary_op, x, y,
                      lhs_target=lhs_target, rhs_target=rhs_target)
    func.__name__ = name
    func.__doc__ = docstring
    return func

def _register_sddmm_func():
    """Register sddmm functions"""
    target = ["u", "v", "e"]
    for lhs, rhs in product(target, target):
        if lhs != rhs:
            for binary_op in ["add", "sub", "mul", "div", "dot"]:
                func = _gen_sddmm_func(lhs, rhs, binary_op)
                setattr(sys.modules[__name__], func.__name__, func)
                __all__.append(func.__name__)

def copy_u(g, x):
    r"""Generalized SDDMM function that copies source node features to edges.

    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph.
    x : tensor
        The source node features.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient).
    """
    return gsddmm(g, 'copy_lhs', x, None)

def copy_v(g, x):
    r"""Generalized SDDMM function that copies destination node features to edges.

    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph.
    x : tensor
        The destination node features.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient).
    """
    return gsddmm(g, 'copy_rhs', None, x)

_register_sddmm_func()
