"""Operators for computing edge data."""
import sys

from .. import ops

__all__ = ["copy_u", "copy_v"]

#######################################################
# Edge-wise operators that fetch node data to edges
#######################################################


def copy_u(g, x_node, etype=None):
    """Compute new edge data by fetching from source node data.

    Given an input graph :math:`G(V, E)` (or a unidirectional bipartite graph
    :math:`G(V_{src}, V_{dst}, E)`) and an input tensor :math:`X`,
    the operator computes a tensor :math:`Y` storing the new edge data.
    For each edge :math:`e=(u,v) \\in E`, it computes:

    .. math:

        Y_e = X_u

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    x_node : Tensor
        The tensor storing the source node data. Shape :math:`(|V_{src}|, *)`.
    etype : str or (str, str, str), optional
        Edge type. If not specified, the input graph must have only one type of
        edges.

    Returns
    -------
    Tensor
        The tensor storing the new edge data. Shape :math:`(|E|, *)`.

    Examples
    --------

    **Homogeneous graph**

    >>> import torch, dgl
    >>> g = dgl.rand_graph(100, 500)  # a random graph of 100 nodes, 500 edges
    >>> x = torch.randn(g.num_nodes(), 5)  # 5 features
    >>> y = dgl.copy_u(g, x)
    >>> print(y.shape)
    (500, 5)

    **Heterogeneous graph**

    >>> hg = dgl.heterograph({
    ...     ('user', 'follow', 'user'): ([0, 1, 2], [2, 3, 4]),
    ...     ('user', 'like', 'movie'): ([3, 3, 1, 2], [0, 0, 1, 1])
    ... })
    >>> x = torch.randn(hg.num_nodes('user'), 5)
    >>> y = dgl.copy_u(hg, x, etype='like')
    >>> print(y.shape)
    (4, 5)
    """
    etype_subg = g if etype is None else g[etype]
    return ops.gsddmm(etype_subg, "copy_lhs", x_node, None)


def copy_v(g, x_node, etype=None):
    """Compute new edge data by fetching from destination node data.

    Given an input graph :math:`G(V, E)` (or a unidirectional bipartite graph
    :math:`G(V_{src}, V_{dst}, E)`) and an input tensor :math:`X`,
    the operator computes a tensor :math:`Y` storing the new edge data.
    For each edge :math:`e=(u,v) \\in E`, it computes:

    .. math:

        Y_e = X_v

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    x_node : Tensor
        The tensor storing the destination node data. Shape :math:`(|V_{dst}|, *)`.
    etype : str or (str, str, str), optional
        Edge type. If not specified, the input graph must have
        only one type of edges.

    Returns
    -------
    Tensor
        The tensor storing the new edge data. Shape :math:`(|E|, *)`.

    Examples
    --------

    **Homogeneous graph**

    >>> import torch, dgl
    >>> g = dgl.rand_graph(100, 500)  # a random graph of 100 nodes, 500 edges
    >>> x = torch.randn(g.num_nodes(), 5)  # 5 features
    >>> y = dgl.copy_v(g, x)
    >>> print(y.shape)
    (500, 5)

    **Heterogeneous graph**

    >>> hg = dgl.heterograph({
    ...     ('user', 'follow', 'user'): ([0, 1, 2], [2, 3, 4]),
    ...     ('user', 'like', 'movie'): ([3, 3, 1, 2], [0, 0, 1, 1])
    ... })
    >>> x = torch.randn(hg.num_nodes('movie'), 5)
    >>> y = dgl.copy_v(hg, x, etype='like')
    >>> print(y.shape)
    (4, 5)
    """
    etype_subg = g if etype is None else g[etype]
    return ops.gsddmm(etype_subg, "copy_rhs", None, x_node)


#######################################################
# Binary edge-wise operators
#######################################################


def _gen_u_op_v(op):
    """Internal helper function to create binary edge-wise operators.

    The function will return a Python function with:

     - Name: u_{op}_v
     - Docstring template

    Parameters
    ----------
    op : str
        Binary operator name. Must be 'add', 'sub', 'mul', 'div' or 'dot'.
    """
    name = f"u_{op}_v"
    op_verb = {
        "add": "adding",
        "sub": "subtracting",
        "mul": "multiplying",
        "div": "dividing",
        "dot": "dot-product",
    }
    docstring = f"""Compute new edge data by {op_verb[op]} the source node data
and destination node data.

Given an input graph :math:`G(V, E)` (or a unidirectional bipartite graph
:math:`G(V_{{src}}, V_{{dst}}, E)`) and two input tensors :math:`X` and
:math:`Y`, the operator computes a tensor :math:`Z` storing the new edge data.
For each edge :math:`e=(u,v) \\in E`, it computes:

.. math:

    Z_e = {op}(X_u, Y_v)

If :math:`X_u` and :math:`Y_v` are vectors or high-dimensional tensors, the
operation is element-wise and supports shape broadcasting. Read more about
`NumPy's broadcasting semantics
<https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_.

Parameters
----------
g : DGLGraph
    The input graph.
x_node : Tensor
    The tensor storing the source node data. Shape :math:`(|V_{{src}}|, *)`.
y_node : Tensor
    The tensor storing the destination node data. Shape :math:`(|V_{{dst}}|, *)`.
etype : str or (str, str, str), optional
    Edge type. If not specified, the input graph must have
    only one type of edges.

Returns
-------
Tensor
    The tensor storing the new edge data. Shape :math:`(|E|, *)`.

Examples
--------

**Homogeneous graph**

>>> import torch, dgl
>>> g = dgl.rand_graph(100, 500)  # a random graph of 100 nodes, 500 edges
>>> x = torch.randn(g.num_nodes(), 5)  # 5 features
>>> y = torch.randn(g.num_nodes(), 5)  # 5 features
>>> z = dgl.{name}(g, x, y)
>>> print(z.shape)
(500, 5)

**Heterogeneous graph**

>>> hg = dgl.heterograph({{
...     ('user', 'follow', 'user'): ([0, 1, 2], [2, 3, 4]),
...     ('user', 'like', 'movie'): ([3, 3, 1, 2], [0, 0, 1, 1])
... }})
>>> x = torch.randn(hg.num_nodes('user'), 5)
>>> y = torch.randn(hg.num_nodes('user'), 5)
>>> z = dgl.{name}(hg, x, y, etype='follow')
>>> print(z.shape)
(3, 5)

**Shape broadcasting**

>>> x = torch.randn(g.num_nodes(), 5)  # 5 features
>>> y = torch.randn(g.num_nodes(), 1)  # one feature
>>> z = dgl.{name}(g, x, y)
>>> print(z.shape)
(500, 5)
"""

    def func(g, x_node, y_node, etype=None):
        etype_subg = g if etype is None else g[etype]
        return ops.gsddmm(
            etype_subg, op, x_node, y_node, lhs_target="u", rhs_target="v"
        )

    func.__name__ = name
    func.__doc__ = docstring
    return func


def _register_func(func):
    setattr(sys.modules[__name__], func.__name__, func)
    __all__.append(func.__name__)


_register_func(_gen_u_op_v("add"))
_register_func(_gen_u_op_v("sub"))
_register_func(_gen_u_op_v("mul"))
_register_func(_gen_u_op_v("div"))
_register_func(_gen_u_op_v("dot"))
