"""Module for dgl kernels for graph computation."""
from __future__ import absolute_import

from .._ffi.function import _init_api
from .. import ndarray as nd

# pylint: disable=invalid-name
def infer_binary_feature_shape(op, lhs, rhs):
    """Infer the output feature shape after a binary operation between lhs and rhs.

    Parameter
    ---------
    op : string
        The binary_op name.
    lhs : dgl.ndarray.NDArray
        The lhs tensor.
    rhs : dgl.ndarray.NDArray
        The rhs tensor.

    Returns
    -------
    tuple of int
        The output feature shape.
    """
    ret = _CAPI_DGLKernelInferBinaryFeatureShape(op, lhs, rhs)
    return tuple(ret.asnumpy())

# pylint: disable=invalid-name
def binary_op_reduce(reducer, op, G, A_target, B_target, A, B, out,
                     A_rows=None, B_rows=None, out_rows=None):
    """Perform binary operation on the edges of graph ``G``, and optionally
    reduce the per-edge result by edge destinations into per-node result.

    Details
    -------
    Concretely, this function could be decomposed into two steps:

    1. Perform binary operations on each edge (u, v, e) on graph ``G`` as
       follows,::

           C[e] = A[select_A_target(u, v, e)] op B[select_B_target(u, v, e)]

       where

       * ``select_A_target`` and ``select_B_target`` would return the source
         node ID, destination node ID, or edge ID, according to ``A_target``
         and ``B_target`` which could take either

         - "source" (dgl.function.TargetCode.SRC),
         - "destination" (dgl.function.TargetCode.DST), or
         - "edge" (dgl.function.TargetCode.EDGE).

       * ``A`` and ``B`` are data tensors.  If ``A_target`` is "edge", then
         ``A.shape[0]`` should equal the number of edges of ``G``. Otherwise
         that should equal the number of nodes of ``G``.  Similar constraints
         apply for ``B``.

       * ``op`` could be either of the following strings: "add", "mul", "sub",
         "div".

    2. Perform the optional reduction step on ``C`` computed previously.

       * If ``reducer`` is None, then no reduction is performed and we return
         the per-edge result ``C`` directly,::

             out[e] = C[e]

       * Otherwise, the per-edge result ``C`` is reduced into per-node result
         according to edge destinations, in a similar fashion as
         ``unsorted_segment_XXX`` in Tensorflow or ``scatter_XXX`` in PyTorch
         or PyTorch-Scatter.  For all ``v`` that has incoming edges,::

             out[v] = reducer_{e: (u, v, e) in G} C[e]

    Broadcasting
    ------------
    Broadcasting is supported on the feature dimensions, following numpy
    semantics.

    Examples::

        A.shape = (N, D1, D2)  # N is the number of nodes
        B.shape = (M, D1, 1)   # M is the number of edges
        C = BinaryOpReduce("sum", "add", graph, A, B, ...)
        C.shape = (N, D1, D2)

    Partial reads/writes
    --------------------
    Optionally, one can provide which rows to read from ``A`` and ``B`` with
    ``A_rows`` and ``B_rows``, both of which are 1D integer arrays.  Similarly,
    one can provide which rows to write to ``out`` with ``out_rows``, which is
    again a 1D integer array.  Concretely,

    * Instead of from ``A`` and ``B``, ``C`` would be computed from
      ``A[A_rows]`` and ``B[B_rows]``.  This implies that

      * ``A`` and ``B`` no longer need to have the same number of rows as
        the number of nodes or edges in ``G``.  However, ``A_rows`` and
        ``B_rows`` must have the same number of elements as the number of
        nodes or edges in ``G``.

    * Instead of directly writing to ``out``, it will selectively write some
      rows of ``C`` or reduced ``C``,::

          out[out_rows[i]] = C[i]     if out_rows[i] != -1

      Or

          out[out_rows[i]] = reducer_{e: (u, v, e) in G} C[e]

    Parameters
    ----------
    reducer : str
        The type of the reducer ("sum", "max", "min", "mean", "prod", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    op : str
        The type of the binary functor ("add", "mul", "sub", "div").
    G : GraphIndex
        The graph
    A_target : int
        Choice of source, destination, or edge ID for edges on left operand
    B_target : int
        Choice of source, destination, or edge ID for edges on right operand
    A : NDArray
        Data tensor of left operand
    B : NDArray
        Data tensor of right operand
    out : NDArray (output)
        Output tensor.  The result will be written there in place.
    A_rows : NDArray, optional
        The rows to read from A.
    B_rows : NDArray, optional
        The rows to read from B.
    out_rows : NDArray
        The rows to write to output tensor.
    """
    if A_rows is None:
        A_rows = nd.NULL[G.dtype]
    if B_rows is None:
        B_rows = nd.NULL[G.dtype]
    if out_rows is None:
        out_rows = nd.NULL[G.dtype]
    _CAPI_DGLKernelBinaryOpReduce(
        reducer, op, G,
        int(A_target), int(B_target),
        A, B, out,
        A_rows, B_rows, out_rows)

# pylint: disable=invalid-name
def backward_lhs_binary_op_reduce(
        reducer, op, G,
        A_target, B_target,
        A, B, out,
        grad_out, grad_A,
        A_rows=None, B_rows=None, out_rows=None):
    """Compute the gradient of ``binary_op_reduce`` w.r.t. ``A`` and store it
    in ``grad_A``.

    See ``binary_op_reduce`` for forward propagation and partial reads/writes.

    Gradient of broadcasted tensors
    -------------------------------
    ``grad_A`` is assumed to be unbroadcasted, i.e. the shape of ``grad_A``
    is the same as ``grad_out`` except the first axis.

    If broadcasting happened in forward propagation, one needs to manually
    sum the gradients along the broadcasted dimension to yield the correct
    gradient.

    Parameter
    ---------
    reducer : str
        The type of the reducer ("sum", "max", "min", "mean", "prod", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    op : str
        The type of the binary functor ("add", "mul", "sub", "div").
    G : GraphIndex
        The graph
    A_target : int
        Choice of source, destination, or edge ID for edges on left operand
    B_target : int
        Choice of source, destination, or edge ID for edges on right operand
    A : NDArray
        Data tensor of left operand
    B : NDArray
        Data tensor of right operand
    out : NDArray
        Output tensor computed in the forward pass.
    grad_out : NDArray
        Gradient w.r.t. ``out``.
    grad_A : NDArray (output)
        Gradient w.r.t. ``A``.  The result will be written there in place.
    A_rows : NDArray, optional
        The rows read from A.
    B_rows : NDArray, optional
        The rows read from B.
    out_rows : NDArray
        The rows written to output tensor.
    """
    if A_rows is None:
        A_rows = nd.NULL[G.dtype]
    if B_rows is None:
        B_rows = nd.NULL[G.dtype]
    if out_rows is None:
        out_rows = nd.NULL[G.dtype]
    _CAPI_DGLKernelBackwardLhsBinaryOpReduce(
        reducer, op, G,
        int(A_target), int(B_target),
        A_rows, B_rows, out_rows,
        A, B, out,
        grad_out, grad_A)

# pylint: disable=invalid-name
def backward_rhs_binary_op_reduce(
        reducer, op, G,
        A_target, B_target,
        A, B, out,
        grad_out, grad_B,
        A_rows=None, B_rows=None, out_rows=None):
    """Compute the gradient of ``binary_op_reduce`` w.r.t. ``B`` and store it
    in ``grad_B``.

    See ``binary_op_reduce`` for forward propagation and partial reads/writes.

    Gradient of broadcasted tensors
    -------------------------------
    ``grad_B`` is assumed to be unbroadcasted, i.e. the shape of ``grad_B``
    is the same as ``grad_out`` except the first axis.

    If broadcasting happened in forward propagation, one needs to manually
    sum the gradients along the broadcasted dimension to yield the correct
    gradient.

    Parameter
    ---------
    reducer : str
        The type of the reducer ("sum", "max", "min", "mean", "prod", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    op : str
        The type of the binary functor ("add", "mul", "sub", "div").
    G : GraphIndex
        The graph
    A_target : int
        Choice of source, destination, or edge ID for edges on left operand
    B_target : int
        Choice of source, destination, or edge ID for edges on right operand
    A : NDArray
        Data tensor of left operand
    B : NDArray
        Data tensor of right operand
    out : NDArray
        Output tensor computed in the forward pass.
    grad_out : NDArray
        Gradient w.r.t. ``out``.
    grad_B : NDArray (output)
        Gradient w.r.t. ``B``.  The result will be written there in place.
    A_rows : NDArray, optional
        The rows read from A.
    B_rows : NDArray, optional
        The rows read from B.
    out_rows : NDArray
        The rows written to output tensor.
    """
    if A_rows is None:
        A_rows = nd.NULL[G.dtype]
    if B_rows is None:
        B_rows = nd.NULL[G.dtype]
    if out_rows is None:
        out_rows = nd.NULL[G.dtype]
    _CAPI_DGLKernelBackwardRhsBinaryOpReduce(
        reducer, op, G,
        int(A_target), int(B_target),
        A_rows, B_rows, out_rows,
        A, B, out,
        grad_out, grad_B)

# pylint: disable=invalid-name
def copy_reduce(reducer, G, target,
                X, out,
                X_rows=None, out_rows=None):
    """Copy data in ``X`` according to source/destination/edge ID onto the
    edges of graph ``G``, and optionally reduce the per-edge result by edge
    destinations into per-node result.

    Details
    -------
    Concretely, this function could be decomposed into two steps:

    1. For each edge (u, v, e) on graph ``G``, set

           C[e] = X[select_target(u, v, e)]

       where

       * ``select_target`` would return the source node ID, destination node,
         ID, or edge ID, according to ``target`` which could take either

         - "source" (dgl.function.TargetCode.SRC),
         - "destination" (dgl.function.TargetCode.DST), or
         - "edge" (dgl.function.TargetCode.EDGE).

       * ``X`` is a data tensor.  If ``target`` is "edge", then ``X.shape[0]``
         should equal the number of edges of ``G``.  Otherwise that should
         equal the number of nodes of ``G``.

    2. Perform the optional reduction step on ``C`` computed previously.

       * If ``reducer`` is None, then no reduction is performed and we return
         the per-edge result ``C`` directly,::

             out[e] = C[e]

       * Otherwise, the per-edge result ``C`` is reduced into per-node result
         according to edge destinations, in a similar fashion as
         ``unsorted_segment_XXX`` in Tensorflow or ``scatter_XXX`` in PyTorch
         or PyTorch-Scatter.  For all ``v`` that has incoming edges,::

             out[v] = reducer_{e: (u, v, e) in G} C[e]

    Partial reads/writes
    --------------------
    Optionally, one can provide which rows to read from ``X`` with ``X_rows``,
    which is a 1D integer array.  Similarly, one can provide which rows to
    write to ``out`` with ``out_rows``, which is again a 1D integer array.
    Concretely,

    * Instead of from ``X``, ``C`` would be copied from ``X[X_rows]``.  This
      implies that

      * ``X`` no longer needs to have the same number of rows as the number of
        nodes or edges in ``G``.  However, ``X_rows`` must have the same
        number of elements as the number of nodes or edges in ``G``.

    * Instead of directly writing to ``out``, it will selectively write some
      rows of ``C`` or reduced ``C``,::

          out[out_rows[i]] = C[i]     if out_rows[i] != -1

      Or

          out[out_rows[i]] = reducer_{e: (u, v, e) in G} C[e]

    Parameter
    ---------
    reducer : str
        The type of the reducer ("sum", "max", "min", "mean", "prod", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    graph : GraphIndex
        The graph
    target : int
        Choice of source, destination, or edge ID for edges to index in data
        tensor.
    X : NDArray
        Data tensor.
    out : NDArray (output)
        Output tensor.  The result will be written there in place.
    X_rows : NDArray, optional
        The rows to read from X.
    out_mapping : NDArray
        The rows to write to output tensor.
    """
    if X_rows is None:
        X_rows = nd.NULL[G.dtype]
    if out_rows is None:
        out_rows = nd.NULL[G.dtype]
    _CAPI_DGLKernelCopyReduce(
        reducer, G, int(target),
        X, out, X_rows, out_rows)

# pylint: disable=invalid-name
def backward_copy_reduce(reducer, G, target,
                         X, out,
                         grad_out, grad_X,
                         X_rows=None, out_rows=None):
    """Compute the gradient of ``copy_reduce`` w.r.t. ``X`` and store it in
    ``grad_X``.

    See ``copy_reduce`` for forward propagation and partial reads/writes.

    Parameter
    ---------
    reducer : str
        The type of the reducer ("sum", "max", "min", "mean", "prod", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    G : GraphIndex
        The graph
    target : int
        Choice of source, destination, or edge ID for edges to index in data
        tensor.
    X : NDArray
        Data tensor.
    out : NDArray
        Output tensor computed in the forward pass.
    grad_out_data : NDArray
        Gradient w.r.t. ``out``.
    grad_X : NDArray (output)
        Gradient w.r.t. ``X``.  The result will be written there in place.
    X_rows : NDArray, optional
        The rows read from X.
    out_rows : NDArray
        The rows written to output tensor.
    """
    if X_rows is None:
        X_rows = nd.NULL[G.dtype]
    if out_rows is None:
        out_rows = nd.NULL[G.dtype]
    _CAPI_DGLKernelBackwardCopyReduce(
        reducer, G, int(target),
        X, out, grad_out, grad_X,
        X_rows, out_rows)

_init_api("dgl._deprecate.kernel")
