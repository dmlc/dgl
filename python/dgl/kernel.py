"""Module for dgl kernels for graph computation."""
from __future__ import absolute_import

from ._ffi.function import _init_api
from .ndarray import empty

def infer_binary_feature_shape(lhs, rhs):
    """Infer the output feature shape after a binary operation between lhs and rhs.

    Parameter
    ---------
    lhs : dgl.ndarray.NDArray
        The lhs tensor.
    rhs : dgl.ndarray.NDArray
        The rhs tensor.

    Returns
    -------
    tuple of int
        The output feature shape.
    """
    ret = _CAPI_DGLKernelInferBinaryFeatureShape(lhs, rhs)
    return tuple(ret.asnumpy())

def binary_op_reduce(reducer, binary_op, graph, lhs, rhs, lhs_data, rhs_data,
                     out_data, lhs_mapping=None, rhs_mapping=None,
                     out_mapping=None):
    """Perform binary operation between the given data and reduce by the graph.

    If the reducer is one of "sum, "max, "min", "prod", the operator computes,
    for each node i,::

        out[i] = Sigma_{j in Neighbor(i)} ( A[s1(i, j, e)] op B[s2(i, j, e)] )

    , where A, B are two input feature tensors, op could be element-wise add/sub/div/mul.
    Depending on the lhs and rhs target, s1 and s2 will select the src/dst/edge
    ids of each neighbor.

    If the reducer is "none", the operator computes, for each edge e,::

        out[e] = A[s1(i, j, e)] op B[s2(i, j, e)]

    Here, the node/edge feature (e.g., A[i], B[e]) could be dense tensor. In such
    case, broadcasting is supported on the feature dimensions, which follows numpy
    semantics.

    Examples::

        A.shape = (N, D1, D2)  # N is the number of nodes
        B.shape = (M, D1, 1)   # M is the number of edges
        C = BinaryOpReduce("sum", "add", graph, A, B, ...)
        C.shape = (N, D1, D2)

    Optional id mapping arrays could be provided to read/write from/to locations
    other than node/edge ids. Each mapping array is a 1D integer vector, whose
    id bit-width should be consistent with the graph bit-width. The length of the
    mapping array should be equal to the number of rows of the corresponding
    operand. For example, if ``lhs=A`` and ``lhs_mapping=M``, then when reading/writing
    to A tensor::

        A[M[s1(i,j,e)]]

    If no id is provided and the operand target is edge 1) lhs or rhs is edge 2) or
    reducer is none so output is edge, edge id will be used to access corresponding
    feature tensor.

    Parameter
    ---------
    reducer : str
        The type of the reducer ("sum", "max", "min", "mean", "prod", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    binary_op : str
        The type of the binary functor ("add", "mul", "sub", "div").
    graph : GraphIndex
        The graph
    lhs : int
        The lhs target (src, dst, edge)
    rhs : int
        The rhs target (src, dst, edge)
    lhs_data : NDArray
        The lhs data.
    rhs_data : NDArray
        The rhs data.
    out_data : NDArray
        The out data.
    lhs_mapping : NDArray
        The lhs id mapping array.
    rhs_mapping : NDArray
        The rhs id mapping array.
    out_mapping : NDArray
        The out id mapping array.
    """
    if lhs_mapping is None:
        lhs_mapping = empty([])
    if rhs_mapping is None:
        rhs_mapping = empty([])
    if out_mapping is None:
        out_mapping = empty([])
    _CAPI_DGLKernelBinaryOpReduce(
        reducer, binary_op, graph._handle,
        int(lhs), int(rhs),
        lhs_data, rhs_data, out_data,
        lhs_mapping, rhs_mapping, out_mapping)

def backward_lhs_binary_op_reduce(
        reducer, binary_op, graph,
        lhs, rhs,
        lhs_data, rhs_data, out_data,
        grad_out_data, grad_lhs_data,
        lhs_mapping=None, rhs_mapping=None, out_mapping=None):
    """Compute lhs gradient of binary_op_reduce.

    The returned gradient tensor has the same shape as the grad_out_data. To compute
    the correct gradient, extra reduction along broadcasting dimensions is required.

    Parameter
    ---------
    reducer : str
        The type of the reducer ("sum", "max", "min", "mean", "prod", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    binary_op : str
        The type of the binary functor ("add", "mul", "sub", "div").
    graph : GraphIndex
        The graph
    lhs : int
        The lhs target (src, dst, edge)
    rhs : int
        The rhs target (src, dst, edge)
    lhs_data : NDArray
        The lhs data.
    rhs_data : NDArray
        The rhs data.
    out_data : NDArray
        The out data.
    grad_out_data : NDArray
        The out gradient data.
    grad_lhs_data : NDArray
        The lhs gradient data.
    lhs_mapping : NDArray
        The lhs id mapping array.
    rhs_mapping : NDArray
        The rhs id mapping array.
    out_mapping : NDArray
        The out id mapping array.
    """
    if lhs_mapping is None:
        lhs_mapping = empty([])
    if rhs_mapping is None:
        rhs_mapping = empty([])
    if out_mapping is None:
        out_mapping = empty([])
    _CAPI_DGLKernelBackwardLhsBinaryOpReduce(
        reducer, binary_op, graph._handle,
        int(lhs), int(rhs),
        lhs_mapping, rhs_mapping, out_mapping,
        lhs_data, rhs_data, out_data,
        grad_out_data, grad_lhs_data)

def backward_rhs_binary_op_reduce(
        reducer, binary_op, graph,
        lhs, rhs,
        lhs_data, rhs_data, out_data,
        grad_out_data, grad_rhs_data,
        lhs_mapping=None, rhs_mapping=None, out_mapping=None):
    """Compute rhs gradient of binary_op_reduce.

    The returned gradient tensor has the same shape as the grad_out_data. To compute
    the correct gradient, extra reduction along broadcasting dimensions is required.

    Parameter
    ---------
    reducer : str
        The type of the reducer ("sum", "max", "min", "mean", "prod", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    binary_op : str
        The type of the binary functor ("add", "mul", "sub", "div").
    graph : GraphIndex
        The graph
    lhs : int
        The lhs target (src, dst, edge)
    rhs : int
        The rhs target (src, dst, edge)
    lhs_data : NDArray
        The lhs data.
    rhs_data : NDArray
        The rhs data.
    out_data : NDArray
        The out data.
    grad_out_data : NDArray
        The out gradient data.
    grad_rhs_data : NDArray
        The lhs gradient data.
    lhs_mapping : NDArray
        The lhs id mapping array.
    rhs_mapping : NDArray
        The rhs id mapping array.
    out_mapping : NDArray
        The out id mapping array.
    """
    if lhs_mapping is None:
        lhs_mapping = empty([])
    if rhs_mapping is None:
        rhs_mapping = empty([])
    if out_mapping is None:
        out_mapping = empty([])
    _CAPI_DGLKernelBackwardRhsBinaryOpReduce(
        reducer, binary_op, graph._handle,
        int(lhs), int(rhs),
        lhs_mapping, rhs_mapping, out_mapping,
        lhs_data, rhs_data, out_data,
        grad_out_data, grad_rhs_data)

def copy_reduce(reducer, graph, target,
                in_data, out_data,
                in_mapping=None, out_mapping=None):
    """Copy target data and perform reduce by graph.

    Optional id mapping arrays could be provided to read/write from/to locations
    other than node/edge ids. Each mapping array is a 1D integer vector, whose
    id bit-width should be consistent with the graph bit-width. The length of the
    mapping array should be equal to the number of rows of the corresponding
    operand. For example, if ``in_data=A`` and ``in_mapping=M``, then when reading/writing
    to A tensor::

        A[M[s1(i,j,e)]]

    If no id is provided and the operand target is edge 1) target is edge 2) or
    reducer is none so output is edge, edge id will be used to access the corresponding
    feature tensor.

    Parameter
    ---------
    reducer : str
        The type of the reducer ("sum", "max", "min", "mean", "prod", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    graph : GraphIndex
        The graph
    target : int
        The input target (src, dst, edge)
    in_data : NDArray
        The input data.
    out_data : NDArray
        The out data.
    in_mapping : NDArray
        The input id mapping array.
    out_mapping : NDArray
        The out id mapping array.
    """
    if in_mapping is None:
        in_mapping = empty([])
    if out_mapping is None:
        out_mapping = empty([])
    _CAPI_DGLKernelCopyReduce(
        reducer, graph._handle, int(target),
        in_data, out_data, in_mapping, out_mapping)

def backward_copy_reduce(reducer, graph, target,
                         in_data, out_data,
                         grad_out_data, grad_in_data,
                         in_mapping=None, out_mapping=None):
    """Backward operator of copy reduce

    Optional id mapping arrays could be provided to read/write from/to locations
    other than node/edge ids.

    Parameter
    ---------
    reducer : str
        The type of the reducer ("sum", "max", "min", "mean", "prod", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    graph : GraphIndex
        The graph
    target : int
        The input target (src, dst, edge)
    in_data : NDArray
        The input data.
    out_data : NDArray
        The out data.
    grad_out_data : NDArray
        The out gradient data.
    grad_in_data : NDArray
        The input gradient data.
    in_mapping : NDArray
        The input id mapping array.
    out_mapping : NDArray
        The out id mapping array.
    """
    if in_mapping is None:
        in_mapping = empty([])
    if out_mapping is None:
        out_mapping = empty([])
    _CAPI_DGLKernelBackwardCopyReduce(
        reducer, graph._handle, int(target),
        in_data, out_data, grad_out_data, grad_in_data,
        in_mapping, out_mapping)

_init_api("dgl.kernel")
