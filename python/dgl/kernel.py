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

    Broadcasting is supported for feature dimensions.

    Optional id mapping arrays could be provided to read/write from/to locations
    other than node/edge ids.

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
    """Copy target data and perform reduce by graph.

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
