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

def binary_op_reduce(reducer, op, graph,
                     lhs, rhs,
                     lhs_data, rhs_data, out_data,
                     lhs_mapping=None, rhs_mapping=None, out_mapping=None):
    """Perform binary operation between the given data and reduce by the graph.

    Broadcasting is supported for feature dimensions.

    Optional id mapping arrays could be provided to read/write from/to locations
    other than node/edge ids.

    reducer : str
        The type of the reducer ("sum", "max", "min", "mean", "prod", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    op : str
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
        reducer, op, graph._handle,
        int(lhs), int(rhs),
        lhs_data, rhs_data, out_data,
        lhs_mapping, rhs_mapping, out_mapping)

def backward_lhs_binary_op_reduce(
        reducer, op, graph,
        lhs, rhs,
        lhs_data, rhs_data, out_data,
        grad_out_data, grad_lhs_data,
        lhs_mapping=None, rhs_mapping=None, out_mapping=None):
    """Compute lhs gradient of binary_op_reduce.

    The returned gradient tensor has the same shape as the grad_out_data. To compute
    the correct gradient, extra reduction along broadcasting dimensions is required.

    reducer : str
        The type of the reducer ("sum", "max", "min", "mean", "prod", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    op : str
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
        reducer, op, graph._handle,
        int(lhs), int(rhs),
        lhs_mapping, rhs_mapping, out_mapping,
        lhs_data, rhs_data, out_data,
        grad_out_data, grad_lhs_data)

def backward_rhs_binary_op_reduce(
        reducer, op, graph,
        lhs, rhs,
        lhs_data, rhs_data, out_data,
        grad_out_data, grad_rhs_data,
        lhs_mapping=None, rhs_mapping=None, out_mapping=None):
    """Compute rhs gradient of binary_op_reduce.

    The returned gradient tensor has the same shape as the grad_out_data. To compute
    the correct gradient, extra reduction along broadcasting dimensions is required.

    reducer : str
        The type of the reducer ("sum", "max", "min", "mean", "prod", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    op : str
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
        reducer, op, graph._handle,
        int(lhs), int(rhs),
        lhs_mapping, rhs_mapping, out_mapping,
        lhs_data, rhs_data, out_data,
        grad_out_data, grad_rhs_data)

def copy_src_reduce(reducer,
                    indptr, indices,
                    rev_indptr, rev_indices,
                    src_mapping, src_data,
                    out_mapping,
                    out_data):
    """Copy src node data and perform reduce.

    Parameter
    ---------
    reducer : str
        The type of the reducer ("sum", "max", "min", "mean", "prod", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    indptr : dgl.ndarray.NDArray
        An int64 row offset array for the graph CSR.
    indices : dgl.ndarray.NDArray
        An int64 column index array for the graph CSR.
    rev_indptr : dgl.ndarray.NDArray
        An int64 row offset array for the reverse graph CSR.
    rev_indices : dgl.ndarray.NDArray
        An int64 column index array for the reverse graph CSR.
    src_mapping : dgl.ndarray.NDArray
        An optional int64 array for source node mapping.
        If empty, source ids are consecutive integers [0, len(indptr) - 1).
        Source ids are used to read source node data.
    src_data : dgl.ndarray.NDArray
        The source node feature tensor.
    out_mapping : dgl.ndarray.NDArray
        An optional int64 array for output mapping. If reducer is
        "none", then it's a mapping to edge ids. Otherwise, it's
        mapping to destination node ids.
    out_size : int
        The size of the first dimension of the output array.
    out_data : dgl.ndarray.NDArray
        The output tensor. Could be either node or edge feature tensor
        depending on the reducer.
    """
    return _CAPI_DGLKernelCopySrcReduce(
        reducer, indptr, indices, rev_indptr, rev_indices,
        src_mapping, src_data, out_mapping,
        out_data)

def backward_copy_src_reduce(
        reducer,
        indptr, indices,
        rev_indptr, rev_indices,
        src_mapping, out_mapping,
        src_data, out_data,
        grad_out_data,
        grad_src_data):
    """Backward operator for CopySrcReduce.

    Parameter
    ---------
    reducer : str
        The type of the reducer ("sum", "max", "mean", "min", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    indptr : dgl.ndarray.NDArray
        An int64 row offset array for the graph CSR.
    indices : dgl.ndarray.NDArray
        An int64 column index array for the graph CSR.
    rev_indptr : dgl.ndarray.NDArray
        An int64 row offset array for the reverse graph CSR.
    rev_indices : dgl.ndarray.NDArray
        An int64 column index array for the reverse graph CSR.
    src_mapping : dgl.ndarray.NDArray
        An int64 array used for read src node data.
        `src_mapping[src_node_id]` stores the location to read data.
        Empty array represents identity mapping.
    out_mapping : dgl.ndarray.NDArray
        An int64 array used for write output data.
        `out_mapping[out_id]` stores the location to read data.
        Empty array represents identity mapping.
    src_data : dgl.ndarray.NDArray
        The source node feature tensor.
    out_data : dgl.ndarray.NDArray
        The forward output tensor.
    grad_out_data : dgl.ndarray.NDArray
        The gradient of the forward output tensor.
    grad_src_data : dgl.ndarray.NDArray
        The gradient of src data.
    """
    _CAPI_DGLKernelBackwardCopySrcReduce(
        reducer, indptr, indices, rev_indptr, rev_indices,
        src_mapping, out_mapping,
        src_data, out_data, grad_out_data, grad_src_data)

def copy_edge_reduce(reducer,
                     indptr, indices,
                     rev_indptr, rev_indices,
                     edge_mapping,
                     edge_data,
                     out_mapping,
                     out_data):
    """Copy edge data and perform reduce to destination node.

    Parameter
    ---------
    reducer : str
        The type of the reducer ("sum", "max", "min", "mean", "prod", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    indptr : dgl.ndarray.NDArray
        An int64 row offset array for the graph CSR.
    indices : dgl.ndarray.NDArray
        An int64 column index array for the graph CSR.
    rev_indptr : dgl.ndarray.NDArray
        An int64 row offset array for the reverse graph CSR.
    rev_indices : dgl.ndarray.NDArray
        An int64 column index array for the reverse graph CSR.
    edge_mapping : dgl.ndarray.NDArray
        An int64 array used for read edge data.
        `edge_mapping[edge_id]` stores the location to read data.
        Empty array represents identity mapping.
    edge_data : dgl.ndarray.NDArray
        The edge feature tensor.
    out_mapping : dgl.ndarray.NDArray
        An int64 array used for write output data.
        `out_mapping[out_id]` stores the location to read data.
        Empty array represents identity mapping.
    out_size : int
        The number of rows of the output tensor.

    Returns
    -------
    dgl.ndarray.NDArray
        The output tensor. Could be either node or edge feature tensor
        depending on the reducer.
    """
    _CAPI_DGLKernelCopyEdgeReduce(
        reducer, indptr, indices, rev_indptr, rev_indices,
        edge_mapping, edge_data, out_mapping,
        out_data)

def backward_copy_edge_reduce(
        reducer,
        indptr, indices,
        rev_indptr, rev_indices,
        edge_mapping, out_mapping,
        edge_data, out_data,
        grad_out_data,
        grad_edge_data):
    """Backward operator for CopyEdgeReduce.

    Parameter
    ---------
    reducer : str
        The type of the reducer ("sum", "max", "mean", "min", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    indptr : dgl.ndarray.NDArray
        An int64 row offset array for the graph CSR.
    indices : dgl.ndarray.NDArray
        An int64 column index array for the graph CSR.
    rev_indptr : dgl.ndarray.NDArray
        An int64 row offset array for the reverse graph CSR.
    rev_indices : dgl.ndarray.NDArray
        An int64 column index array for the reverse graph CSR.
    edge_mapping : dgl.ndarray.NDArray
        An int64 array used for read edge data.
        `edge_mapping[edge_id]` stores the location to read data.
        Empty array represents identity mapping.
    out_mapping : dgl.ndarray.NDArray
        An int64 array used for write output data.
        `out_mapping[out_id]` stores the location to read data.
        Empty array represents identity mapping.
    edge_data : dgl.ndarray.NDArray
        The edge feature tensor.
    out_data : dgl.ndarray.NDArray
        The forward output tensor.
    grad_out_data : dgl.ndarray.NDArray
        The gradient of the forward output tensor.
    grad_edge_data : dgl.ndarray.NDArray
        The gradient of edge data.
    """
    _CAPI_DGLKernelBackwardCopyEdgeReduce(
        reducer, indptr, indices, rev_indptr, rev_indices,
        edge_mapping, out_mapping,
        edge_data, out_data, grad_out_data, grad_edge_data)

_init_api("dgl.kernel")
