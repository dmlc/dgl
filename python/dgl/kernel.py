"""Module for dgl kernels for graph computation."""
from __future__ import absolute_import

from ._ffi.function import _init_api


def src_mul_edge_reduce(reducer,
                        mul_op,
                        indptr,
                        indices,
                        src_mapping,
                        edge_mapping,
                        src_data,
                        edge_data,
                        out_mapping,
                        out_size):
    """Multiply src node data with edge data and perform reduce.

    Parameter
    ---------
    reducer : str
        The type of the reducer ("sum", "max", "mean", "min", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    mul : str
        The type of the mul functor ("mul", "add").
    indptr : dgl.ndarray.NDArray
        An int64 row offset array for the graph CSR.
    indices : dgl.ndarray.NDArray
        An int64 column index array for the graph CSR.
    src_mapping : dgl.ndarray.NDArray
        An int64 array used for read src node data.
        `src_mapping[src_node_id]` stores the location to read data.
        Empty array represents identity mapping.
    edge_mapping : dgl.ndarray.NDArray
        An int64 array used for read edge data.
        `edge_mapping[edge_id]` stores the location to read data.
        Empty array represents identity mapping.
    src_data : dgl.ndarray.NDArray
        The source node feature tensor.
    edge_data : dgl.ndarray.NDArray
        The edge feature tensor.
    edge_mapping : dgl.ndarray.NDArray
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
    return _CAPI_DGLKernelSrcMulEdgeReduce(
        reducer, mul_op, indptr, indices, src_mapping, edge_mapping, src_data,
        edge_data, out_mapping, int(out_size))


def src_mul_dst_reduce(reducer,
                       mul_op,
                       indptr,
                       indices,
                       src_mapping,
                       dst_mapping,
                       src_data,
                       dst_data,
                       out_mapping,
                       out_size):
    """Multiply src node data with dst node data and perform reduce.

    Parameter
    ---------
    reducer : str
        The type of the reducer ("sum", "max", "mean", "min", "none").
        If the reducer is "none", the output is an edge feature tensor.
        Otherwise, a node feature tensor is returned.
    mul : str
        The type of the mul functor ("mul", "add").
    indptr : dgl.ndarray.NDArray
        An int64 row offset array for the graph CSR.
    indices : dgl.ndarray.NDArray
        An int64 column index array for the graph CSR.
    edge_ids : dgl.ndarray.NDArray
        An int64 array for the edge ids. If empty,
        the edge ids are consecutive integers [0, len(indices)).
        The edge ids are used to read and write edge data.
    src_data : dgl.ndarray.NDArray
        The source node feature tensor.
    dst_data : dgl.ndarray.NDArray
        The destination node feature tensor.

    Returns
    -------
    dgl.ndarray.NDArray
        The output tensor. Could be either node or edge feature tensor
        depending on the reducer.
    """
    return _CAPI_DGLKernelSrcMulEdgeReduce(
        reducer, mul_op, indptr, indices, src_mapping, dst_mapping, src_data,
        dst_data, out_mapping, out_size)


def copy_src_reduce(reducer,
                    indptr,
                    indices,
                    src_mapping,
                    src_data,
                    out_mapping,
                    out_size):
    """Copy src node data and perform reduce.

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

    Returns
    -------
    dgl.ndarray.NDArray
        The output tensor. Could be either node or edge feature tensor
        depending on the reducer.
    """
    return _CAPI_DGLKernelCopySrcReduce(
        reducer, indptr, indices, src_mapping, src_data, out_mapping,
        int(out_size))


def copy_edge_reduce(reducer,
                     indptr,
                     indices,
                     edge_mapping,
                     edge_data,
                     out_mapping,
                     out_size):
    """Copy edge data and perform reduce to destination node.

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
    edge_mapping : dgl.ndarray.NDArray
        An optional int64 array for edge mapping.
        If empty, source ids are consecutive integers [0, len(indptr) - 1).
        Source ids are used to read edge data.
    edge_data : dgl.ndarray.NDArray
        The edge feature tensor.
    out_mapping : dgl.ndarray.NDArray
        An optional int64 array for output mapping. If reducer is
        "none", then it's a mapping to edge ids. Otherwise, it's
        mapping to destination node ids.
    out_size : int
        The size of the first dimension of the output array.

    Returns
    -------
    dgl.ndarray.NDArray
        The output tensor. Could be either node or edge feature tensor
        depending on the reducer.
    """
    return _CAPI_DGLKernelCopySrcReduce(
        reducer, indptr, indices, edge_mapping, edge_data, out_mapping,
        int(out_size))


_init_api("dgl.kernel")
