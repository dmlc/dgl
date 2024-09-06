/**
 *  Copyright (c) 2023 by Contributors
 * @file index_select.h
 * @brief Index select operators.
 */
#ifndef GRAPHBOLT_INDEX_SELECT_H_
#define GRAPHBOLT_INDEX_SELECT_H_

#include <graphbolt/async.h>
#include <torch/script.h>

namespace graphbolt {
namespace ops {

/**
 * @brief Select columns for a sparse matrix in a CSC format according to nodes
 * tensor.
 *
 * NOTE:
 * 1. The shape of all tensors must be 1-D.
 * 2. If indices is on pinned memory and nodes is on pinned memory or GPU
 * memory, then UVAIndexSelectCSCImpl will be called. If indices is on GPU
 * memory, then IndexSelectCSCImpl will be called. Otherwise,
 * FusedCSCSamplingGraph::InSubgraph will be called.
 *
 * @param indptr Indptr tensor containing offsets with shape (N,).
 * @param indices Indices tensor with edge information of shape (indptr[N],).
 * @param nodes Nodes tensor with shape (M,).
 * @param output_size The total number of edges being copied.
 * @return (torch::Tensor, torch::Tensor) Output indptr and indices tensors of
 * shapes (M + 1,) and ((indptr[nodes + 1] - indptr[nodes]).sum(),).
 */
std::tuple<torch::Tensor, torch::Tensor> IndexSelectCSC(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes,
    torch::optional<int64_t> output_size = torch::nullopt);

/**
 * @brief Select rows from input tensor according to index tensor.
 *
 * NOTE:
 * 1. The shape of input tensor can be multi-dimensional, but the index tensor
 * must be 1-D.
 * 2. If input is on pinned memory and index is on pinned memory or GPU memory,
 * then UVAIndexSelectImpl will be called. Otherwise, torch::index_select will
 * be called.
 *
 * @param input Input tensor with shape (N, ...).
 * @param index Index tensor with shape (M,).
 * @return torch::Tensor Output tensor with shape (M, ...).
 */
torch::Tensor IndexSelect(torch::Tensor input, torch::Tensor index);

/**
 * @brief The async version of IndexSelect, available for only CPU tensors.
 *
 * @return Returns a future containing a torch::Tensor.
 */
c10::intrusive_ptr<Future<torch::Tensor>> IndexSelectAsync(
    torch::Tensor input, torch::Tensor index);

/**
 * @brief The async version of operation input[index] = src.
 * @param input The input tensor.
 * @param index The index tensor into input.
 * @param src The src tensor being assigned into input.
 *
 * @return Returns a future containing input, a torch::Tensor.
 */
c10::intrusive_ptr<Future<torch::Tensor>> ScatterAsync(
    torch::Tensor input, torch::Tensor index, torch::Tensor src);

/**
 * @brief Select columns for a sparse matrix in a CSC format according to nodes
 * tensor.
 *
 * NOTE: The shape of all tensors must be 1-D.
 *
 * @param indptr Indptr tensor containing offsets with shape (N,).
 * @param indices_list Vector of indices tensor with edge information of shape
 * (indptr[N],).
 * @param nodes Nodes tensor with shape (M,).
 * @param with_edge_ids Whether to return edge ids tensor corresponding to
 * sliced edges as the last element of the output.
 * @param output_size The total number of edges being copied.
 *
 * @return (torch::Tensor, std::vector<torch::Tensor>) Output indptr and vector
 * of indices tensors of shapes (M + 1,) and ((indptr[nodes + 1] -
 * indptr[nodes]).sum(),).
 */
std::tuple<torch::Tensor, std::vector<torch::Tensor>> IndexSelectCSCBatched(
    torch::Tensor indptr, std::vector<torch::Tensor> indices_list,
    torch::Tensor nodes, bool with_edge_ids,
    torch::optional<int64_t> output_size);

c10::intrusive_ptr<
    Future<std::tuple<torch::Tensor, std::vector<torch::Tensor>>>>
IndexSelectCSCBatchedAsync(
    torch::Tensor indptr, std::vector<torch::Tensor> indices_list,
    torch::Tensor nodes, bool with_edge_ids,
    torch::optional<int64_t> output_size);

}  // namespace ops
}  // namespace graphbolt

#endif  // GRAPHBOLT_INDEX_SELECT_H_
