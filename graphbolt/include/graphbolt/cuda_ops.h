/**
 *   Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 *   All rights reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 * @file graphbolt/cuda_ops.h
 * @brief Available CUDA operations in Graphbolt.
 */
#ifndef GRAPHBOLT_CUDA_OPS_H_
#define GRAPHBOLT_CUDA_OPS_H_

#include <torch/script.h>

#include <type_traits>

namespace graphbolt {
namespace ops {

/**
 * @brief Sorts the given input and optionally returns the original indexes.
 *
 * @param input         A pointer to storage containing IDs.
 * @param num_items     Size of the input storage.
 * @param num_bits      An integer such that all elements of input tensor are
 *                      are less than (1 << num_bits).
 *
 * @return
 * - A tuple of tensors if return_original_positions is true, where the first
 * one includes sorted input, the second contains original positions of the
 * sorted result. If return_original_positions is false, then returns only the
 * sorted input.
 */
template <bool return_original_positions, typename scalar_t>
std::conditional_t<
    return_original_positions, std::pair<torch::Tensor, torch::Tensor>,
    torch::Tensor>
Sort(const scalar_t* input, int64_t num_items, int num_bits);

/**
 * @brief Sorts the given input and optionally returns the original indexes.
 *
 * @param input         A tensor containing IDs.
 * @param num_bits      An integer such that all elements of input tensor are
 *                      are less than (1 << num_bits).
 *
 * @return
 * - A tuple of tensors if return_original_positions is true, where the first
 * one includes sorted input, the second contains original positions of the
 * sorted result. If return_original_positions is false, then returns only the
 * sorted input.
 */
template <bool return_original_positions = true>
std::conditional_t<
    return_original_positions, std::pair<torch::Tensor, torch::Tensor>,
    torch::Tensor>
Sort(torch::Tensor input, int num_bits = 0);

/**
 * @brief Tests if each element of elements is in test_elements. Returns a
 * boolean tensor of the same shape as elements that is True for elements
 * in test_elements and False otherwise. Enhance torch.isin by implementing
 * multi-threaded searching, as detailed in the documentation at
 * https://pytorch.org/docs/stable/generated/torch.isin.html."
 *
 * @param elements        Input elements
 * @param test_elements   Values against which to test for each input element.
 *
 * @return
 * A boolean tensor of the same shape as elements that is True for elements
 * in test_elements and False otherwise.
 */
torch::Tensor IsIn(torch::Tensor elements, torch::Tensor test_elements);

/**
 * @brief Returns the indexes of the nonzero elements in the given boolean mask
 * if logical_not is false. Otherwise, returns the indexes of the zero elements
 * instead.
 *
 * @param mask        Input boolean mask.
 * @param logical_not Whether mask should be treated as ~mask.
 *
 * @return An int64_t tensor of the same shape as mask containing the indexes
 * of the selected elements.
 */
torch::Tensor Nonzero(torch::Tensor mask, bool logical_not);

/**
 * @brief Select columns for a sparse matrix in a CSC format according to nodes
 * tensor.
 *
 * NOTE: The shape of all tensors must be 1-D.
 *
 * @param in_degree Indegree tensor containing degrees of nodes being copied.
 * @param sliced_indptr Sliced_indptr tensor containing indptr values of nodes
 * being copied.
 * @param indices Indices tensor with edge information of shape (indptr[N],).
 * @param nodes Nodes tensor with shape (M,).
 * @param nodes_max An upperbound on `nodes.max()`.
 * @param output_size The total number of edges being copied.
 * @return (torch::Tensor, torch::Tensor) Output indptr and indices tensors of
 * shapes (M + 1,) and ((indptr[nodes + 1] - indptr[nodes]).sum(),).
 */
std::tuple<torch::Tensor, torch::Tensor> IndexSelectCSCImpl(
    torch::Tensor in_degree, torch::Tensor sliced_indptr, torch::Tensor indices,
    torch::Tensor nodes, int64_t nodes_max,
    torch::optional<int64_t> output_size = torch::nullopt);

/**
 * @brief Select columns for a sparse matrix in a CSC format according to nodes
 * tensor.
 *
 * NOTE: The shape of all tensors must be 1-D.
 *
 * @param indptr Indptr tensor containing offsets with shape (N,).
 * @param indices Indices tensor with edge information of shape (indptr[N],).
 * @param nodes Nodes tensor with shape (M,).
 * @param output_size The total number of edges being copied.
 * @return (torch::Tensor, torch::Tensor) Output indptr and indices tensors of
 * shapes (M + 1,) and ((indptr[nodes + 1] - indptr[nodes]).sum(),).
 */
std::tuple<torch::Tensor, torch::Tensor> IndexSelectCSCImpl(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes,
    torch::optional<int64_t> output_size = torch::nullopt);

/**
 * @brief Select columns for a sparse matrix in a CSC format according to nodes
 * tensor for a given list of tensors.
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
 * @return (torch::Tensor, std::vector<torch::Tensor>) Output indptr and vector
 * of indices tensors of shapes (M + 1,) and ((indptr[nodes + 1] -
 * indptr[nodes]).sum(),).
 */
std::tuple<torch::Tensor, std::vector<torch::Tensor>> IndexSelectCSCBatchedImpl(
    torch::Tensor indptr, std::vector<torch::Tensor> indices_list,
    torch::Tensor nodes, bool with_edge_ids,
    torch::optional<int64_t> output_size);

/**
 * @brief Slices the indptr tensor with nodes and returns the indegrees of the
 * given nodes and their indptr values.
 *
 * @param indptr The indptr tensor.
 * @param nodes  The nodes to read from indptr. If not provided, assumed to be
 * equal to torch.arange(indptr.size(0) - 1).
 *
 * @return Tuple of tensors with values:
 * (indptr[nodes + 1] - indptr[nodes], indptr[nodes]), the returned indegrees
 * tensor (first one) has size nodes.size(0) + 1 so that calling ExclusiveCumSum
 * on it gives the output indptr.
 */
std::tuple<torch::Tensor, torch::Tensor> SliceCSCIndptr(
    torch::Tensor indptr, torch::optional<torch::Tensor> nodes);

/**
 * @brief Given the compacted sub_indptr tensor, edge type tensor and
 * sliced_indptr tensor of the original graph, returns the heterogenous
 * versions of sub_indptr, indegrees and sliced_indptr.
 *
 * @param sub_indptr     The compacted indptr tensor.
 * @param etypes         The compacted type_per_edge tensor.
 * @param sliced_indptr  The sliced_indptr tensor of original graph.
 * @param num_fanouts    The number of fanout values.
 *
 * @return Tuple of tensors (new_sub_indptr, new_indegrees, new_sliced_indptr):
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SliceCSCIndptrHetero(
    torch::Tensor sub_indptr, torch::Tensor etypes, torch::Tensor sliced_indptr,
    int64_t num_fanouts);

/**
 * @brief Computes the exclusive prefix sum of the given input.
 *
 * @param input The input tensor.
 *
 * @return The prefix sum result such that r[i] = \sum_{j=0}^{i-1} input[j]
 */
torch::Tensor ExclusiveCumSum(torch::Tensor input);

/**
 * @brief Computes the gather operation on a given input and index tensor.
 *
 * @param input The input tensor.
 * @param index The index tensor.
 * @param dtype The optional output dtype. If not given, inferred from the input
 * tensor.
 *
 * @return The result of the input.gather(0, index).to(dtype) operation.
 */
torch::Tensor Gather(
    torch::Tensor input, torch::Tensor index,
    torch::optional<torch::ScalarType> dtype = torch::nullopt);

/**
 * @brief Select rows from input tensor according to index tensor.
 *
 * NOTE:
 * 1. The shape of input tensor can be multi-dimensional, but the index tensor
 * must be 1-D.
 * 2. Should be called if input is on pinned memory and index is on pinned
 * memory or GPU memory.
 *
 * @param input Input tensor with shape (N, ...).
 * @param index Index tensor with shape (M,).
 * @return torch::Tensor Output tensor with shape (M, ...).
 */
torch::Tensor UVAIndexSelectImpl(torch::Tensor input, torch::Tensor index);

/**
 * @brief ExpandIndptrImpl implements conversion from a given indptr offset
 * tensor to a COO format tensor. If node_ids is not given, it is assumed to be
 * equal to torch::arange(indptr.size(0) - 1, dtype=dtype).
 *
 * @param indptr       The indptr offset tensor.
 * @param dtype        The dtype of the returned output tensor.
 * @param node_ids     Optional 1D tensor represents the node ids.
 * @param output_size  Optional value of indptr[-1]. Passing it eliminates CPU
 * GPU synchronization.
 *
 * @return The resulting tensor.
 */
torch::Tensor ExpandIndptrImpl(
    torch::Tensor indptr, torch::ScalarType dtype,
    torch::optional<torch::Tensor> node_ids = torch::nullopt,
    torch::optional<int64_t> output_size = torch::nullopt);

/**
 * @brief IndptrEdgeIdsImpl implements conversion from a given indptr offset
 * tensor to a COO edge ids tensor. For a given indptr [0, 2, 5, 7] and offset
 * tensor [0, 100, 200], the output will be [0, 1, 100, 101, 102, 201, 202]. If
 * offset was not provided, the output would be [0, 1, 0, 1, 2, 0, 1].
 *
 * @param indptr       The indptr offset tensor.
 * @param dtype        The dtype of the returned output tensor.
 * @param offset       The offset tensor.
 * @param output_size  Optional value of indptr[-1]. Passing it eliminates CPU
 * GPU synchronization.
 *
 * @return The resulting tensor.
 */
torch::Tensor IndptrEdgeIdsImpl(
    torch::Tensor indptr, torch::ScalarType dtype,
    torch::optional<torch::Tensor> offset,
    torch::optional<int64_t> output_size);

/**
 * @brief Removes duplicate elements from the concatenated 'unique_dst_ids' and
 * 'src_ids' tensor and applies the uniqueness information to compact both
 * source and destination tensors.
 *
 * The function performs two main operations:
 *   1. Unique Operation: 'unique(concat(unique_dst_ids, src_ids))', in which
 * the unique operator will guarantee the 'unique_dst_ids' are at the head of
 * the result tensor.
 *   2. Compact Operation: Utilizes the reverse mapping derived from the unique
 * operation to transform 'src_ids' and 'dst_ids' into compacted IDs.
 *
 * When world_size is greater than 1, then the given ids are partitioned between
 * the available ranks. The ids corresponding to the given rank are guaranteed
 * to come before the ids of other ranks. To do this, the partition ids are
 * rotated backwards by the given rank so that the ids are ordered as:
 * [rank, rank + 1, world_size, 0, ..., rank - 1]. This is supported only for
 * Volta and later generation NVIDIA GPUs.
 *
 * @param src_ids         A tensor containing source IDs.
 * @param dst_ids         A tensor containing destination IDs.
 * @param unique_dst_ids  A tensor containing unique destination IDs, which is
 *                        exactly all the unique elements in 'dst_ids'.
 * @param rank            The rank of the current GPU.
 * @param world_size      The total # GPUs, world size.
 *
 * @return (unique_ids, compacted_src_ids, compacted_dst_ids, unique_offsets)
 * - A tensor representing all unique elements in 'src_ids' and 'dst_ids' after
 * removing duplicates. The indices in this tensor precisely match the compacted
 * IDs of the corresponding elements.
 * - The tensor corresponding to the 'src_ids' tensor, where the entries are
 * mapped to compacted IDs.
 * - The tensor corresponding to the 'dst_ids' tensor, where the entries are
 * mapped to compacted IDs.
 * - The tensor corresponding to the offsets into the unique_ids tensor. Has
 * size `world_size + 1` and unique_ids[offsets[i]: offsets[i + 1]] belongs to
 * the rank `(rank + i) % world_size`.
 *
 * @example
 *   torch::Tensor src_ids = src
 *   torch::Tensor dst_ids = dst
 *   torch::Tensor unique_dst_ids = torch::unique(dst);
 *   auto result = UniqueAndCompact(src_ids, dst_ids, unique_dst_ids);
 *   torch::Tensor unique_ids = std::get<0>(result);
 *   torch::Tensor compacted_src_ids = std::get<1>(result);
 *   torch::Tensor compacted_dst_ids = std::get<2>(result);
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
UniqueAndCompact(
    const torch::Tensor src_ids, const torch::Tensor dst_ids,
    const torch::Tensor unique_dst_ids, const int64_t rank,
    const int64_t world_size);

/**
 * @brief Batched version of UniqueAndCompact. The ith element of the return
 * value is equal to the passing the ith elements of the input arguments to
 * UniqueAndCompact.
 */
std::vector<
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
UniqueAndCompactBatched(
    const std::vector<torch::Tensor>& src_ids,
    const std::vector<torch::Tensor>& dst_ids,
    const std::vector<torch::Tensor>& unique_dst_ids, const int64_t rank,
    const int64_t world_size);

}  //  namespace ops
}  //  namespace graphbolt

#endif  // GRAPHBOLT_CUDA_OPS_H_
