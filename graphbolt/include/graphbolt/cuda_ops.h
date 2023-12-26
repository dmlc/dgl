/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file graphbolt/cuda_ops.h
 * @brief Available CUDA operations in Graphbolt.
 */

#include <graphbolt/fused_sampled_subgraph.h>
#include <torch/script.h>

namespace graphbolt {
namespace ops {

/**
 * @brief Sorts the given input and also returns the original indexes.
 *
 * @param input         A tensor containing IDs.
 * @param num_bits      An integer such that all elements of input tensor are
 *                      are less than (1 << num_bits).
 *
 * @return
 * - A tuple of tensors, the first one includes sorted input, the second
 * contains original positions of the sorted result.
 */
std::pair<torch::Tensor, torch::Tensor> Sort(
    torch::Tensor input, int num_bits = 0);

/**
 * @brief Select columns for a sparse matrix in a CSC format according to nodes
 * tensor.
 *
 * NOTE:
 * 1. The shape of all tensors must be 1-D.
 * 2. Should be called if all input tensors are on device memory.
 *
 * @param indptr Indptr tensor containing offsets with shape (N,).
 * @param indices Indices tensor with edge information of shape (indptr[N],).
 * @param nodes Nodes tensor with shape (M,).
 * @return (torch::Tensor, torch::Tensor) Output indptr and indices tensors of
 * shapes (M + 1,) and ((indptr[nodes + 1] - indptr[nodes]).sum(),).
 */
std::tuple<torch::Tensor, torch::Tensor> IndexSelectCSCImpl(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes);

/**
 * @brief Select columns for a sparse matrix in a CSC format according to nodes
 * tensor.
 *
 * NOTE:
 * 1. The shape of all tensors must be 1-D.
 * 2. Should be called if indices tensor is on pinned memory.
 *
 * @param indptr Indptr tensor containing offsets with shape (N,).
 * @param indices Indices tensor with edge information of shape (indptr[N],).
 * @param nodes Nodes tensor with shape (M,).
 * @return (torch::Tensor, torch::Tensor) Output indptr and indices tensors of
 * shapes (M + 1,) and ((indptr[nodes + 1] - indptr[nodes]).sum(),).
 */
std::tuple<torch::Tensor, torch::Tensor> UVAIndexSelectCSCImpl(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes);

/**
 * @brief Slices the indptr tensor with nodes and returns the indegrees of the
 * given nodes and their indptr values.
 *
 * @param indptr The indptr tensor.
 * @param nodes  The nodes to read from indptr
 *
 * @return Tuple of tensors with values:
 * (indptr[nodes + 1] - indptr[nodes], indptr[nodes]), the returned indegrees
 * tensor (first one) has size nodes.size(0) + 1 so that calling ExclusiveCumSum
 * on it gives the output indptr.
 */
std::tuple<torch::Tensor, torch::Tensor> SliceCSCIndptr(
    torch::Tensor indptr, torch::Tensor nodes);

/**
 * @brief Computes the exclusive prefix sum of the given input.
 *
 * @param input The input tensor.
 *
 * @return The prefix sum result such that r[i] = \sum_{j=0}^{i-1} input[j]
 */
torch::Tensor ExclusiveCumSum(torch::Tensor input);

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
 * @brief Sample neighboring edges of the given nodes and return the induced
 * subgraph.
 *
 * @param indptr Index pointer array of the CSC.
 * @param indices Indices array of the CSC.
 * @param nodes The nodes from which to sample neighbors.
 * @param fanouts The number of edges to be sampled for each node with or
 * without considering edge types.
 *   - When the length is 1, it indicates that the fanout applies to all
 * neighbors of the node as a collective, regardless of the edge type.
 *   - Otherwise, the length should equal to the number of edge types, and
 * each fanout value corresponds to a specific edge type of the node.
 * The value of each fanout should be >= 0 or = -1.
 *   - When the value is -1, all neighbors will be chosen for sampling. It is
 * equivalent to selecting all neighbors with non-zero probability when the
 * fanout is >= the number of neighbors (and replacement is set to false).
 *   - When the value is a non-negative integer, it serves as a minimum
 * threshold for selecting neighbors.
 * @param replace Boolean indicating whether the sample is preformed with or
 * without replacement. If True, a value can be selected multiple times.
 * Otherwise, each value can be selected only once.
 * @param layer Boolean indicating whether neighbors should be sampled in a
 * layer sampling fashion. Uses the LABOR-0 algorithm to increase overlap of
 * sampled edges, see arXiv:2210.13339.
 * @param return_eids Boolean indicating whether edge IDs need to be returned,
 * typically used when edge features are required.
 * @param type_per_edge A tensor representing the type of each edge, if present.
 * @param probs_or_mask An optional tensor with (unnormalized) probabilities
 * corresponding to each neighboring edge of a node. It must be
 * a 1D tensor, with the number of elements equaling the total number of edges.
 * @param random_seed An optional random_seed parameter to fix the random_seed
 * of the sampling operation. (Useful when layer=True)
 *
 * @return An intrusive pointer to a FusedSampledSubgraph object containing
 * the sampled graph's information.
 */
c10::intrusive_ptr<sampling::FusedSampledSubgraph> SampleNeighbors(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes,
    const std::vector<int64_t>& fanouts, bool replace, bool layer,
    bool return_eids,
    torch::optional<torch::Tensor> type_per_edge = torch::nullopt,
    torch::optional<torch::Tensor> probs_or_mask = torch::nullopt);

/**
 * @brief CSRToCOO implements conversion from a given indptr offset tensor to a
 * COO format tensor including ids in [0, indptr.size(0) - 1).
 *
 * @param input          A tensor containing IDs.
 * @param output_dtype   Dtype of output.
 *
 * @return
 * - The resulting tensor with output_dtype.
 */
torch::Tensor CSRToCOO(torch::Tensor indptr, torch::ScalarType output_dtype);

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
 * @param src_ids         A tensor containing source IDs.
 * @param dst_ids         A tensor containing destination IDs.
 * @param unique_dst_ids  A tensor containing unique destination IDs, which is
 *                        exactly all the unique elements in 'dst_ids'.
 *
 * @return
 * - A tensor representing all unique elements in 'src_ids' and 'dst_ids' after
 * removing duplicates. The indices in this tensor precisely match the compacted
 * IDs of the corresponding elements.
 * - The tensor corresponding to the 'src_ids' tensor, where the entries are
 * mapped to compacted IDs.
 * - The tensor corresponding to the 'dst_ids' tensor, where the entries are
 * mapped to compacted IDs.
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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> UniqueAndCompact(
    const torch::Tensor src_ids, const torch::Tensor dst_ids,
    const torch::Tensor unique_dst_ids, int num_bits = 0);

}  //  namespace ops
}  //  namespace graphbolt
