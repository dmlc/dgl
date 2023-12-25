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

std::pair<torch::Tensor, torch::Tensor> Sort(torch::Tensor input, int num_bits);

/**
 * @brief Computes the exclusive prefix sum of the given input.
 *
 * @param input The input tensor.
 *
 * @return The prefix sum result such that r[i] = \sum_{j=0}^{i-1} input[j]
 */
torch::Tensor ExclusiveCumSum(torch::Tensor input);

std::tuple<torch::Tensor, torch::Tensor> IndexSelectCSCImpl(
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

std::tuple<torch::Tensor, torch::Tensor> UVAIndexSelectCSCImpl(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes);

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
    torch::optional<torch::Tensor> probs_or_mask = torch::nullopt,
    torch::optional<int64_t> random_seed = torch::nullopt);

}  //  namespace ops
}  //  namespace graphbolt
