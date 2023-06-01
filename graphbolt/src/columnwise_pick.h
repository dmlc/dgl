/**
 *  Copyright (c) 2023 by Contributors
 * @file columnwise_pick.h
 * @brief Contains the methods definition for column wise pick.
 */

#ifndef GRAPHBOLT_COLUMNWISE_PICK_H_
#define GRAPHBOLT_COLUMNWISE_PICK_H_

#include <graphbolt/csc_sampling_graph.h>
#include <graphbolt/sampled_subgraph.h>
#include <graphbolt/utils.h>

using namespace graphbolt::utils;

namespace graphbolt {
namespace sampling {

/**
 * @brief Alias for a function type that takes start, end, and num_samples as arguments and returns a torch Tensor.
 *
 * @param start The starting index for picking elements.
 * @param end The ending index for picking elements.
 * @param num_samples The number of samples to be picked.
 * @return A torch Tensor containing the picked elements.
 */
using RangePickFn = std::function<torch::Tensor(int64_t start, int64_t end, int64_t num_samples)>;

/**
 *  @brief  Alias for a vector of torch Tensors.
 */
using TensorList = std::vector<torch::Tensor>;


static constexpr int kDefaultPickGrainSize = 32;

/**
 * @brief Performs column-wise pick in a graph and returns a sampled subgraph.
 *
 * @param graph A pointer to the CSCPtr graph.
 * @param columns The tensor containing column indices.
 * @param num_picks A vector containing the number of picks for each edge type.
 * @param probs An optional tensor containing probability values.
 * @param require_eids Boolean indicating whether to require edge IDs.
 * @param replace Boolean indicating whether to replace picked elements.
 * @param pick_fn A function that takes start, end, and num_samples as arguments and returns a tensor.
 * @return A pointer to the SampledSubgraph representing the sampled subgraph.
 */
c10::intrusive_ptr<SampledSubgraph> ColumnWisePick(
    const CSCPtr graph, const torch::Tensor& columns,
    const std::vector<int64_t>& num_picks,
    const torch::optional<torch::Tensor>& probs, bool require_eids,
    bool replace, RangePickFn& pick_fn);


}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_COLUMNWISE_PICK_H_
