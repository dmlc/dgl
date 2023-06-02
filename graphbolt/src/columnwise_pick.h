/**
 *  Copyright (c) 2023 by Contributors
 * @file columnwise_pick.h
 * @brief Contains the methods definition for column wise pick.
 */

#ifndef GRAPHBOLT_COLUMNWISE_PICK_H_
#define GRAPHBOLT_COLUMNWISE_PICK_H_

#include <graphbolt/csc_sampling_graph.h>
#include <graphbolt/sampled_subgraph.h>

namespace graphbolt {
namespace sampling {

/**
 * @brief Alias for a function type that takes start, end, and num_samples as
 * arguments and returns a torch Tensor.
 *
 * @param start The starting index for picking elements.
 * @param end The ending index for picking elements.
 * @param num_samples The number of samples to be picked.
 * @return A torch Tensor containing the picked elements.
 */
using RangePickFn = std::function<torch::Tensor(
    int64_t start, int64_t end, int64_t num_samples)>;

/**
 *  @brief  Alias for a vector of torch Tensors.
 */
using TensorList = std::vector<torch::Tensor>;

/**
 * @brief Default pick grain size for parallelized picking.
 */
static constexpr int kDefaultPickGrainSize = 32;

/**
 * @brief Picks random values uniformly from a range [start, end) with
 * replacement.
 *
 * @param start The start of the range (inclusive).
 * @param end The end of the range (exclusive).
 * @param num_samples The number of samples to pick.
 * @return A tensor containing the picked values.
 */
inline torch::Tensor UniformPickWithReplace(
    int64_t start, int64_t end, int64_t num_samples);

/**
 * @brief Picks random values uniformly from a range [start, end) without
 * replacement.
 *
 * @param start The start of the range (inclusive).
 * @param end The end of the range (exclusive).
 * @param num_samples The number of samples to pick.
 * @return A tensor containing the picked values.
 */
inline torch::Tensor UniformPick(
    int64_t start, int64_t end, int64_t num_samples);

/**
 * @brief Retrieves the appropriate range picking function based on the given
 * parameters.
 *
 * @param probs Optional tensor of probabilities for each value in the range.
 * @param replace Determines whether values can be picked with replacement or
 * not.
 * @return The range picking function.
 */
RangePickFn GetRangePickFn(
    const torch::optional<torch::Tensor>& probs, bool replace);

/**
 * Picks rows for one column without considering etype.
 *
 * @param off The offset of the range within the tensor.
 * @param len The length of the range.
 * @param replace Determines whether values can be picked with replacement or
 * not.
 * @param probs Optional tensor of probabilities for each value in the range.
 * @param options Tensor options specifying the desired dtype and device of the
 * result.
 * @param num_pick The number of values to pick.
 * @param pick_fn The range picking function to use.
 * @return A tensor containing the picked rows.
 */
torch::Tensor Pick(
    int64_t off, int64_t len, bool replace,
    const torch::optional<torch::Tensor>& probs,
    const torch::TensorOptions& options, int64_t num_pick, RangePickFn pick_fn);

/**
 * Picks rows for one column with considering etype.
 *
 * @param off The offset of the range within the tensor.
 * @param len The length of the range.
 * @param replace Determines whether values can be picked with replacement or
 * not.
 * @param probs Optional tensor of probabilities for each value in the range.
 * @param options Tensor options specifying the desired dtype and device of the
 * result.
 * @param type_per_edge A tensor specifying the element type per edge.
 * @param num_picks A vector specifying the number of values to pick for each
 * element type.
 * @param pick_fn The range picking function to use.
 * @return A tensor containing the picked rows.
 */
torch::Tensor PickEtype(
    int64_t off, int64_t len, bool replace,
    const torch::optional<torch::Tensor>& probs,
    const torch::TensorOptions& options, const torch::Tensor& type_per_edge,
    const std::vector<int64_t>& num_picks, RangePickFn pick_fn);

/**
 * @brief Performs column-wise pick in a graph and returns a sampled subgraph.
 *
 * @param graph A pointer to the CSCPtr graph.
 * @param columns The tensor containing column indices.
 * @param num_picks A vector containing the number of picks for each edge type.
 * @param probs An optional tensor containing probability values.
 * @param require_eids Boolean indicating whether to require edge IDs.
 * @param replace Boolean indicating whether to replace picked elements.
 * @param consider_etype Boolean indicates whether considering etype during
 * sampling.
 * @param pick_fn A function that takes start, end, and num_samples as arguments
 * and returns a tensor.
 * @return A pointer to the SampledSubgraph representing the sampled subgraph.
 */
c10::intrusive_ptr<SampledSubgraph> ColumnWisePick(
    const CSCPtr graph, const torch::Tensor& columns,
    const std::vector<int64_t>& num_picks,
    const torch::optional<torch::Tensor>& probs, bool require_eids,
    bool replace, bool consider_etype, RangePickFn& pick_fn);

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_COLUMNWISE_PICK_H_
