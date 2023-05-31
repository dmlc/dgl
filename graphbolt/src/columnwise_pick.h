/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/src/columnwise_pick.h
 * @brief Contains the functions declaration for column pick.
 */

#ifndef GRAPHBOLT_COLUMN_PICK_H_
#define GRAPHBOLT_COLUMN_PICK_H_

#include <graphbolt/csc_sampling_graph.h>
#include <graphbolt/threading_utils.h>

namespace graphbolt {
namespace sampling {

/**
 * Function used to pick samples from a range.
 *
 * @param start The starting value of the range.
 * @param end The ending value of the range.
 * @param num_samples The number of samples to pick from the range.
 * @param out [out] The output tensor to store the picked samples.
 */
using RangePickFn = std::function<void(
    int64_t start, int64_t end, int64_t num_samples, torch::Tensor& out)>;

/**
 * Function used to calculate how many samples will be picked for a given range.
 *
 * @param start The starting value of the range.
 * @param end The ending value of the range.
 * @param num_samples The total number of samples to pick.
 * @return The number of samples that will be picked from the range.
 */
using NumPickFn =
    std::function<int64_t(int64_t start, int64_t end, int64_t num_samples)>;

using TensorList = std::vector<torch::Tensor>;

/**
 * Default pick grain size used for picking samples.
 * The pick grain size determines the number of elements processed in each
 * thread.
 */
static constexpr int kDefaultPickGrainSize = 2;

NumPickFn GetNumPickFn(
    const torch::optional<torch::Tensor>& probs, bool replace);

inline void UniformRangePickWithReplacement(
    int64_t start, int64_t end, int64_t num_samples, torch::Tensor& out) {
  torch::randint_out(out, start, end, {num_samples});
}

inline void UniformRangePickWithoutReplacement(
    int64_t start, int64_t end, int64_t num_samples, torch::Tensor& out) {
  auto perm = torch::randperm(end - start) + start;
  out.copy_(perm.slice(0, 0, num_samples));
}

RangePickFn GetRangePickFn(
    const torch::optional<torch::Tensor>& probs, bool replace);

/**
 * @brief Performs column-wise picking based on the given parameters.
 *
 * @param graph The pointer to a csc sampling graph.
 * @param columns The tensor containing the column indices.
 * @param num_picks The tensor containing the number of picks per edge type.
 * @param probs Optional tensor containing probabilities for picking.
 * @param return_eids Boolean indicating if edge IDs need to be returned. The
 * last TensorList in the tuple is this value when required.
 * @param consider_etype Boolean indicating if considering edge type during
 * sampling, if set, sampling for each edge type of each seed node, otherwise
 * just sample once for each node.s
 * @param replace Boolean indicating if picking is done with replacement.
 * @param pick_fn The function used for picking.
 * @return A pointer to a 'SampledSubgraph'.
 */
template <typename EtypeType>
c10::intrusive_ptr<SampledSubgraph> ColumnWisePickPerEtype(
    const CSCPtr graph, const torch::Tensor& columns,
    const torch::Tensor& fanouts, const torch::optional<torch::Tensor>& probs,
    bool return_eids, bool replace, bool consider_etype, NumPickFn num_pick_fn,
    RangePickFn pick_fn);

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_COLUMN_PICK_H_
