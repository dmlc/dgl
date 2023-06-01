/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/src/columnwise_pick.h
 * @brief Contains the functions declaration for column pick.
 */

#ifndef GRAPHBOLT_COLUMN_PICK_H_
#define GRAPHBOLT_COLUMN_PICK_H_

#include <graphbolt/csc_sampling_graph.h>
#include <graphbolt/threading_utils.h>

#include "macro.h"

namespace graphbolt {
namespace sampling {

/**
 * Function used to pick samples from a range.
 *
 * @param start The starting value of the range.
 * @param end The ending value of the range.
 * @param num_samples The number of samples to pick from the range.
 * @param out [out] The output to store the picked samples.
 */
template <typename IdxType>
using RangePickFn = std::function<void(
    IdxType off, IdxType len, IdxType num_samples, IdxType* out)>;

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

template <typename IdxType, typename ProbType>
inline RangePickFn<IdxType> GetRangePickFn(
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
template <typename NodeIdType, typename EdgeIdType, typename EtypeIdType>
c10::intrusive_ptr<SampledSubgraph> ColumnWisePickPerEtype(
    const CSCPtr graph, const torch::Tensor& columns,
    const torch::Tensor& fanouts, const torch::optional<torch::Tensor>& probs,
    bool return_eids, bool replace, bool consider_etype, NumPickFn num_pick_fn,
    RangePickFn<EdgeIdType> pick_fn);

c10::intrusive_ptr<SampledSubgraph> ColumnWiseSampling(
    const CSCPtr graph, const torch::Tensor& seed_nodes, const torch::Tensor& fanouts, bool replace,
    bool return_eids, bool consider_etype,
    const torch::optional<torch::Tensor>& probs);


}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_COLUMN_PICK_H_
