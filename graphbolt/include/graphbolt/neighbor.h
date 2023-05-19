/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/neighbor.h
 * @brief Header file of neighbor sampling.
 */

#ifndef GRAPHBOLT_NEIGHBOR_H_
#define GRAPHBOLT_NEIGHBOR_H_

#include <graphbolt/csc_sampling_graph.h>

namespace graphbolt {
namespace sampling {

using TensorList = std::vector<torch::Tensor>;
using CSCPtr = c10::intrusive_ptr<CSCSamplingGraph>;
using RangePickFn = std::function<torch::Tensor(
    int64_t start, int64_t end, int64_t num_samples)>;

/**
 * @brief Performs neighbor sampling for a given set of seed nodes taking edge type into account, where each edge type has a specified pick number.
 *
 * @param graph The pointer to a csc sampling graph.
 * @param seed_nodes The tensor containing the seed nodes.
 * @param fanouts The vector containing the number of neighbors to sample per edge type.
 * @param replace Boolean indicating if sampling is done with replacement.
 * @param require_eids Boolean indicating if edge IDs are required.
 * @param probs Optional tensor containing probabilities for sampling.
 * @return A tuple containing the sampled neighbor nodes and their corresponding edge IDs (if required).
 */
std::tuple<TensorList, TensorList> SampleEtypeNeighbors(
    const CSCPtr graph,
    torch::Tensor seed_nodes,
    const std::vector<int64_t>& fanouts,
    bool replace,
    bool require_eids,
    const torch::optional<torch::Tensor>& probs);

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_NEIGHBOR_H_
