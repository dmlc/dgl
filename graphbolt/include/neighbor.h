/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/include/neighbor.h
 * @brief Header file of neighbor sampling.
 */

#include "csc_sampling_graph.h"

namespace graphbolt {
namespace sampling {

    using TensorList = std::vector<torch::Tensor>;
    using CSRPtr = c10::intrusive_ptr<CSCSamplingGraph>;
    using RangePickFn = std::function<torch::Tensor(
        int64_t start, int64_t end, int64_t num_samples)>;
    
    std::tuple<TensorList, TensorList, TensorList> SampleEtypeNeighbors(
        const CSRPtr& graph, torch::Tensor seed_nodes,
        const std::vector<int64_t>& fanouts,
        bool replace, bool require_eids,
        const torch::optional<torch::Tensor>& probs);

}  // namespace sampling
}  // namespace graphbolt
