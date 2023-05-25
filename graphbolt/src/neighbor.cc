/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/src/neighbor.cc
 * @brief Source file of neighbor sampling.
 */

#include <graphbolt/rowwise_pick.h>
#include <graphbolt/sampled_subgraph.h>

#include <algorithm>

namespace graphbolt {
namespace sampling {

c10::intrusive_ptr<SampledSubgraph> CSCSamplingGraph::SampleEtypeNeighbors(
    torch::Tensor seed_nodes, torch::Tensor fanouts, bool replace,
    bool return_eids, const torch::optional<torch::Tensor>& probs) {
  TORCH_CHECK(
      type_per_edge_.has_value(),
      "SampleNeighborsEType only works on graphs where the number of edge "
      "types > 1.")

  const int64_t num_nodes = seed_nodes.size(0);
  bool all_fanout_zero = (fanouts == 0).all().item<bool>();
  torch::Tensor picked_row_ptr, picked_cols, picked_etypes, picked_eids;
  torch::optional<torch::Tensor> picked_eids_or_null = torch::nullopt;
  if (return_eids) picked_eids_or_null = torch::tensor({}, indptr_.options());

  if (num_nodes == 0 || all_fanout_zero) {
    // Empty graph
    picked_row_ptr = torch::zeros({num_nodes + 1}, indptr_.options());
    picked_cols = torch::tensor({}, indices_.options());
    picked_etypes = torch::tensor({}, type_per_edge_.value().options());
  } else {
    auto pick_fn = GetRangePickFn(probs, replace);
    std::tie(picked_row_ptr, picked_cols, picked_etypes, picked_eids) =
        RowWisePickPerEtype(
            this, seed_nodes, fanouts, probs, return_eids, replace, pick_fn);
    if (return_eids) picked_eids_or_null = picked_eids;
  }
  return c10::make_intrusive<SampledSubgraph>(
      picked_row_ptr, picked_cols, seed_nodes, torch::nullopt,
      picked_eids_or_null, picked_etypes);
}

}  // namespace sampling
}  // namespace graphbolt
