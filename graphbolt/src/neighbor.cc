/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/src/neighbor.cc
 * @brief Source file of neighbor sampling.
 */

#include <algorithm>

#include <graphbolt/rowwise_pick.h>

namespace graphbolt {
namespace sampling {

RangePickFn GetRangePickFn(
    const torch::optional<torch::Tensor>& probs, bool replace) {
  RangePickFn pick_fn;
  if (probs.has_value()) {
    pick_fn = [probs, replace](
                  int64_t start, int64_t end, int64_t num_samples) {
      // what if probs is mask having uint type
      auto local_probs = probs.value().slice(0, start, end);
      return torch::multinomial(local_probs, num_samples, replace) + start;
    };
  } else {
    if (replace) {
      pick_fn = [](int64_t start, int64_t end, int64_t num_samples) {
        return torch::randint(
            start, end,
            {
                num_samples,
            });
      };
    } else {
      pick_fn = [](int64_t start, int64_t end, int64_t num_samples) {
        auto perm = torch::randperm(end - start) + start;
        return perm.slice(0, 0, num_samples);
      };
    }
  }
  return pick_fn;
}

std::tuple<TensorList, TensorList> SampleEtypeNeighbors(
    const CSCPtr graph, torch::Tensor seed_nodes,
    const std::vector<int64_t>& fanouts, bool replace, bool require_eids,
    const torch::optional<torch::Tensor>& probs) {
  std::cout << "here" << std::endl;
  TORCH_CHECK(
      graph->IsHeterogeneous(),
      "SampleNeighborsEType only work with heterogeneous graph")
  TORCH_CHECK(
      fanouts.size() == graph->EdgeTypes().size(),
      "The length of Fanouts and edge type should be equal.")

  const int64_t num_etypes = fanouts.size();
  const int64_t num_nodes = seed_nodes.size(0);

  int64_t fanout_value = fanouts[0];
  bool same_fanout = std::all_of(
      fanouts.begin(), fanouts.end(),
      [fanout_value](auto elem) { return elem == fanout_value; });

  if (num_nodes == 0 || (same_fanout && fanout_value == 0)) {
    // Empty graph
    return std::tuple<TensorList, TensorList>();
  } else {
    auto pick_fn = GetRangePickFn(probs, replace);
    TensorList picked_rows, picked_cols, picked_eids;
    std::tie(picked_rows, picked_cols, picked_eids) = RowWisePickPerEtype(
        graph, seed_nodes, fanouts, probs, require_eids, replace, pick_fn);
    TensorList induced_coos(num_etypes);
    for (int64_t i = 0; i < num_etypes; i++) {
      // Note the graph is csc, so row and col should be reversed.
      induced_coos[i] = torch::stack({picked_cols[i], picked_rows[i]});
    }
    return std::tuple<TensorList, TensorList>(induced_coos, picked_eids);
  }
}

}  // namespace sampling
}  // namespace graphbolt
