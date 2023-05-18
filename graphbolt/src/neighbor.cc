/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/src/neighbor.cc
 * @brief Source file of neighbor sampling.
 */

#include "rowwise_pick.h"
#include <algorithm>

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

std::tuple<TensorList, TensorList, TensorList> SampleEtypeNeighbors(
    const CSRPtr& graph, const torch::Tensor seed_nodes,
    const std::vector<int64_t>& fanouts,
    const torch::optional<torch::Tensor>& probs, bool replace,
    bool require_eids) {
  std::cout << "here" << std::endl;
  TORCH_CHECK(
      fanouts.size() == 1 || fanouts.size() == graph->GetEdgeTypes().size())

  const int64_t num_nodes = seed_nodes.size(0);

  int64_t fanout_value = fanouts[0];
  bool same_fanout = std::all_of(fanouts.begin(), fanouts.end(), [fanout_value](auto
  elem) {
    return elem == fanout_value;
  });

  if (fanouts.size() == 1) {
    // do homogeneous sampling
  } else {
    if (num_nodes == 0 || (same_fanout && fanout_value == 0)) {
      // Empty graph

    } else {
      auto pick_fn = GetRangePickFn(probs, replace);
      return RowWisePickPerEtype(graph, seed_nodes, fanouts, probs, require_eids, replace, pick_fn);
    }
  }

  return std::tuple<TensorList, TensorList, TensorList>();
}

}  // namespace sampling
}  // namespace graphbolt
