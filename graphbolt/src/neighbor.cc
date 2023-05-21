/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/src/neighbor.cc
 * @brief Source file of neighbor sampling.
 */

#include <graphbolt/rowwise_pick.h>

#include <algorithm>

namespace graphbolt {
namespace sampling {

inline torch::Tensor UniformRangeWithRepeat(
    int64_t start, int64_t end, int64_t num_samples) {
  return torch::randint(
      start, end,
      {
          num_samples,
      });
}

inline torch::Tensor UniformRangeWithOutRepeat(
    int64_t start, int64_t end, int64_t num_samples) {
  auto perm = torch::randperm(end - start) + start;
  return perm.slice(0, 0, num_samples);
}

RangePickFn GetRangePickFn(
    const torch::optional<torch::Tensor>& probs, bool replace) {
  RangePickFn pick_fn;
  if (probs.has_value()) {
    if (probs.value().dtype() == torch::kBool) {
      pick_fn = [probs, replace](
                    int64_t start, int64_t end, int64_t num_samples) {
        auto local_probs = probs.value().slice(0, start, end);
        auto true_indices = local_probs.nonzero().view(-1);
        auto true_num = true_indices.size(0);
        auto choosed =
            replace ? UniformRangeWithRepeat(0, true_num, num_samples)
                    : UniformRangeWithOutRepeat(0, true_num, num_samples);
        return true_indices[choosed];
      };
    } else {
      pick_fn = [probs, replace](
                    int64_t start, int64_t end, int64_t num_samples) {
        auto local_probs = probs.value().slice(0, start, end);
        return torch::multinomial(local_probs, num_samples, replace) + start;
      };
    }
  } else {
    pick_fn = replace ? UniformRangeWithRepeat : UniformRangeWithOutRepeat;
  }
  return pick_fn;
}

std::tuple<torch::Tensor, torch::Tensor> SampleEtypeNeighbors(
    const CSCPtr graph, torch::Tensor seed_nodes,
    const std::vector<int64_t>& fanouts, bool replace, bool require_eids,
    const torch::optional<torch::Tensor>& probs) {
  TORCH_CHECK(
      graph->IsHeterogeneous(),
      "SampleNeighborsEType only work with heterogeneous graph")
  TORCH_CHECK(
      fanouts.size() == graph->EdgeTypes().size(),
      "The length of Fanouts and edge type should be equal.")

  const int64_t num_nodes = seed_nodes.size(0);

  int64_t fanout_value = fanouts[0];
  bool same_fanout = std::all_of(
      fanouts.begin(), fanouts.end(),
      [fanout_value](auto elem) { return elem == fanout_value; });

  if (num_nodes == 0 || (same_fanout && fanout_value == 0)) {
    // Empty graph
    return std::tuple<torch::Tensor, torch::Tensor>();
  } else {
    auto pick_fn = GetRangePickFn(probs, replace);
    torch::Tensor picked_rows, picked_cols, picked_etypes, picked_eids;
    std::tie(picked_rows, picked_cols, picked_etypes, picked_eids) =
        RowWisePickPerEtype(
            graph, seed_nodes, fanouts, probs, require_eids, replace, pick_fn);
    torch::Tensor induced_coos;
    // Note the graph is csc, so row and col should be reversed.
    induced_coos = torch::stack({picked_cols, picked_rows, picked_etypes});
    return std::make_tuple(induced_coos, picked_eids);
  }
}

}  // namespace sampling
}  // namespace graphbolt
