/**
 *  Copyright (c) 2023 by Contributors
 * @file columnwise_pick.cc
 * @brief Contains the methods implementation for column wise pick.
 */

#include "./columnwise_pick.h"

namespace graphbolt {
namespace sampling {

inline torch::Tensor UniformPickWithReplace(
    int64_t start, int64_t end, int64_t num_samples) {
  return torch::randint(
      start, end,
      {
          num_samples,
      });
}

inline torch::Tensor UniformPick(
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
        auto choosed = replace
                           ? UniformPickWithReplace(0, true_num, num_samples)
                           : UniformPick(0, true_num, num_samples);
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
    pick_fn = replace ? UniformPickWithReplace : UniformPick;
  }
  return pick_fn;
}

torch::Tensor Pick(
    int64_t off, int64_t len, bool replace,
    const torch::optional<torch::Tensor>& probs,
    const torch::TensorOptions& options, int64_t num_pick,
    RangePickFn pick_fn) {
  torch::Tensor picked_indices;
  if ((num_pick == -1) || (len <= num_pick && !replace)) {
    // Fast path.
    picked_indices = torch::arange(off, off + len, options);
    if (probs.has_value()) {
      auto mask_tensor = probs.value().slice(0, off, off + len) > 0;
      picked_indices = torch::masked_select(picked_indices, mask_tensor);
    }
  } else {
    picked_indices = pick_fn(off, off + len, num_pick);
  }
  return picked_indices;
}

torch::Tensor PickEtype(
    int64_t off, int64_t len, bool replace,
    const torch::optional<torch::Tensor>& probs,
    const torch::TensorOptions& options, const torch::Tensor& type_per_edge,
    const std::vector<int64_t>& num_picks, RangePickFn pick_fn) {
  TensorList pick_indices_per_etype(
      num_picks.size(), torch::tensor({}, options));
  for (int64_t r = off, l = off; r < off + len; l = r) {
    auto cur_et = type_per_edge[r].item<int64_t>();
    auto cur_num_pick = num_picks[cur_et];
    while (r < off + len && type_per_edge[r].item<int64_t>() == cur_et) r++;
    // Do sampling for one etype.
    if (cur_num_pick != 0)
      pick_indices_per_etype[cur_et] =
          Pick(l, r - l, replace, probs, options, cur_num_pick, pick_fn);
  }
  return torch::cat(pick_indices_per_etype, 0);
}

c10::intrusive_ptr<SampledSubgraph> ColumnWisePick(
    const CSCPtr graph, const torch::Tensor& columns,
    const std::vector<int64_t>& num_picks,
    const torch::optional<torch::Tensor>& probs, bool return_eids, bool replace,
    bool consider_etype, RangePickFn pick_fn) {
  const auto indptr = graph->CSCIndptr();
  const auto indices = graph->Indices();
  const auto type_per_edge = graph->TypePerEdge();
  const int64_t num_columns = columns.size(0);

  // Don't do initialization here as it's very time-consuming, but make sure no
  // elements inside is undefined when using cat.
  TensorList picked_indices_per_column(num_columns);
  torch::Tensor picked_num_per_column =
      torch::zeros({num_columns + 1}, indptr.options());

  torch::parallel_for(
      0, num_columns, kDefaultPickGrainSize, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) {
          const auto cid = columns[i].item<int64_t>();
          TORCH_CHECK(
              cid >= 0 && cid < graph->NumNodes(),
              "Seed nodes should be in range [0, num_nodes)");
          const auto off = indptr[cid].item<int64_t>();
          const auto len = indptr[cid + 1].item<int64_t>() - off;

          if (len == 0) {
            // Init, otherwise cat will crash.
            picked_indices_per_column[i] = torch::tensor({}, indptr.options());
            continue;
          }

          torch::Tensor picked_indices_this_column;
          if (consider_etype) {
            picked_indices_this_column = PickEtype(
                off, len, replace, probs, indptr.options(),
                type_per_edge.value(), num_picks, pick_fn);
          } else {
            picked_indices_this_column = Pick(
                off, len, replace, probs, indptr.options(), num_picks[0],
                pick_fn);
          }

          picked_indices_per_column[i] = picked_indices_this_column;
          picked_num_per_column[i + 1] = picked_indices_this_column.size(0);
        }
      });  // End of the thread.

  // Get result csc indptr.
  torch::Tensor res_indptr = torch::cumsum(picked_num_per_column, 0);

  torch::Tensor picked_indices = torch::cat(picked_indices_per_column);
  torch::Tensor picked_rows = torch::index_select(indices, 0, picked_indices);

  torch::optional<torch::Tensor> picked_etypes = torch::nullopt;
  if (consider_etype)
    picked_etypes =
        torch::index_select(type_per_edge.value(), 0, picked_indices);
  torch::optional<torch::Tensor> picked_eids = torch::nullopt;
  if (return_eids) picked_eids = std::move(picked_indices);

  return c10::make_intrusive<SampledSubgraph>(
      res_indptr, picked_rows, columns, torch::nullopt, picked_eids,
      picked_etypes);
}

}  // namespace sampling
}  // namespace graphbolt
