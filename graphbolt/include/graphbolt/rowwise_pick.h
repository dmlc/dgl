/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/include/rowwise_pick.h
 * @brief Contains the function definition for RowWisePickPerEtype.
 */

#ifndef GRAPHBOLT_ROWWISE_PICK_H_
#define GRAPHBOLT_ROWWISE_PICK_H_

#include <graphbolt/neighbor.h>

namespace graphbolt {
namespace sampling {

static constexpr int kDefaultPickGrainSize = 100;

/**
 * @brief Performs row-wise picking based on the given parameters.
 *
 * @param graph The pointer to a csc sampling graph.
 * @param rows The tensor containing the row indices.
 * @param num_picks The vector containing the number of picks per edge type.
 * @param probs Optional tensor containing probabilities for picking.
 * @param require_eids Boolean indicating if edge IDs need to be returned. The
 * last TensorList in the tuple is this value when required.
 * @param replace Boolean indicating if picking is done with replacement.
 * @param pick_fn The function used for picking.
 * @return A tuple containing the picked rows, picked columns, and picked edge
 * IDs (if required).
 */
std::tuple<TensorList, TensorList, TensorList> RowWisePickPerEtype(
    const CSCPtr graph, const torch::Tensor& rows,
    const std::vector<int64_t>& num_picks,
    const torch::optional<torch::Tensor>& probs, bool require_eids,
    bool replace, RangePickFn pick_fn) {
  const auto indptr = graph->CSCIndptr();
  const auto indices = graph->Indices();
  const auto type_per_edge = graph->TypePerEdge();
  const int64_t num_rows = rows.size(0);
  const int64_t num_etypes = num_picks.size();
  std::vector<TensorList> picked_rows_per_etype(
      num_etypes, TensorList(num_rows));
  std::vector<TensorList> picked_cols_per_etype(
      num_etypes, TensorList(num_rows));
  std::vector<TensorList> picked_eids_per_etype;
  if (require_eids) {
    picked_eids_per_etype.resize(num_etypes, TensorList(num_rows));
  }
  int64_t min_num_picks = -1;
  bool pick_all = true;
  for (auto num_pick : num_picks) {
    if (num_picks[i] != -1)  {
      if (min_num_picks == -1 || num_picks[i] < min_num_pick)
        min_num_picks = num_picks[i];
      pick_all = false;
    }
  }
  // TODO: all -1 fast path?

  const bool has_probs = probs.has_value();
  torch::parallel_for(
      0, num_rows, kDefaultPickGrainSize, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) {
          const auto rid = rows[i].item<int>();
          TORCH_CHECK(rid < num_rows);

          const auto off = indptr[rid].item<int>();
          const auto len = (indptr[rid + 1] - off).item<int>();

          if (len == 0) continue;

          // fast path
          if (len <= min_num_pick && !replace) {
              picked_indices = torch::arange(et_start, et_end);
              if (has_probs) {
                auto mask_tensor =
                    probs.value().slice(0, et_start, et_end) > 0;
                picked_indices =
                    torch::masked_select(picked_indices, mask_tensor);
              }
            } else {
              picked_indices = pick_fn(et_start, et_end, cur_num_pick);
            }
          } else {
            auto cur_et = type_per_edge[off].item<int>();
            auto cur_num_pick = num_picks[cur_et];
            int64_t et_len = 1;
            for (int64_t j = 0; j < len; ++j) {
              TORCH_CHECK(
                  j + 1 == len ||
                      cur_et <= type_per_edge[off + j + 1].item<int>(),
                  "Edge type is not sorted. Please sort in advance");

              if ((j + 1 == len) ||
                  cur_et != type_per_edge[off + j + 1].item<int>()) {
                // 1 end of the current etype
                // 2 end of the row
                // random pick for current etype
                if (cur_num_pick != 0) {
                  torch::Tensor picked_indices;
                  int64_t et_end = off + j + 1;
                  int64_t et_start = et_end - et_len;
                  if ((cur_num_pick == -1) ||
                      (et_len <= cur_num_pick && !replace)) {
                    // fast path
                    picked_indices = torch::arange(et_start, et_end);
                    if (has_probs) {
                      auto mask_tensor =
                          probs.value().slice(0, et_start, et_end) > 0;
                      picked_indices =
                          torch::masked_select(picked_indices, mask_tensor);
                    }
                  } else {
                    picked_indices = pick_fn(et_start, et_end, cur_num_pick);
                  }

                  int64_t picked_num = picked_indices.size(0);
                  picked_rows_per_etype[cur_et][i] = torch::full({picked_num}, rid);
                  picked_cols_per_etype[cur_et][i] = indices[picked_indices];
                  if (require_eids)
                    picked_eids_per_etype[cur_et][i] = picked_indices;
                  if (j + 1 == len) break;
                }
                cur_et = type_per_edge[off + j + 1].item<int>();
                et_len = 1;
                cur_num_pick = num_picks[cur_et];
              } else {
                et_len++;
              }
            }
          }
        }
      });

  TensorList picked_rows(num_etypes);
  TensorList picked_cols(num_etypes);
  TensorList picked_eids;
  if (require_eids) picked_eids.resize(num_etypes);

  for (int i = 0; i < num_etypes; i++) {
    picked_rows[i] = torch::cat(picked_rows_per_etype[i]);
    picked_cols[i] = torch::cat(picked_cols_per_etype[i]);
    if (require_eids) picked_eids[i] = torch::cat(picked_eids_per_etype[i]);
  }

  return std::tuple<TensorList, TensorList, TensorList>(
      picked_rows, picked_cols, picked_eids);
}

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_ROWWISE_PICK_H_
