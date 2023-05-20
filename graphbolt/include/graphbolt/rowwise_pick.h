/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/rowwise_pick.h
 * @brief Contains the function definition for RowWisePickPerEtype.
 */

#ifndef GRAPHBOLT_ROWWISE_PICK_H_
#define GRAPHBOLT_ROWWISE_PICK_H_

#include <graphbolt/neighbor.h>

namespace graphbolt {
namespace sampling {

using TensorList = std::vector<torch::Tensor>;
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
 * @return A tuple containing the picked rows, picked columns, picked etypes,
 * and picked edge IDs (if required).
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RowWisePickPerEtype(
    const CSCPtr graph, const torch::Tensor& rows,
    const std::vector<int64_t>& num_picks,
    const torch::optional<torch::Tensor>& probs, bool require_eids,
    bool replace, RangePickFn pick_fn) {
  const auto indptr = graph->CSCIndptr();
  const auto indices = graph->Indices();
  const auto type_per_edge = graph->TypePerEdge();
  const int64_t num_rows = rows.size(0);
  const int64_t num_etypes = num_picks.size();
  TensorList picked_rows_per_row(num_rows);
  TensorList picked_cols_per_row(num_rows);
  TensorList picked_etypes_per_row(num_rows);
  TensorList picked_eids_per_row;
  if (require_eids) {
    picked_eids_per_row.resize(num_rows);
  }
  int64_t min_num_pick = -1;
  bool pick_all = true;
  for (auto num_pick : num_picks) {
    if (num_pick != -1) {
      if (min_num_pick == -1 || num_pick < min_num_pick)
        min_num_pick = num_pick;
      pick_all = false;
    }
  }

  const bool has_probs = probs.has_value();
  torch::parallel_for(
      0, num_rows, kDefaultPickGrainSize, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) {
          const auto rid = rows[i].item<int>();
          TORCH_CHECK(rid < num_rows);

          const auto off = indptr[rid].item<int>();
          const auto len = (indptr[rid + 1] - off).item<int>();

          if (len == 0) continue;

          torch::Tensor picked_indices_row;
          // fast path
          if ((pick_all || len <= min_num_pick) && !replace) {
            picked_indices_row = torch::arange(off, off + len);
            if (has_probs) {
              auto mask_tensor = probs.value().slice(0, off, off + len) > 0;
              picked_indices_row =
                  torch::masked_select(picked_indices_row, mask_tensor);
            }
          } else {
            TensorList pick_indices_per_etype(num_etypes);
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
                torch::Tensor picked_indices;
                if (cur_num_pick != 0) {
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
                  pick_indices_per_etype[cur_et] = picked_indices;
                }

                if (j + 1 == len) {
                  picked_indices_row = torch::cat(pick_indices_per_etype, 0);
                  break;
                }
                cur_et = type_per_edge[off + j + 1].item<int>();
                et_len = 1;
                cur_num_pick = num_picks[cur_et];
              } else {
                et_len++;
              }
            }
          }
          int64_t picked_num = picked_indices_row.size(0);
          picked_rows_per_row[i] = torch::full({picked_num}, rid);
          picked_cols_per_row[i] = indices[picked_indices_row];
          picked_eids_per_row[i] = type_per_edge[picked_indices_row];
          if (require_eids) picked_eids_per_row[i] = picked_indices_row;
        }
      });

  torch::Tensor picked_rows = torch::cat(picked_rows_per_row);
  torch::Tensor picked_cols = torch::cat(picked_cols_per_row);
  torch::Tensor picked_etypes = torch::cat(picked_etypes_per_row);
  torch::Tensor picked_eids;
  if (require_eids) picked_eids = torch::cat(picked_eids_per_row);

  return std::make_tuple(picked_rows, picked_cols, picked_etypes, picked_eids);
}

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_ROWWISE_PICK_H_
