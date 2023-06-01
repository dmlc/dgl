/**
 *  Copyright (c) 2023 by Contributors
 * @file columnwise_pick.cc
 * @brief Contains the methods implementation for column wise pick.
 */

#include "columnwise_pick.h"

using namespace graphbolt::utils;

namespace graphbolt {
namespace sampling {

c10::intrusive_ptr<SampledSubgraph> ColumnWisePick(
    const CSCPtr graph, const torch::Tensor& columns,
    const torch::Tensor& num_picks, const torch::optional<torch::Tensor>& probs,
    bool return_eids, bool replace, RangePickFn pick_fn) {
  const auto indptr = graph->CSCIndptr();
  const auto indices = graph->Indices();
  const auto type_per_edge = graph->TypePerEdge().value();
  const int64_t num_columns = columns.size(0);
  const int64_t num_etypes = num_picks.size(0);
  std::vector<int64_t> picked_num_per_row(num_columns + 1);
  auto num_picks_accessor = num_picks.accessor<int64_t, 1>();

  auto type_per_edge_accessor = type_per_edge.accessor<int64_t, 1>();
  const bool has_probs = probs.has_value();
  // Use torch get_num_threads.
  auto thread_num = compute_num_threads(0, num_columns, kDefaultPickGrainSize);
  std::vector<size_t> block_offset(thread_num + 1, 0);
  TensorList picked_cols_per_thread(thread_num);
  TensorList picked_etypes_per_thread(thread_num);
  TensorList picked_eids_per_thread;
  if (return_eids) {
    picked_eids_per_thread.resize(
        thread_num, torch::tensor({}, indptr.options()));
  }
  torch::parallel_for(
      0, num_columns, kDefaultPickGrainSize, [&](size_t b, size_t e) {
        int64_t sampled_num_this_thread = 0;
        TensorList picked_indices_this_thread;
        for (size_t i = b; i < e; ++i) {
          const auto rid = columns[i].item<int64_t>();
          TORCH_CHECK(
              rid >= 0 && rid < graph->NumNodes(),
              "Seed nodes should be in range [0, num_nodes)");
          const auto off = indptr[rid].item<int64_t>();
          const auto len = indptr[rid + 1].item<int64_t>() - off;

          if (len == 0) continue;

          torch::Tensor picked_indices_this_row;
          TensorList pick_indices_per_etype(
              num_etypes, torch::tensor({}, indptr.options()));
          auto cur_et = type_per_edge_accessor[off];
          auto cur_num_pick = num_picks_accessor[cur_et];
          int64_t et_len = 1;

          for (int64_t j = 0; j < len; ++j) {
            TORCH_CHECK(
                j + 1 == len || cur_et <= type_per_edge_accessor[off + j + 1],
                "Edge type is not sorted. Please sort in advance");

            if ((j + 1 == len) ||
                cur_et != type_per_edge_accessor[off + j + 1]) {
              // 1 end of the current etype.
              // 2 end of the row.
              // Random pick for current etype.
              if (cur_num_pick != 0) {
                torch::Tensor picked_indices;
                int64_t et_end = off + j + 1;
                int64_t et_start = et_end - et_len;
                if ((cur_num_pick == -1) ||
                    (et_len <= cur_num_pick && !replace)) {
                  // Fast path.
                  picked_indices =
                      torch::arange(et_start, et_end, indptr.options());
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

              if (j + 1 == len) break;

              cur_et = type_per_edge_accessor[off + j + 1];
              et_len = 1;
              cur_num_pick = num_picks_accessor[cur_et];
            } else {
              et_len++;
            }
          }
          picked_indices_this_row = torch::cat(pick_indices_per_etype, 0);

          sampled_num_this_thread += picked_indices_this_row.size(0);
          picked_indices_this_thread.emplace_back(picked_indices_this_row);
          picked_num_per_row[i + 1] = sampled_num_this_thread;
        }  // End of the one row pick.

        auto thread_id = torch::get_thread_num();
        auto picked_indices = torch::cat(picked_indices_this_thread);
        picked_cols_per_thread[thread_id] =
            torch::index_select(indices, 0, picked_indices);
        picked_etypes_per_thread[thread_id] =
            torch::index_select(type_per_edge, 0, picked_indices);
        if (return_eids) picked_eids_per_thread[thread_id] = picked_indices;
        block_offset[thread_id + 1] = sampled_num_this_thread;
      });  // End of the thread.

  if (thread_num > 1) {
    // Get ExclusiveSum of each block.
    std::partial_sum(
        block_offset.begin() + 1, block_offset.end(), block_offset.begin() + 1);

    torch::parallel_for(
        0, num_columns, kDefaultPickGrainSize, [&](int64_t b, int64_t e) {
          auto tid = omp_get_thread_num();
          auto off = block_offset[tid];
          for (int64_t i = b; i < e; i++) {
            picked_num_per_row[i + 1] += off;
          }
        });
  }

  torch::Tensor picked_rows =
      torch::tensor(picked_num_per_row, {indices.dtype()});
  torch::Tensor picked_cols = torch::cat(picked_cols_per_thread);
  torch::Tensor picked_etypes = torch::cat(picked_etypes_per_thread);
  torch::optional<torch::Tensor> picked_eids = torch::nullopt;
  if (return_eids) picked_eids = torch::cat(picked_eids_per_thread);

  return c10::make_intrusive<SampledSubgraph>(
      picked_rows, picked_cols, columns, torch::nullopt, picked_eids,
      picked_etypes);
}

}  // namespace sampling
}  // namespace graphbolt
