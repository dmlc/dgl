/**
 *  Copyright (c) 2023 by Contributors
 * @file cpu/matrix_ops_impl.cc
 * @brief DGL C++ matrix operators.
 */
#include "./matrix_ops_impl.h"

namespace dgl {
namespace sparse {

std::tuple<torch::Tensor, torch::Tensor> CompactId(
    const torch::Tensor &row,
    const torch::optional<torch::Tensor> &leading_indices) {
  torch::Tensor sort_row, sort_idx;
  std::tie(sort_row, sort_idx) = row.sort(-1);
  torch::Tensor rev_sort_idx = torch::empty_like(sort_idx);
  rev_sort_idx.index_put_({sort_idx}, torch::arange(0, sort_idx.numel()));

  torch::Tensor uniqued, uniq_idx;
  int64_t n_leading_indices = 0;
  if (leading_indices.has_value()) {
    n_leading_indices = leading_indices.value().numel();
    std::tie(uniqued, uniq_idx) = torch::_unique(
        torch::cat({leading_indices.value(), sort_row}), false, true);
  } else {
    std::tie(uniqued, uniq_idx) = torch::_unique(sort_row, false, true);
  }

  auto new_row =
      torch::arange(uniqued.numel() - 1, -1, -1)
          .index_select(
              0, uniq_idx.slice(
                     0, n_leading_indices, n_leading_indices + row.size(-1)))
          .index_select(0, rev_sort_idx);
  return {new_row, uniqued};
}

}  // namespace sparse
}  // namespace dgl
