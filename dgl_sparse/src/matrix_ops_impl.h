/**
 *  Copyright (c) 2023 by Contributors
 * @file matrix_ops_impl.h
 * @brief DGL C++ sparse matrix operator implementations.
 */
#ifndef DGL_SPARSE_MATRIX_OPS_IMPL_H_
#define DGL_SPARSE_MATRIX_OPS_IMPL_H_

#include <sparse/sparse_format.h>
#include <sparse/sparse_matrix.h>

#include <tuple>
#include <vector>

#include "./utils.h"

namespace dgl {
namespace sparse {

template <c10::DeviceType XPU, typename IdType, typename ValType>
std::tuple<c10::intrusive_ptr<SparseMatrix>, torch::optional<torch::Tensor>>
CompactImpl(
    const c10::intrusive_ptr<SparseMatrix>& mat, int64_t dim,
    torch::optional<torch::Tensor> leading_indices) {
  torch::Tensor row, col;
  auto coo = mat->COOTensors();
  if (dim == 0)
    std::tie(row, col) = coo;
  else
    std::tie(col, row) = coo;

  //// [ For debug and delete soon ]
  // auto row_acc = row.accessor<IdType, 1>();
  // auto col_acc = col.accessor<IdType, 1>();
  // printf("COO format\n");
  // for (int i = 0; i < row.size(-1); i++) printf("%ld ", row_acc[i]);
  // printf("\n");
  // for (int i = 0; i < col.size(-1); i++) printf("%ld ", col_acc[i]);
  // printf("\n");

  torch::Tensor sort_idx, rev_sort_idx;
  std::tie(row, sort_idx) = row.sort(-1);
  rev_sort_idx = std::get<1>(sort_idx.sort(-1));

  torch::Tensor uniqued, uniq_idx;
  int64_t n_leading_indices = 0;
  if (leading_indices.has_value()) {
    n_leading_indices = leading_indices.value().numel();
    std::tie(uniqued, uniq_idx) =
        torch::_unique(torch::cat({leading_indices.value(), row}), false, true);
  } else {
    std::tie(uniqued, uniq_idx) = torch::_unique(row, false, true);
  }
  auto n_uniqued = uniqued.numel();

  auto new_row =
      torch::arange(n_uniqued - 1, -1, -1)
          .index_select(
              0, uniq_idx.slice(
                     0, n_leading_indices, n_leading_indices + row.size(-1)))
          .index_select(0, rev_sort_idx);

  if (dim == 0) {
    auto ret = SparseMatrix::FromCOO(
        torch::stack({new_row, col}, 0), mat->value(),
        std::vector<int64_t>{n_uniqued, mat->shape()[1]});
    auto ret_idx = torch::optional<torch::Tensor>(uniqued.flip(-1));
    return {ret, ret_idx};
  } else {
    auto ret = SparseMatrix::FromCOO(
        torch::stack({col, new_row}, 0), mat->value(),
        std::vector<int64_t>{mat->shape()[0], n_uniqued});
    auto ret_idx = torch::optional<torch::Tensor>(uniqued.flip(-1));
    return {ret, ret_idx};
  }
}

}  // namespace sparse
}  // namespace dgl

#endif  // DGL_SPARSE_MATRIX_OPS_IMPL_H_
