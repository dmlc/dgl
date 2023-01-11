/**
 *  Copyright (c) 2022 by Contributors
 * @file elementwise_op.cc
 * @brief DGL C++ sparse elementwise operator implementation.
 */
// clang-format off
#include <sparse/dgl_headers.h>
// clang-format on

#include <sparse/elementwise_op.h>
#include <sparse/sparse_matrix.h>
#include <torch/script.h>

#include <memory>

#include "./utils.h"

namespace dgl {
namespace sparse {

c10::intrusive_ptr<SparseMatrix> SpSpAdd(
    const c10::intrusive_ptr<SparseMatrix>& A,
    const c10::intrusive_ptr<SparseMatrix>& B) {
  ElementwiseOpSanityCheck(A, B);
  torch::Tensor sum;
  {
    // To reduce peak memory usage.
    auto torch_A = COOToTorchCOO(A->COOPtr(), A->value());
    auto torch_B = COOToTorchCOO(B->COOPtr(), B->value());
    sum = torch_A + torch_B;
  }
  sum = sum.coalesce();
  auto indices = sum.indices();
  auto row = indices[0];
  auto col = indices[1];
  return SparseMatrix::FromCOO(row, col, sum.values(), A->shape());
}

}  // namespace sparse
}  // namespace dgl
