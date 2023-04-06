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
  auto torch_A = COOToTorchCOO(A->COOPtr(), A->value());
  auto torch_B = COOToTorchCOO(B->COOPtr(), B->value());
  auto sum = (torch_A + torch_B).coalesce();
  return SparseMatrix::FromCOO(sum.indices(), sum.values(), A->shape());
}

}  // namespace sparse
}  // namespace dgl
