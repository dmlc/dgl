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
  auto fmt = FindAnyExistingFormat(A, B);
  auto value = A->value() + B->value();
  ElementwiseOpSanityCheck(A, B);
  if (fmt == SparseFormat::kCOO) {
    return SparseMatrix::FromCOO(A->COOPtr(), value, A->shape());
  } else if (fmt == SparseFormat::kCSR) {
    return SparseMatrix::FromCSR(A->CSRPtr(), value, A->shape());
  } else {
    return SparseMatrix::FromCSC(A->CSCPtr(), value, A->shape());
  }
}

}  // namespace sparse
}  // namespace dgl
