/**
 *  Copyright (c) 2022 by Contributors
 * @file sparse_matrix_coalesce.cc
 * @brief Operators related to sparse matrix coalescing.
 */
// clang-format off
#include <sparse/dgl_headers.h>
// clang-format on

#include <sparse/sparse_matrix.h>

#include "./utils.h"

namespace dgl {
namespace sparse {

c10::intrusive_ptr<SparseMatrix> SparseMatrix::Coalesce() {
  auto torch_coo = COOToTorchCOO(this->COOPtr(), this->value());
  auto coalesced_coo = torch_coo.coalesce();
  torch::Tensor indices = coalesced_coo.indices();
  torch::Tensor row = indices[0];
  torch::Tensor col = indices[1];
  return SparseMatrix::FromCOO(row, col, coalesced_coo.values(), this->shape());
}

bool SparseMatrix::HasDuplicate() {
  aten::CSRMatrix dgl_csr;
  // The format for calculation will be chosen in the following order: CSR,
  // CSC. CSR is created if the sparse matrix only has CSC format.
  if (HasCSR() || !HasCSC()) {
    dgl_csr = CSRToOldDGLCSR(CSRPtr());
  } else {
    dgl_csr = CSRToOldDGLCSR(CSCPtr());
  }
  return aten::CSRHasDuplicate(dgl_csr);
}

}  // namespace sparse
}  // namespace dgl
