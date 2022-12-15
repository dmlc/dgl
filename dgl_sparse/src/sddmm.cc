/**
 *  Copyright (c) 2022 by Contributors
 * @file sddmm.cc
 * @brief DGL C++ sparse SDDMM operator implementation.
 */
// clang-format off
#include <sparse/dgl_headers.h>
// clang-format on

#include <sparse/sparse_matrix.h>
#include <torch/script.h>

#include "./utils.h"

namespace dgl {
namespace sparse {

torch::Tensor SDDMMImpl(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat, torch::Tensor mat1,
    torch::Tensor mat2_tr) {
  SparseFormat fmt;
  // Use CSR if the sparse matrix has CSR or does not have COO. Otherwise use
  // COO.
  if (sparse_mat->HasCSR() || !sparse_mat->HasCOO()) {
    fmt = SparseFormat::kCSR;
  } else {
    fmt = SparseFormat::kCOO;
  }
  if (mat2_tr.dim() == 1) {
    mat1 = mat1.view({-1, 1});
    mat2_tr = mat2_tr.view({-1, 1});
  }
  int64_t out_row = sparse_mat->nnz();
  auto shape = std::vector<int64_t>({out_row});
  auto ret = torch::zeros(shape, mat1.options());
  const std::string op = "dot";
  auto dgl_mat1 = TorchTensorToDGLArray(mat1);
  auto dgl_mat2_tr = TorchTensorToDGLArray(mat2_tr);
  auto dgl_ret = TorchTensorToDGLArray(ret);
  if (fmt == SparseFormat::kCSR) {
    auto csr = CSRToOldDGLCSR(sparse_mat->CSRPtr());
    aten::CSRSDDMM(
        op.c_str(), csr, dgl_mat1, dgl_mat2_tr, dgl_ret, 0 /* Lhs target: u */,
        2 /* rhs target: v */);
  } else {  // COO
    auto coo = COOToOldDGLCOO(sparse_mat->COOPtr());
    aten::COOSDDMM(
        op.c_str(), coo, dgl_mat1, dgl_mat2_tr, dgl_ret, 0 /* Lhs target: u */,
        2 /* rhs target: v */);
  }
  return ret;
}

}  // namespace sparse
}  // namespace dgl
