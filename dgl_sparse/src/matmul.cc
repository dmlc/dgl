/**
 *  Copyright (c) 2022 by Contributors
 * @file matmul.cc
 * @brief DGL sparse matrix multiplication functions.
 */
#include "./matmul.h"

// clang-format off
#include <sparse/dgl_headers.h>
// clang-format on

#include <sparse/sparse_matrix.h>
#include <torch/script.h>

#include "./utils.h"

namespace dgl {
namespace sparse {

torch::Tensor SpMMNoAutoGrad(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat,
    torch::Tensor sparse_val, torch::Tensor dense_mat, bool transpose_sparse) {
  const std::string op = "mul";
  const std::string reduce = "sum";
  const int64_t out_row =
      transpose_sparse ? sparse_mat->shape()[1] : sparse_mat->shape()[0];
  const std::vector<int64_t> shape = {out_row, dense_mat.size(1)};

  auto ret = torch::zeros(shape, dense_mat.options());
  auto dgl_sparse_val = TorchTensorToDGLArray(sparse_val);
  auto dgl_dense_mat = TorchTensorToDGLArray(dense_mat);
  auto dgl_ret = TorchTensorToDGLArray(ret);
  if (!transpose_sparse) {
    // The format for calculation will be chosen in the following order: CSR,
    // COO. CSR is created if the sparse matrix only has CSC format.
    if (sparse_mat->HasCSR() || !sparse_mat->HasCOO()) {
      // sparse_mat->CSRPtr() will implicitly convert CSC to CSR format if CSR
      // does not exist.
      auto csr = CSRToOldDGLCSR(sparse_mat->CSRPtr());
      aten::CSRSpMM(
          op.c_str(), reduce.c_str(), csr, dgl_dense_mat, dgl_sparse_val,
          dgl_ret, {});
    } else {  // COO
      // Use the reverse order of aten::COOSpMM because it calculates A^T @ X.
      auto coo = COOToOldDGLCOO(sparse_mat->COOPtr());
      coo = aten::COOTranspose(coo);
      aten::COOSpMM(
          op.c_str(), reduce.c_str(), coo, dgl_dense_mat, dgl_sparse_val,
          dgl_ret, {});
    }
  } else {  // transpose_sparse
    // The format for calculation will be chosen in the following order: CSC,
    // COO. CSC is created if the sparse matrix only has CSR format.
    if (sparse_mat->HasCSC() || !sparse_mat->HasCOO()) {
      // sparse_mat->CSCPtr() will implicitly convert CSR to CSC format if CSR
      // does not exist.
      // Use CSC in DGL's CSRSpMM is equivalent as computing A^T @ X.
      auto csc = CSRToOldDGLCSR(sparse_mat->CSCPtr());
      aten::CSRSpMM(
          op.c_str(), reduce.c_str(), csc, dgl_dense_mat, dgl_sparse_val,
          dgl_ret, {});
    } else {  // COO
      // Use the reverse order of aten::COOSpMM because it calculates A^T @ X.
      auto coo = COOToOldDGLCOO(sparse_mat->COOPtr());
      aten::COOSpMM(
          op.c_str(), reduce.c_str(), coo, dgl_dense_mat, dgl_sparse_val,
          dgl_ret, {});
    }
  }
  return ret;
}

torch::Tensor SDDMMNoAutoGrad(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat, torch::Tensor mat1,
    torch::Tensor mat2_tr) {
  const int64_t out_row = sparse_mat->nnz();
  const std::vector<int64_t> shape({out_row});
  auto ret = torch::zeros(shape, mat1.options());
  const std::string op = "dot";
  auto dgl_mat1 = TorchTensorToDGLArray(mat1);
  auto dgl_mat2_tr = TorchTensorToDGLArray(mat2_tr);
  auto dgl_ret = TorchTensorToDGLArray(ret);
  // The format for calculation will be chosen in the following order: CSR,
  // COO. CSR is created if the sparse matrix only has CSC format.
  if (sparse_mat->HasCSR() || !sparse_mat->HasCOO()) {
    // sparse_mat->CSRPtr() will implicitly convert CSC to CSR format if CSR
    // does not exist.
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
