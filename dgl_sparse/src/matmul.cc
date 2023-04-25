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
  std::vector<int64_t> shape = {out_row, dense_mat.size(1)};
  // Batched SpMM
  if (sparse_val.dim() >= 2) {
    shape = {out_row, dense_mat.size(1), sparse_val.size(1)};
  }

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
  std::vector<int64_t> shape({out_row});
  // Batched SDDMM
  if (mat1.dim() >= 3) {
    shape.push_back(mat1.size(2));
    // (N, K, B) -> (N, B, K)
    mat1 = mat1.transpose(1, 2);
    // (M, K, B) -> (M, B, K)
    mat2_tr = mat2_tr.transpose(1, 2);
  }
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

torch::Tensor BroadcastOpNoAutoGrad(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat, torch::Tensor dense_mat,
    const std::string& op, int64_t dim) {
  auto sparse_val = sparse_mat->value();
  const int64_t out_row = sparse_mat->nnz();
  const std::vector<int64_t> shape({out_row, sparse_val.size(1)});
  auto ret = torch::zeros(shape, sparse_val.options());

  auto dgl_sparse_val = TorchTensorToDGLArray(sparse_val);
  auto dgl_dense_mat = TorchTensorToDGLArray(dense_mat);
  auto dgl_ret = TorchTensorToDGLArray(ret);
  // Setting dgl_rhs_target to 0 or 2 means using row or column coordinators
  // to access dgl_dense_mat for each edge, respectively.
  auto dgl_rhs_target = dim == 0 ? 2 : 0;

  // The format for calculation will be chosen in the following order: COO, CSR
  // . COO is created if the sparse matrix only has CSC format.
  if (sparse_mat->HasCOO() || !sparse_mat->HasCSR()) {
    // sparse_mat->COOPtr() will implicitly convert CSC to COO format if COO
    // does not exist.
    auto coo = COOToOldDGLCOO(sparse_mat->COOPtr());
    aten::COOSDDMM(
        op.c_str(), coo, dgl_sparse_val, dgl_dense_mat, dgl_ret,
        1 /* Lhs target: e */, dgl_rhs_target);
  } else {
    auto csr = CSRToOldDGLCSR(sparse_mat->CSRPtr());
    aten::CSRSDDMM(
        op.c_str(), csr, dgl_sparse_val, dgl_dense_mat, dgl_ret,
        1 /* Lhs target: e */, dgl_rhs_target);
  }
  return ret;
}

torch::Tensor BroadcastSubNoAutoGrad(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat, torch::Tensor dense_mat,
    int64_t dim) {
  return BroadcastOpNoAutoGrad(sparse_mat, dense_mat, "sub", dim);
}

torch::Tensor BroadcastDivNoAutoGrad(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat, torch::Tensor dense_mat,
    int64_t dim) {
  return BroadcastOpNoAutoGrad(sparse_mat, dense_mat, "div", dim);
}

torch::Tensor BroadcastMulNoAutoGrad(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat, torch::Tensor dense_mat,
    int64_t dim) {
  return BroadcastOpNoAutoGrad(sparse_mat, dense_mat, "mul", dim);
}

c10::intrusive_ptr<SparseMatrix> SpSpMMNoAutoGrad(
    const c10::intrusive_ptr<SparseMatrix>& lhs_mat, torch::Tensor lhs_val,
    const c10::intrusive_ptr<SparseMatrix>& rhs_mat, torch::Tensor rhs_val,
    bool lhs_transpose, bool rhs_transpose) {
  aten::CSRMatrix lhs_dgl_csr, rhs_dgl_csr;
  if (!lhs_transpose) {
    lhs_dgl_csr = CSRToOldDGLCSR(lhs_mat->CSRPtr());
  } else {
    lhs_dgl_csr = CSRToOldDGLCSR(lhs_mat->CSCPtr());
  }
  if (!rhs_transpose) {
    rhs_dgl_csr = CSRToOldDGLCSR(rhs_mat->CSRPtr());
  } else {
    rhs_dgl_csr = CSRToOldDGLCSR(rhs_mat->CSCPtr());
  }
  auto lhs_dgl_val = TorchTensorToDGLArray(lhs_val);
  auto rhs_dgl_val = TorchTensorToDGLArray(rhs_val);
  const int64_t ret_row =
      lhs_transpose ? lhs_mat->shape()[1] : lhs_mat->shape()[0];
  const int64_t ret_col =
      rhs_transpose ? rhs_mat->shape()[0] : rhs_mat->shape()[1];
  std::vector<int64_t> ret_shape({ret_row, ret_col});
  aten::CSRMatrix ret_dgl_csr;
  runtime::NDArray ret_val;
  std::tie(ret_dgl_csr, ret_val) =
      aten::CSRMM(lhs_dgl_csr, lhs_dgl_val, rhs_dgl_csr, rhs_dgl_val);
  return SparseMatrix::FromCSRPointer(
      CSRFromOldDGLCSR(ret_dgl_csr), DGLArrayToTorchTensor(ret_val), ret_shape);
}

}  // namespace sparse
}  // namespace dgl
