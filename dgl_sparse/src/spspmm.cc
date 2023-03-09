/**
 *  Copyright (c) 2022 by Contributors
 * @file spspmm.cc
 * @brief DGL C++ sparse SpSpMM operator implementation.
 */

#include <sparse/sddmm.h>
#include <sparse/sparse_matrix.h>
#include <sparse/spspmm.h>
#include <torch/script.h>

#include "./matmul.h"
#include "./utils.h"

namespace dgl {
namespace sparse {

using namespace torch::autograd;

class SpSpMMAutoGrad : public Function<SpSpMMAutoGrad> {
 public:
  static variable_list forward(
      AutogradContext* ctx, c10::intrusive_ptr<SparseMatrix> lhs_mat,
      torch::Tensor lhs_val, c10::intrusive_ptr<SparseMatrix> rhs_mat,
      torch::Tensor rhs_val);

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs);
};

void _SpSpMMSanityCheck(
    const c10::intrusive_ptr<SparseMatrix>& lhs_mat,
    const c10::intrusive_ptr<SparseMatrix>& rhs_mat) {
  const auto& lhs_shape = lhs_mat->shape();
  const auto& rhs_shape = rhs_mat->shape();
  TORCH_CHECK(
      lhs_shape[1] == rhs_shape[0],
      "SpSpMM: the second dim of lhs_mat should be equal to the first dim ",
      "of the second matrix");
  TORCH_CHECK(
      lhs_mat->value().dim() == 1,
      "SpSpMM: the value shape of lhs_mat should be 1-D");
  TORCH_CHECK(
      rhs_mat->value().dim() == 1,
      "SpSpMM: the value shape of rhs_mat should be 1-D");
  TORCH_CHECK(
      lhs_mat->device() == rhs_mat->device(),
      "SpSpMM: lhs_mat and rhs_mat should be on the same device");
  TORCH_CHECK(
      lhs_mat->dtype() == rhs_mat->dtype(),
      "SpSpMM: lhs_mat and rhs_mat should have the same dtype");
  TORCH_CHECK(
      !lhs_mat->HasDuplicate(),
      "SpSpMM does not support lhs_mat with duplicate indices. ",
      "Call A = A.coalesce() to dedup first.");
  TORCH_CHECK(
      !rhs_mat->HasDuplicate(),
      "SpSpMM does not support rhs_mat with duplicate indices. ",
      "Call A = A.coalesce() to dedup first.");
}

// Mask select value of `mat` by `sub_mat`.
torch::Tensor _CSRMask(
    const c10::intrusive_ptr<SparseMatrix>& mat, torch::Tensor value,
    const c10::intrusive_ptr<SparseMatrix>& sub_mat) {
  auto csr = CSRToOldDGLCSR(mat->CSRPtr());
  auto val = TorchTensorToDGLArray(value);
  auto row = TorchTensorToDGLArray(sub_mat->COOPtr()->indices.index({0}));
  auto col = TorchTensorToDGLArray(sub_mat->COOPtr()->indices.index({1}));
  runtime::NDArray ret = aten::CSRGetFloatingData(csr, row, col, val, 0.);
  return DGLArrayToTorchTensor(ret);
}

variable_list SpSpMMAutoGrad::forward(
    AutogradContext* ctx, c10::intrusive_ptr<SparseMatrix> lhs_mat,
    torch::Tensor lhs_val, c10::intrusive_ptr<SparseMatrix> rhs_mat,
    torch::Tensor rhs_val) {
  auto ret_mat =
      SpSpMMNoAutoGrad(lhs_mat, lhs_val, rhs_mat, rhs_val, false, false);

  ctx->saved_data["lhs_mat"] = lhs_mat;
  ctx->saved_data["rhs_mat"] = rhs_mat;
  ctx->saved_data["ret_mat"] = ret_mat;
  ctx->saved_data["lhs_require_grad"] = lhs_val.requires_grad();
  ctx->saved_data["rhs_require_grad"] = rhs_val.requires_grad();
  ctx->save_for_backward({lhs_val, rhs_val});

  auto csr = ret_mat->CSRPtr();
  auto val = ret_mat->value();
  TORCH_CHECK(!csr->value_indices.has_value());
  return {csr->indptr, csr->indices, val};
}

tensor_list SpSpMMAutoGrad::backward(
    AutogradContext* ctx, tensor_list grad_outputs) {
  auto saved = ctx->get_saved_variables();
  auto lhs_val = saved[0];
  auto rhs_val = saved[1];
  auto output_grad = grad_outputs[2];
  auto lhs_mat = ctx->saved_data["lhs_mat"].toCustomClass<SparseMatrix>();
  auto rhs_mat = ctx->saved_data["rhs_mat"].toCustomClass<SparseMatrix>();
  auto ret_mat = ctx->saved_data["ret_mat"].toCustomClass<SparseMatrix>();
  torch::Tensor lhs_val_grad, rhs_val_grad;

  if (ctx->saved_data["lhs_require_grad"].toBool()) {
    // A @ B = C -> dA = dC @ (B^T)
    auto lhs_mat_grad =
        SpSpMMNoAutoGrad(ret_mat, output_grad, rhs_mat, rhs_val, false, true);
    lhs_val_grad = _CSRMask(lhs_mat_grad, lhs_mat_grad->value(), lhs_mat);
  }
  if (ctx->saved_data["rhs_require_grad"].toBool()) {
    // A @ B = C -> dB = (A^T) @ dC
    auto rhs_mat_grad =
        SpSpMMNoAutoGrad(lhs_mat, lhs_val, ret_mat, output_grad, true, false);
    rhs_val_grad = _CSRMask(rhs_mat_grad, rhs_mat_grad->value(), rhs_mat);
  }
  return {torch::Tensor(), lhs_val_grad, torch::Tensor(), rhs_val_grad};
}

c10::intrusive_ptr<SparseMatrix> DiagSpSpMM(
    const c10::intrusive_ptr<SparseMatrix>& lhs_mat,
    const c10::intrusive_ptr<SparseMatrix>& rhs_mat) {
  if (lhs_mat->HasDiag() && rhs_mat->HasDiag()) {
    // Diag @ Diag
    const int64_t m = lhs_mat->shape()[0];
    const int64_t n = lhs_mat->shape()[1];
    const int64_t p = rhs_mat->shape()[1];
    const int64_t common_diag_len = std::min({m, n, p});
    const int64_t new_diag_len = std::min(m, p);
    auto slice = torch::indexing::Slice(0, common_diag_len);
    auto new_val =
        lhs_mat->value().index({slice}) * rhs_mat->value().index({slice});
    new_val =
        torch::constant_pad_nd(new_val, {0, new_diag_len - common_diag_len}, 0);
    return SparseMatrix::FromDiag(new_val, {m, p});
  }
  if (lhs_mat->HasDiag() && !rhs_mat->HasDiag()) {
    // Diag @ Sparse
    auto row = rhs_mat->Indices().index({0});
    auto val = lhs_mat->value().index_select(0, row) * rhs_mat->value();
    return SparseMatrix::ValLike(rhs_mat, val);
  }
  if (!lhs_mat->HasDiag() && rhs_mat->HasDiag()) {
    // Sparse @ Diag
    auto col = lhs_mat->Indices().index({1});
    auto val = rhs_mat->value().index_select(0, col) * lhs_mat->value();
    return SparseMatrix::ValLike(lhs_mat, val);
  }
  TORCH_CHECK(
      false,
      "For DiagSpSpMM, at least one of the sparse matries need to have kDiag "
      "format");
  return c10::intrusive_ptr<SparseMatrix>();
}

c10::intrusive_ptr<SparseMatrix> SpSpMM(
    const c10::intrusive_ptr<SparseMatrix>& lhs_mat,
    const c10::intrusive_ptr<SparseMatrix>& rhs_mat) {
  _SpSpMMSanityCheck(lhs_mat, rhs_mat);
  if (lhs_mat->HasDiag() || rhs_mat->HasDiag()) {
    return DiagSpSpMM(lhs_mat, rhs_mat);
  }
  auto results = SpSpMMAutoGrad::apply(
      lhs_mat, lhs_mat->value(), rhs_mat, rhs_mat->value());
  std::vector<int64_t> ret_shape({lhs_mat->shape()[0], rhs_mat->shape()[1]});
  auto indptr = results[0];
  auto indices = results[1];
  auto value = results[2];
  return SparseMatrix::FromCSR(indptr, indices, value, ret_shape);
}

}  // namespace sparse
}  // namespace dgl
