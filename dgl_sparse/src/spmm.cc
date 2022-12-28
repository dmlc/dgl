/**
 *  Copyright (c) 2022 by Contributors
 * @file spmm.cc
 * @brief DGL C++ sparse SpMM operator implementation.
 */

#include <sparse/sddmm.h>
#include <sparse/sparse_matrix.h>
#include <sparse/spmm.h>
#include <torch/script.h>

#include <sstream>

#include "./matmul.h"
#include "./utils.h"

namespace dgl {
namespace sparse {

using namespace torch::autograd;

class SpMMAutoGrad : public Function<SpMMAutoGrad> {
 public:
  static torch::Tensor forward(
      AutogradContext* ctx, c10::intrusive_ptr<SparseMatrix> sparse_mat,
      torch::Tensor sparse_val, torch::Tensor dense_mat);

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs);
};

void _SpMMSanityCheck(
    c10::intrusive_ptr<SparseMatrix> sparse_mat, torch::Tensor sparse_val,
    torch::Tensor dense_mat) {
  const auto& sparse_mat_shape = sparse_mat->shape();
  auto val_shape = sparse_val.sizes();
  auto dense_shape = dense_mat.sizes();
  bool shape_check = true;
  shape_check &= sparse_mat_shape[1] == dense_shape[0];
  shape_check &= val_shape.size() <= 2;
  shape_check &= val_shape[0] == sparse_mat->nnz();
  shape_check &= dense_shape.size() <= 3;
  if (dense_shape.size() == 3 || val_shape.size() == 2) {
    shape_check &= dense_shape.size() == val_shape.size() + 1;
    shape_check &= dense_shape[2] == val_shape[1];
  }
  if (!shape_check) {
    std::stringstream error;
    error << "SpMM: Invalid input shapes. sparse_mat: "
          << c10::IntArrayRef(sparse_mat->shape())
          << ", sparse_val: " << sparse_mat->value().sizes()
          << ", dense_mat: " << dense_mat.sizes()
          << ". Valid input shapes (sparse_mat, dense_mat) are: (1) (n, m) and "
             "(m, k); (2) (n, m) and (m,); (3) (n, m, b) and (m, k, b).";
    TORCH_CHECK(false, error.str());
  }
  TORCH_CHECK(
      sparse_val.dtype() == dense_mat.dtype(),
      "SpMM: the non-zero values does not have the same dtype as the dense "
      "matrix.");
  TORCH_CHECK(
      sparse_val.device() == sparse_mat->device() &&
          sparse_val.device() == dense_mat.device(),
      "SpMM: sparse matrix, non-zero values and the dense matrix should be "
      "on the same device.");
}

torch::Tensor SpMMAutoGrad::forward(
    AutogradContext* ctx, c10::intrusive_ptr<SparseMatrix> sparse_mat,
    torch::Tensor sparse_val, torch::Tensor dense_mat) {
  auto ret = SpMMNoAutoGrad(sparse_mat, sparse_val, dense_mat, false);

  const bool sparse_requires_grad = sparse_val.requires_grad();
  const bool dense_requires_grad = dense_mat.requires_grad();
  torch::Tensor cache_sparse_val, cache_dense_mat;
  if (dense_requires_grad) {
    cache_sparse_val = sparse_val;
  }
  if (sparse_requires_grad) {
    cache_dense_mat = dense_mat;
  }
  ctx->saved_data["sparse_matrix"] = sparse_mat;
  ctx->saved_data["sparse_requires_grad"] = sparse_requires_grad;
  ctx->saved_data["dense_requires_grad"] = dense_requires_grad;
  ctx->save_for_backward({cache_sparse_val, cache_dense_mat});
  return ret;
}

tensor_list SpMMAutoGrad::backward(
    AutogradContext* ctx, tensor_list grad_outputs) {
  auto saved = ctx->get_saved_variables();
  auto sparse_val = saved[0];
  auto dense_mat = saved[1];
  auto output_grad = grad_outputs[0];

  auto sparse_mat =
      ctx->saved_data["sparse_matrix"].toCustomClass<SparseMatrix>();
  const bool sparse_requires_grad =
      ctx->saved_data["sparse_requires_grad"].toBool();
  const bool dense_requires_grad =
      ctx->saved_data["dense_requires_grad"].toBool();

  torch::Tensor dense_mat_grad, sparse_val_grad;
  if (sparse_requires_grad) {
    // A @ B = C -> dA = dC @ (B^T)
    sparse_val_grad = SDDMMNoAutoGrad(sparse_mat, output_grad, dense_mat);
  }
  if (dense_requires_grad) {
    // A @ B = C -> dB = (A^T) @ dC
    dense_mat_grad = SpMMNoAutoGrad(sparse_mat, sparse_val, output_grad, true);
  }
  return {torch::Tensor(), sparse_val_grad, dense_mat_grad};
}

torch::Tensor SpMM(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat,
    torch::Tensor dense_mat) {
  _SpMMSanityCheck(sparse_mat, sparse_mat->value(), dense_mat);
  bool expand_dim = false;
  if (dense_mat.dim() == 1) {
    dense_mat = dense_mat.view({-1, 1});
    expand_dim = true;
  }
  auto ret = SpMMAutoGrad::apply(sparse_mat, sparse_mat->value(), dense_mat);
  if (expand_dim) {
    ret = ret.view(-1);
  }
  return ret;
}

}  // namespace sparse
}  // namespace dgl
