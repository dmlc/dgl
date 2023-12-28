/**
 *  Copyright (c) 2022 by Contributors
 * @file sddmm.cc
 * @brief DGL C++ sparse SDDMM operator implementation.
 */
#include <sparse/sparse_matrix.h>
#include <sparse/spmm.h>
#include <torch/script.h>

#include <sstream>

#include "./matmul.h"
#include "./utils.h"

namespace dgl {
namespace sparse {

using namespace torch::autograd;

class SDDMMAutoGrad : public Function<SDDMMAutoGrad> {
 public:
  static torch::Tensor forward(
      AutogradContext* ctx, const c10::intrusive_ptr<SparseMatrix>& sparse_mat,
      torch::Tensor mat1, torch::Tensor mat2_tr);

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs);
};

void _SDDMMSanityCheck(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat, torch::Tensor mat1,
    torch::Tensor mat2) {
  bool shape_check = true;
  shape_check &= mat1.dim() == mat2.dim();
  shape_check &= mat1.dim() <= 3;
  shape_check &= sparse_mat->shape()[0] == mat1.size(0);
  if (mat1.dim() == 3) {
    shape_check &= sparse_mat->shape()[1] == mat2.size(1);
    shape_check &= mat1.size(2) == mat2.size(2);
    if (sparse_mat->value().dim() > 1) {
      shape_check &= sparse_mat->value().size(1) == mat1.size(2);
    }
  } else {
    shape_check &= sparse_mat->shape()[1] == mat2.size(mat2.dim() - 1);
  }
  if (mat1.dim() >= 2) {
    shape_check &= mat1.size(1) == mat2.size(0);
  }
  if (!shape_check) {
    std::stringstream error;
    error << "SDDMM: Invalid input shapes. sparse_mat: "
          << c10::IntArrayRef(sparse_mat->shape())
          << ", sparse_val: " << sparse_mat->value().sizes()
          << ", mat1: " << mat1.sizes() << ", mat2: " << mat2.sizes()
          << ". Valid input shapes (sparse_mat, mat1, mat2) are: (1) (n, m), "
             "(n, k), and (k, m); (2) (n, m), (n,), and (m,); (3) (n, m, b), "
             "(n, k, b) and (k, m, b); (4) "
             "(n, m), (n, k, b), and (k, m, b).";
    TORCH_CHECK(false, error.str());
  }
  TORCH_CHECK(
      mat1.dtype() == mat2.dtype(),
      "SDDMM: the two dense matrices should have the same dtype.");
  TORCH_CHECK(
      mat1.device() == mat2.device() && sparse_mat->device() == mat2.device(),
      "SDDMM: the two dense matrices and sparse matrix should on the same "
      "device.");
}

torch::Tensor SDDMMAutoGrad::forward(
    AutogradContext* ctx, const c10::intrusive_ptr<SparseMatrix>& sparse_mat,
    torch::Tensor mat1, torch::Tensor mat2) {
  auto mat2_tr = mat2.transpose(0, 1);
  auto ret = SDDMMNoAutoGrad(sparse_mat, mat1, mat2_tr);
  torch::Tensor cache_mat1, cache_mat2;
  if (mat1.requires_grad()) {
    cache_mat2 = mat2;
  }
  if (mat2.requires_grad()) {
    cache_mat1 = mat1;
  }
  ctx->save_for_backward({cache_mat1, cache_mat2});
  ctx->saved_data["mat1_requires_grad"] = mat1.requires_grad();
  ctx->saved_data["mat2_requires_grad"] = mat2.requires_grad();
  ctx->saved_data["sparse_mat"] = sparse_mat;
  return ret;
}

tensor_list SDDMMAutoGrad::backward(
    AutogradContext* ctx, tensor_list grad_outputs) {
  auto saved = ctx->get_saved_variables();
  auto mat1 = saved[0];
  auto mat2 = saved[1];
  auto sparse_mat = ctx->saved_data["sparse_mat"].toCustomClass<SparseMatrix>();
  auto grad = grad_outputs[0];
  torch::Tensor mat1_grad, mat2_grad;
  if (ctx->saved_data["mat1_requires_grad"].toBool()) {
    // SDDMM(M, A, B) = C. dA = SpMM(dC, B^T)
    mat1_grad = SpMMNoAutoGrad(sparse_mat, grad, mat2.transpose(0, 1), false);
  }
  if (ctx->saved_data["mat2_requires_grad"].toBool()) {
    // SDDMM(M, A, B) = C. dB = SpMM(dC^T, A)^T
    auto mat2_tr_grad = SpMMNoAutoGrad(sparse_mat, grad, mat1, true);
    mat2_grad = mat2_tr_grad.transpose(0, 1);
  }
  return {torch::Tensor(), mat1_grad, mat2_grad};
}

c10::intrusive_ptr<SparseMatrix> SDDMM(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat, torch::Tensor mat1,
    torch::Tensor mat2) {
  if (mat1.dim() == 1) {
    mat1 = mat1.view({mat1.size(0), 1});
  }
  if (mat2.dim() == 1) {
    mat2 = mat2.view({1, mat2.size(0)});
  }
  _SDDMMSanityCheck(sparse_mat, mat1, mat2);
  auto val = SDDMMAutoGrad::apply(sparse_mat, mat1, mat2);
  auto sparse_val = sparse_mat->value();
  // Broadcast the sparse value in batched SDDMM.
  if (sparse_val.dim() < val.dim()) {
    sparse_val = sparse_val.unsqueeze(-1);
  }
  val = val * sparse_val;
  return SparseMatrix::ValLike(sparse_mat, val);
}

}  // namespace sparse
}  // namespace dgl
