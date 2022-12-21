/**
 *  Copyright (c) 2022 by Contributors
 * @file sddmm.cc
 * @brief DGL C++ sparse SDDMM operator implementation.
 */
#include <sparse/sparse_matrix.h>
#include <sparse/spmm.h>
#include <torch/script.h>

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
  const int64_t mat1_dim = mat1.dim();
  const int64_t mat2_dim = mat2.dim();
  CHECK_EQ(mat1_dim, mat2_dim)
      << "SDDMM: the two dense matrices should have the same dimensions.";
  CHECK_LE(mat1_dim, 2)
      << "SDDMM: the first dense matrix should have at most two dimensions.";
  CHECK_EQ(sparse_mat->shape()[0], mat1.size(0))
      << "SDDMM: the first dense matrix should have the same first dimension "
         "as the sparse matrix";
  CHECK_EQ(sparse_mat->shape()[1], mat2.size(mat2_dim - 1))
      << "SDDMM: the second dense matrix should have the same last dimension "
         "as the sparse matrix";
  if (mat1_dim == 2) {
    CHECK_EQ(mat1.size(1), mat2.size(0))
        << "SDDMM: the second dimension of the first dense matrix should be "
           "equal to the first dimension of the second dense matrix.";
  }
  CHECK_EQ(mat1.dtype(), mat2.dtype())
      << "SDDMM: the two dense matrices should have the same dtype.";
  CHECK_EQ(mat1.device(), mat2.device())
      << "SDDMM: the two dense matrices should on the same device.";
}

torch::Tensor SDDMMAutoGrad::forward(
    AutogradContext* ctx, const c10::intrusive_ptr<SparseMatrix>& sparse_mat,
    torch::Tensor mat1, torch::Tensor mat2) {
  auto mat2_tr = mat2.transpose(0, 1).contiguous();
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
    mat1_grad = SpMMNoAutoGrad(
        sparse_mat, grad, mat2.transpose(0, 1).contiguous(), false);
  }
  if (ctx->saved_data["mat2_requires_grad"].toBool()) {
    // SDDMM(M, A, B) = C. dB = SpMM(dC^T, A)^T
    auto mat2_tr_grad = SpMMNoAutoGrad(sparse_mat, grad, mat1, true);
    mat2_grad = mat2_tr_grad.transpose(0, 1).contiguous();
  }
  return {torch::Tensor(), mat1_grad, mat2_grad};
}

c10::intrusive_ptr<SparseMatrix> SDDMM(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat, torch::Tensor mat1,
    torch::Tensor mat2) {
  if (mat1.dim() == 1) {
    mat1 = mat1.view({mat1.size(0), 1});
    mat2 = mat2.view({1, mat2.size(0)});
  }
  _SDDMMSanityCheck(sparse_mat, mat1, mat2);
  auto val = SDDMMAutoGrad::apply(sparse_mat, mat1, mat2);
  val = val * sparse_mat->value();
  return CreateValLike(sparse_mat, val);
}

}  // namespace sparse
}  // namespace dgl
