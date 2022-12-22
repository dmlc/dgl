/**
 *  Copyright (c) 2022 by Contributors
 * @file softmax.cc
 * @brief DGL C++ Softmax operator implementation
 */
// clang-format off
#include <sparse/dgl_headers.h>
// clang-format on

#include <sparse/sparse_matrix.h>
#include <torch/script.h>

#include "./matmul.h"
#include "./utils.h"

namespace dgl {
namespace sparse {

using namespace torch::autograd;

class SoftmaxAutoGrad : public Function<SoftmaxAutoGrad> {
 public:
  static torch::Tensor forward(
      AutogradContext* ctx, c10::intrusive_ptr<SparseMatrix> sparse_mat,
      torch::Tensor sparse_val);

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs);
};

void _SoftmaxSanityCheck(
    c10::intrusive_ptr<SparseMatrix> sparse_mat, torch::Tensor sparse_val) {
  auto val_shape = sparse_val.sizes();
  CHECK_EQ(val_shape[0], sparse_mat->nnz())
      << "Softmax: the value shape does not match nnz of the sparse matrix.";
  CHECK(sparse_val.device() == sparse_mat->device())
      << "Softmax: sparse matrix and non-zero values should be on the same "
         "device.";
}

torch::Tensor SoftmaxAutoGrad::forward(
    AutogradContext* ctx, c10::intrusive_ptr<SparseMatrix> sparse_mat,
    torch::Tensor sparse_val) {
  torch::Tensor sparse_score;
  if (sparse_val.device() == torch::kCPU) {
    auto csr = CSRToOldDGLCSR(sparse_mat->CSRPtr());
    auto dgl_ufeat = aten::NullArray();
    auto dgl_sparse_val = TorchTensorToDGLArray(sparse_val);
    sparse_score = torch::zeros(sparse_val.sizes(), sparse_val.options());
    auto dgl_sparse_score = TorchTensorToDGLArray(sparse_score);
    aten::CSREdgeSoftmaxForward(
        "copy_rhs", csr, dgl_ufeat, dgl_sparse_val, dgl_sparse_score);
  } else {
    auto sparse_val_max = SpMMNoAutoGrad(sparse_mat, sparse_val, "max");
    auto sparse_val_exp = SDDMMNoAutoGrad(
        sparse_mat, sparse_val, sparse_val_max, "sub").exp();
    auto sparse_val_sum = SpMMNoAutoGrad(sparse_mat, sparse_val_exp, "sum");
    sparse_score = SDDMMNoAutoGrad(
        sparse_mat, sparse_val_exp, sparse_val_sum, "div");
  }

  const bool sparse_requires_grad = sparse_val.requires_grad();
  torch::Tensor cache_sparse_score;
  if (sparse_requires_grad) {
    cache_sparse_score = sparse_score;
  }
  ctx->saved_data["sparse_matrix"] = sparse_mat;
  ctx->saved_data["sparse_requires_grad"] = sparse_requires_grad;
  ctx->save_for_backward({cache_sparse_score});
  return sparse_score;
}

tensor_list SoftmaxAutoGrad::backward(
    AutogradContext* ctx, tensor_list grad_outputs) {
  auto saved = ctx->get_saved_variables();
  auto sparse_score = saved[0];
  auto output_grad = grad_outputs[0];

  auto sparse_mat =
      ctx->saved_data["sparse_matrix"].toCustomClass<SparseMatrix>();
  const bool sparse_requires_grad =
      ctx->saved_data["sparse_requires_grad"].toBool();

  torch::Tensor sparse_val_grad;
  if (sparse_requires_grad) {
    if (sparse_score.device() == torch::kCPU) {

    } else {
      auto sds = sparse_score * output_grad;
      auto accum = SpMMNoAutoGrad(sparse_mat, sds, "sum");
      sparse_val_grad = sds - SDDMMNoAutoGrad(
          sparse_mat, sparse_score, accum, "mul");
    }
  }

  return {torch::Tensor(), sparse_val_grad};
}

c10::intrusive_ptr<SparseMatrix> Softmax(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat) {
  auto sparse_val = sparse_mat->value();
  _SoftmaxSanityCheck(sparse_mat, sparse_val);
  bool expand_dim = false;
  if (sparse_val.dim() == 1) {
    sparse_val = sparse_val.view({-1, 1});
    expand_dim = true;
  }
  auto new_sparse_val = SoftmaxAutoGrad::apply(sparse_mat, sparse_val);
  if (expand_dim) {
    new_sparse_val = new_sparse_val.view(-1);
  }
  return CreateValLike(sparse_mat, new_sparse_val);
}

}  // namespace sparse
}  // namespace dgl
