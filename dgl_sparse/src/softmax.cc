/**
 *  Copyright (c) 2022 by Contributors
 * @file softmax.cc
 * @brief DGL C++ Softmax operator implementation
 */

#include <sparse/sparse_matrix.h>
#include <torch/script.h>

#include "./matmul.h"

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
  const auto& sparse_mat_shape = sparse_mat->shape();
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
  auto sparse_val_max = SpMMNoAutoGrad(sparse_mat, sparse_val, "max");
  return sparse_val_max;
}

tensor_list SoftmaxAutoGrad::backward(
    AutogradContext* ctx, tensor_list grad_outputs) {
  return {};
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
