/**
 *  Copyright (c) 2022 by Contributors
 * @file spmm.cc
 * @brief DGL C++ sparse SpMM operator implementation.
 */
// clang-format off
#include <sparse/dgl_headers.h>
// clang-format on

#include <sparse/sddmm.h>
#include <sparse/sparse_matrix.h>
#include <sparse/spmm.h>
#include <torch/script.h>

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
  CHECK(sparse_mat_shape[1] == dense_shape[0]);
  CHECK(val_shape.size() == 1 && val_shape[0] == sparse_mat->nnz());
  CHECK_LE(dense_shape.size(), 2);
  CHECK(sparse_val.dtype() == dense_mat.dtype());
  CHECK(
      sparse_val.device() == sparse_mat->device() &&
      sparse_val.device() == dense_mat.device());
}

torch::Tensor SpMMImpl(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat,
    torch::Tensor sparse_val, torch::Tensor dense_mat) {
  // Transpose the sparse matrix because dgl::aten::SpMM calculates A^T @ X.
  auto sparse_mat_tr = sparse_mat->Transpose();
  HeteroGraphPtr dgl_graph;
  // Use CSR if the spars matrix has CSR or does not have COO. Otherwise use
  // COO.
  if (sparse_mat->HasCSC() || !sparse_mat->HasCOO()) {
    auto csc = sparse_mat_tr->CSCPtr();
    dgl_graph = CSCToDGLGraph(csc);
  } else {
    auto coo = sparse_mat_tr->COOPtr();
    dgl_graph = COOToDGLGraph(coo);
  }
  const std::string op = "mul";
  const std::string reduce = "sum";
  int64_t out_row = sparse_mat->shape()[0];
  std::vector<int64_t> shape;

  if (dense_mat.dim() == 1) {
    shape = {out_row};
  } else {
    shape = {out_row, dense_mat.size(1)};
  }
  auto ret = torch::zeros(shape, dense_mat.options());

  aten::SpMM(
      op.c_str(), reduce.c_str(), dgl_graph, TorchTensorToDGLArray(dense_mat),
      TorchTensorToDGLArray(sparse_val), TorchTensorToDGLArray(ret),
      std::vector<runtime::NDArray>());
  return ret;
}

torch::Tensor SpMMAutoGrad::forward(
    AutogradContext* ctx, c10::intrusive_ptr<SparseMatrix> sparse_mat,
    torch::Tensor sparse_val, torch::Tensor dense_mat) {
  _SpMMSanityCheck(sparse_mat, sparse_val, dense_mat);
  auto ret = SpMMImpl(sparse_mat, sparse_val, dense_mat);

  bool sparse_require_grad = sparse_val.requires_grad();
  bool dense_require_grad = dense_mat.requires_grad();
  torch::Tensor cache_sparse_val, cache_dense_mat;
  if (dense_require_grad) {
    cache_sparse_val = sparse_val;
  }
  if (sparse_require_grad) {
    cache_dense_mat = dense_mat;
  }
  ctx->saved_data["sparse_matrix"] = sparse_mat;
  ctx->saved_data["sparse_require_grad"] = sparse_require_grad;
  ctx->saved_data["dense_require_grad"] = dense_require_grad;
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
  bool sparse_require_grad = ctx->saved_data["sparse_require_grad"].toBool();
  bool dense_require_grad = ctx->saved_data["dense_require_grad"].toBool();

  torch::Tensor dense_mat_grad, sparse_val_grad;
  if (sparse_require_grad) {
    sparse_val_grad = SDDMMImpl(sparse_mat, output_grad, dense_mat);
  }
  if (dense_require_grad) {
    auto sparse_mat_tr = sparse_mat->Transpose();
    dense_mat_grad = SpMMImpl(sparse_mat_tr, sparse_val, output_grad);
  }
  return {torch::Tensor(), sparse_val_grad, dense_mat_grad};
}

torch::Tensor SpMM(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat,
    torch::Tensor dense_mat) {
  return SpMMAutoGrad::apply(sparse_mat, sparse_mat->value(), dense_mat);
}

}  // namespace sparse
}  // namespace dgl
