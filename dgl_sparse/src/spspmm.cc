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
    const c10::intrusive_ptr<SparseMatrix>& lhs_mat, torch::Tensor lhs_val,
    const c10::intrusive_ptr<SparseMatrix>& rhs_mat, torch::Tensor rhs_val) {
  const auto& lhs_shape = lhs_mat->shape();
  const auto& rhs_shape = rhs_mat->shape();
  CHECK_EQ(lhs_shape[1], rhs_shape[0])
      << "SpSpMM: the second dim of lhs_mat should be equal to the first dim "
         "of the second matrix";
  CHECK_EQ(lhs_val.dim(), 1)
      << "SpSpMM: the value shape of lhs_mat should be 1-D";
  CHECK_EQ(rhs_val.dim(), 1)
      << "SpSpMM: the value shape of rhs_mat should be 1-D";
  CHECK_EQ(lhs_val.size(0), lhs_mat->nnz())
      << "SpSpMM: the number of lhs_val should be equal to the nnz of lhs_mat";
  CHECK_EQ(rhs_val.size(0), rhs_mat->nnz())
      << "SpSpMM: the number of rhs_val should be equal to the nnz of rhs_mat";
  CHECK_EQ(lhs_mat->device(), rhs_mat->device())
      << "SpSpMM: lhs_mat and rhs_mat should on the same device";
  CHECK_EQ(lhs_val.dtype(), rhs_val.dtype())
      << "SpSpMM: lhs_val and rhs_val should have the same dtype";
  CHECK_EQ(lhs_val.device(), rhs_val.device())
      << "SpSpMM: lhs_val and rhs_val should on the same device";
}

// Mask select value of `mat` by `sub_mat`.
torch::Tensor _CSRMask(
    const c10::intrusive_ptr<SparseMatrix>& mat, torch::Tensor value,
    const c10::intrusive_ptr<SparseMatrix>& sub_mat) {
  auto csr = CSRToOldDGLCSR(mat->CSRPtr());
  auto val = TorchTensorToDGLArray(value);
  auto row = TorchTensorToDGLArray(sub_mat->COOPtr()->row);
  auto col = TorchTensorToDGLArray(sub_mat->COOPtr()->col);
  runtime::NDArray ret;
  ATEN_FLOAT_TYPE_SWITCH(val->dtype, DType, "Value Type", {
    ret = aten::CSRGetData<DType>(csr, row, col, val, 0.);
  });
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
  CHECK(!csr->value_indices.has_value());
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

c10::intrusive_ptr<SparseMatrix> SpSpMM(
    const c10::intrusive_ptr<SparseMatrix>& lhs_mat,
    const c10::intrusive_ptr<SparseMatrix>& rhs_mat) {
  _SpSpMMSanityCheck(lhs_mat, lhs_mat->value(), rhs_mat, rhs_mat->value());
  auto results = SpSpMMAutoGrad::apply(
      lhs_mat, lhs_mat->value(), rhs_mat, rhs_mat->value());
  std::vector<int64_t> ret_shape({lhs_mat->shape()[0], rhs_mat->shape()[1]});
  auto indptr = results[0];
  auto indices = results[1];
  auto value = results[2];
  return CreateFromCSR(indptr, indices, value, ret_shape);
}

}  // namespace sparse
}  // namespace dgl
