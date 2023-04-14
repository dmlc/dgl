/**
 *  Copyright (c) 2022 by Contributors
 * @file elementwise_op.cc
 * @brief DGL C++ sparse elementwise operator implementation.
 */

#include <sparse/elementwise_op.h>
#include <sparse/matrix_ops.h>
#include <sparse/sparse_matrix.h>
#include <torch/script.h>

#include <memory>

#include "./utils.h"

namespace dgl {
namespace sparse {

using namespace torch::autograd;

c10::intrusive_ptr<SparseMatrix> SpSpAdd(
    const c10::intrusive_ptr<SparseMatrix>& lhs_mat,
    const c10::intrusive_ptr<SparseMatrix>& rhs_mat) {
  ElementwiseOpSanityCheck(lhs_mat, rhs_mat);
  if (lhs_mat->HasDiag() && rhs_mat->HasDiag()) {
    return SparseMatrix::FromDiagPointer(
        lhs_mat->DiagPtr(), lhs_mat->value() + rhs_mat->value(),
        lhs_mat->shape());
  }
  auto torch_lhs = COOToTorchCOO(lhs_mat->COOPtr(), lhs_mat->value());
  auto torch_rhs = COOToTorchCOO(rhs_mat->COOPtr(), rhs_mat->value());
  auto sum = (torch_lhs + torch_rhs).coalesce();
  return SparseMatrix::FromCOO(sum.indices(), sum.values(), lhs_mat->shape());
}

class SpSpMulAutoGrad : public Function<SpSpMulAutoGrad> {
 public:
  static variable_list forward(
      AutogradContext* ctx, c10::intrusive_ptr<SparseMatrix> lhs_mat,
      torch::Tensor lhs_val, c10::intrusive_ptr<SparseMatrix> rhs_mat,
      torch::Tensor rhs_val);

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs);
};

variable_list SpSpMulAutoGrad::forward(
    AutogradContext* ctx, c10::intrusive_ptr<SparseMatrix> lhs_mat,
    torch::Tensor lhs_val, c10::intrusive_ptr<SparseMatrix> rhs_mat,
    torch::Tensor rhs_val) {
  std::shared_ptr<COO> intersection;
  torch::Tensor lhs_indices, rhs_indices;
  std::tie(intersection, lhs_indices, rhs_indices) =
      COOIntersection(lhs_mat->COOPtr(), rhs_mat->COOPtr());
  auto lhs_intersect_val = lhs_val.index_select(0, lhs_indices);
  auto rhs_intersect_val = rhs_val.index_select(0, rhs_indices);
  auto ret_val = lhs_intersect_val * rhs_intersect_val;
  auto ret_mat =
      SparseMatrix::FromCOOPointer(intersection, ret_val, lhs_mat->shape());

  ctx->saved_data["lhs_require_grad"] = lhs_val.requires_grad();
  ctx->saved_data["rhs_require_grad"] = rhs_val.requires_grad();
  if (lhs_val.requires_grad()) {
    ctx->saved_data["lhs_val_shape"] = lhs_val.sizes().vec();
    ctx->saved_data["rhs_intersect_lhs"] =
        SparseMatrix::ValLike(ret_mat, rhs_intersect_val);
    ctx->saved_data["lhs_indices"] = lhs_indices;
  }
  if (rhs_val.requires_grad()) {
    ctx->saved_data["rhs_val_shape"] = rhs_val.sizes().vec();
    ctx->saved_data["lhs_intersect_rhs"] =
        SparseMatrix::ValLike(ret_mat, lhs_intersect_val);
    ctx->saved_data["rhs_indices"] = rhs_indices;
  }
  return {intersection->indices, ret_val};
}

tensor_list SpSpMulAutoGrad::backward(
    AutogradContext* ctx, tensor_list grad_outputs) {
  torch::Tensor lhs_val_grad, rhs_val_grad;
  auto output_grad = grad_outputs[1];
  if (ctx->saved_data["lhs_require_grad"].toBool()) {
    auto rhs_intersect_lhs =
        ctx->saved_data["rhs_intersect_lhs"].toCustomClass<SparseMatrix>();
    const auto& lhs_val_shape = ctx->saved_data["lhs_val_shape"].toIntVector();
    auto lhs_indices = ctx->saved_data["lhs_indices"].toTensor();
    lhs_val_grad = torch::zeros(lhs_val_shape, output_grad.options());
    auto intersect_grad = rhs_intersect_lhs->value() * output_grad;
    lhs_val_grad.index_put_({lhs_indices}, intersect_grad);
  }
  if (ctx->saved_data["rhs_require_grad"].toBool()) {
    auto lhs_intersect_rhs =
        ctx->saved_data["lhs_intersect_rhs"].toCustomClass<SparseMatrix>();
    const auto& rhs_val_shape = ctx->saved_data["rhs_val_shape"].toIntVector();
    auto rhs_indices = ctx->saved_data["rhs_indices"].toTensor();
    rhs_val_grad = torch::zeros(rhs_val_shape, output_grad.options());
    auto intersect_grad = lhs_intersect_rhs->value() * output_grad;
    rhs_val_grad.index_put_({rhs_indices}, intersect_grad);
  }
  return {torch::Tensor(), lhs_val_grad, torch::Tensor(), rhs_val_grad};
}

c10::intrusive_ptr<SparseMatrix> SpSpMul(
    const c10::intrusive_ptr<SparseMatrix>& lhs_mat,
    const c10::intrusive_ptr<SparseMatrix>& rhs_mat) {
  ElementwiseOpSanityCheck(lhs_mat, rhs_mat);
  if (lhs_mat->HasDiag() && rhs_mat->HasDiag()) {
    return SparseMatrix::FromDiagPointer(
        lhs_mat->DiagPtr(), lhs_mat->value() * rhs_mat->value(),
        lhs_mat->shape());
  }
  TORCH_CHECK(
      !lhs_mat->HasDuplicate() && !rhs_mat->HasDuplicate(),
      "Only support SpSpMul on sparse matrices without duplicate values")
  auto results = SpSpMulAutoGrad::apply(
      lhs_mat, lhs_mat->value(), rhs_mat, rhs_mat->value());
  const auto& indices = results[0];
  const auto& val = results[1];
  return SparseMatrix::FromCOO(indices, val, lhs_mat->shape());
}

c10::intrusive_ptr<SparseMatrix> SpSpDiv(
    const c10::intrusive_ptr<SparseMatrix>& lhs_mat,
    const c10::intrusive_ptr<SparseMatrix>& rhs_mat) {
  ElementwiseOpSanityCheck(lhs_mat, rhs_mat);
  if (lhs_mat->HasDiag() && rhs_mat->HasDiag()) {
    return SparseMatrix::FromDiagPointer(
        lhs_mat->DiagPtr(), lhs_mat->value() / rhs_mat->value(),
        lhs_mat->shape());
  }
  std::shared_ptr<COO> sorted_lhs, sorted_rhs;
  torch::Tensor lhs_sorted_perm, rhs_sorted_perm;
  std::tie(sorted_lhs, lhs_sorted_perm) = COOSort(lhs_mat->COOPtr());
  std::tie(sorted_rhs, rhs_sorted_perm) = COOSort(rhs_mat->COOPtr());
  TORCH_CHECK(
      !lhs_mat->HasDuplicate() && !rhs_mat->HasDuplicate(),
      "Only support SpSpDiv on sparse matrices without duplicate values")
  TORCH_CHECK(
      torch::equal(sorted_lhs->indices, sorted_rhs->indices),
      "Cannot divide two COO matrices with different sparsities.");
  // This is to make sure the return matrix is in the same order as the lhs_mat
  auto lhs_sorted_rperm = lhs_sorted_perm.argsort();
  auto rhs_perm_on_lhs = rhs_sorted_perm.index_select(0, lhs_sorted_rperm);
  auto lhs_value = lhs_mat->value();
  auto rhs_value = rhs_mat->value().index_select(0, rhs_perm_on_lhs);
  auto ret_val = lhs_value / rhs_value;
  return SparseMatrix::FromCOOPointer(
      lhs_mat->COOPtr(), ret_val, lhs_mat->shape());
}

}  // namespace sparse
}  // namespace dgl
