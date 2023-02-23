/**
 *  Copyright (c) 2022 by Contributors
 * @file reduction.cc
 * @brief DGL C++ sparse matrix reduction operator implementation.
 */
// clang-format off
#include <sparse/dgl_headers.h>
// clang-format on

#include <sparse/elementwise_op.h>
#include <sparse/reduction.h>
#include <sparse/sparse_matrix.h>
#include <torch/script.h>

#include <string>
#include <vector>

namespace dgl {
namespace sparse {

namespace {

torch::Tensor ReduceAlong(
    const c10::intrusive_ptr<SparseMatrix>& A, const std::string& reduce,
    int64_t dim) {
  auto value = A->value();
  auto coo = A->COOPtr();

  std::string reduce_op;
  if (reduce == "sum") {
    reduce_op = "sum";
  } else if (reduce == "smin") {
    reduce_op = "amin";
  } else if (reduce == "smax") {
    reduce_op = "amax";
  } else if (reduce == "smean") {
    reduce_op = "mean";
  } else if (reduce == "sprod") {
    reduce_op = "prod";
  } else {
    TORCH_CHECK(false, "unknown reduce function ", reduce);
    return torch::Tensor();
  }

  // Create the output tensor with shape
  //
  //   [A.num_rows if dim == 1 else A.num_cols] + A.val.shape[1:]
  std::vector<int64_t> output_shape = value.sizes().vec();
  std::vector<int64_t> view_dims(output_shape.size(), 1);
  view_dims[0] = -1;
  torch::Tensor idx;
  if (dim == 0) {
    output_shape[0] = coo->num_cols;
    idx = coo->indices.index({1}).view(view_dims).expand_as(value);
  } else if (dim == 1) {
    output_shape[0] = coo->num_rows;
    idx = coo->indices.index({0}).view(view_dims).expand_as(value);
  }
  torch::Tensor out = torch::zeros(output_shape, value.options());

  if (dim == 0) {
    out.scatter_reduce_(0, idx, value, reduce_op, false);
  } else if (dim == 1) {
    out.scatter_reduce_(0, idx, value, reduce_op, false);
  }

  return out;
}

torch::Tensor ReduceAll(
    const c10::intrusive_ptr<SparseMatrix>& A, const std::string& reduce) {
  if (reduce == "sum") {
    return A->value().sum(0);
  } else if (reduce == "smin") {
    return A->value().amin(0);
  } else if (reduce == "smax") {
    return A->value().amax(0);
  } else if (reduce == "smean") {
    return A->value().mean(0);
  } else if (reduce == "sprod") {
    return A->value().prod(0);
  }

  TORCH_CHECK(false, "unknown reduce function ", reduce);
  return torch::Tensor();
}

}  // namespace

torch::Tensor Reduce(
    const c10::intrusive_ptr<SparseMatrix>& A, const std::string& reduce,
    const torch::optional<int64_t>& dim) {
  return dim.has_value() ? ReduceAlong(A, reduce, dim.value())
                         : ReduceAll(A, reduce);
}

}  // namespace sparse
}  // namespace dgl
