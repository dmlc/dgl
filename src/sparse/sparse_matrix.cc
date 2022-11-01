/*!
 *  Copyright (c) 2022 by Contributors
 * \file sparse/sparse_matrix.cc
 * \brief DGL C++ sparse matrix implementations
 */
#include <dgl/sparse/elementwise_op.h>
#include <dgl/sparse/sparse_matrix.h>

namespace dgl {
namespace sparse {

std::vector<torch::Tensor> SparseMatrix::COOTensors() {
  auto coo = COOPtr();
  auto val = Value();
  if (coo->value_indices.has_value()) {
    val = val[coo->value_indices.value()];
  }
  return {coo->row, coo->col, val};
}

c10::intrusive_ptr<SparseMatrix> CreateFromCOOPtr(
    const std::shared_ptr<COO>& coo, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  return c10::make_intrusive<SparseMatrix>(coo, value, shape);
}

c10::intrusive_ptr<SparseMatrix> CreateFromCOO(
    torch::Tensor row, torch::Tensor col, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  auto coo =
      std::make_shared<COO>(COO{row, col, torch::optional<torch::Tensor>()});
  return CreateFromCOOPtr(coo, value, shape);
}

}  // namespace sparse
}  // namespace dgl
