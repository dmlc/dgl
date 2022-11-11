/**
 *  Copyright (c) 2022 by Contributors
 * @file sparse_matrix.cc
 * @brief DGL C++ sparse matrix implementations.
 */
// clang-format off
#include <sparse/dgl_headers.h>
// clang-format on

#include <c10/util/Logging.h>
#include <sparse/elementwise_op.h>
#include <sparse/sparse_matrix.h>
#include <torch/script.h>

namespace dgl {
namespace sparse {

SparseMatrix::SparseMatrix(
    const std::shared_ptr<COO>& coo, const std::shared_ptr<CSR>& csr,
    const std::shared_ptr<CSR>& csc, torch::Tensor value,
    const std::vector<int64_t>& shape)
    : coo_(coo), csr_(csr), csc_(csc), value_(value), shape_(shape) {
  CHECK(coo != nullptr || csr != nullptr || csc != nullptr)
      << "At least one of CSR/COO/CSC is provided to construct a "
         "SparseMatrix";
  CHECK_EQ(shape.size(), 2)
      << "The shape of a sparse matrix should be 2-dimensional";
  // NOTE: Currently all the tensors of a SparseMatrix should on the same
  // device. Do we allow the graph structure and values are on different
  // devices?
  if (coo != nullptr) {
    CHECK_EQ(coo->row.dim(), 1);
    CHECK_EQ(coo->col.dim(), 1);
    CHECK_EQ(coo->row.size(0), coo->col.size(0));
    CHECK_EQ(coo->row.size(0), value.size(0));
    CHECK_EQ(coo->row.device(), value.device());
    CHECK_EQ(coo->col.device(), value.device());
  }
  if (csr != nullptr) {
    CHECK_EQ(csr->indptr.dim(), 1);
    CHECK_EQ(csr->indices.dim(), 1);
    CHECK_EQ(csr->indptr.size(0), shape[0] + 1);
    CHECK_EQ(csr->indices.size(0), value.size(0));
    CHECK_EQ(csr->indptr.device(), value.device());
    CHECK_EQ(csr->indices.device(), value.device());
  }
  if (csc != nullptr) {
    CHECK_EQ(csc->indptr.dim(), 1);
    CHECK_EQ(csc->indices.dim(), 1);
    CHECK_EQ(csc->indptr.size(0), shape[1] + 1);
    CHECK_EQ(csc->indices.size(0), value.size(0));
    CHECK_EQ(csc->indptr.device(), value.device());
    CHECK_EQ(csc->indices.device(), value.device());
  }
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::FromCOO(
    const std::shared_ptr<COO>& coo, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  return c10::make_intrusive<SparseMatrix>(coo, nullptr, nullptr, value, shape);
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::FromCSR(
    const std::shared_ptr<CSR>& csr, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  return c10::make_intrusive<SparseMatrix>(nullptr, csr, nullptr, value, shape);
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::FromCSC(
    const std::shared_ptr<CSR>& csc, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  return c10::make_intrusive<SparseMatrix>(nullptr, nullptr, csc, value, shape);
}

std::shared_ptr<COO> SparseMatrix::COOPtr() {
  if (coo_ == nullptr) {
    _CreateCOO();
  }
  return coo_;
}

std::shared_ptr<CSR> SparseMatrix::CSRPtr() {
  if (csr_ == nullptr) {
    _CreateCSR();
  }
  return csr_;
}

std::shared_ptr<CSR> SparseMatrix::CSCPtr() {
  if (csc_ == nullptr) {
    _CreateCSC();
  }
  return csc_;
}

std::vector<torch::Tensor> SparseMatrix::COOTensors() {
  auto coo = COOPtr();
  auto val = value();
  return {coo->row, coo->col, val};
}

std::vector<torch::Tensor> SparseMatrix::CSRTensors() {
  auto csr = CSRPtr();
  auto val = value();
  if (csr->value_indices.has_value()) {
    val = val[csr->value_indices.value()];
  }
  return {csr->indptr, csr->indices, val};
}

std::vector<torch::Tensor> SparseMatrix::CSCTensors() {
  auto csc = CSCPtr();
  auto val = value();
  if (csc->value_indices.has_value()) {
    val = val[csc->value_indices.value()];
  }
  return {csc->indptr, csc->indices, val};
}

void SparseMatrix::SetValue(torch::Tensor value) { value_ = value; }

void SparseMatrix::_CreateCOO() {
  if (HasCOO()) {
    return;
  }
  if (HasCSR()) {
    coo_ = CSRToCOO(shape_[0], shape_[1], csr_);
  } else if (HasCSC()) {
    // TODO(zhenkun)
  } else {
    LOG(FATAL) << "SparseMatrix does not have any sparse format";
  }
}

void SparseMatrix::_CreateCSR() {}
void SparseMatrix::_CreateCSC() {}

c10::intrusive_ptr<SparseMatrix> CreateFromCOO(
    torch::Tensor row, torch::Tensor col, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  auto coo = std::make_shared<COO>(COO{row, col});
  return SparseMatrix::FromCOO(coo, value, shape);
}

c10::intrusive_ptr<SparseMatrix> CreateFromCSR(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  auto csr = std::make_shared<CSR>(
      CSR{indptr, indices, torch::optional<torch::Tensor>()});
  return SparseMatrix::FromCSR(csr, value, shape);
}

c10::intrusive_ptr<SparseMatrix> CreateFromCSC(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  auto csc = std::make_shared<CSR>(
      CSR{indptr, indices, torch::optional<torch::Tensor>()});
  return SparseMatrix::FromCSC(csc, value, shape);
}

}  // namespace sparse
}  // namespace dgl
