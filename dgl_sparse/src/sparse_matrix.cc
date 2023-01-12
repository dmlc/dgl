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
  TORCH_CHECK(
      coo != nullptr || csr != nullptr || csc != nullptr, "At least ",
      "one of CSR/COO/CSC is required to construct a SparseMatrix.")
  TORCH_CHECK(
      shape.size() == 2, "The shape of a sparse matrix should be ",
      "2-dimensional.");
  // NOTE: Currently all the tensors of a SparseMatrix should on the same
  // device. Do we allow the graph structure and values are on different
  // devices?
  if (coo != nullptr) {
    TORCH_CHECK(coo->row.dim() == 1);
    TORCH_CHECK(coo->col.dim() == 1);
    TORCH_CHECK(coo->row.size(0) == coo->col.size(0));
    TORCH_CHECK(coo->row.size(0) == value.size(0));
    TORCH_CHECK(coo->row.device() == value.device());
    TORCH_CHECK(coo->col.device() == value.device());
  }
  if (csr != nullptr) {
    TORCH_CHECK(csr->indptr.dim() == 1);
    TORCH_CHECK(csr->indices.dim() == 1);
    TORCH_CHECK(csr->indptr.size(0) == shape[0] + 1);
    TORCH_CHECK(csr->indices.size(0) == value.size(0));
    TORCH_CHECK(csr->indptr.device() == value.device());
    TORCH_CHECK(csr->indices.device() == value.device());
  }
  if (csc != nullptr) {
    TORCH_CHECK(csc->indptr.dim() == 1);
    TORCH_CHECK(csc->indices.dim() == 1);
    TORCH_CHECK(csc->indptr.size(0) == shape[1] + 1);
    TORCH_CHECK(csc->indices.size(0) == value.size(0));
    TORCH_CHECK(csc->indptr.device() == value.device());
    TORCH_CHECK(csc->indices.device() == value.device());
  }
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::FromCOOPointer(
    const std::shared_ptr<COO>& coo, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  return c10::make_intrusive<SparseMatrix>(coo, nullptr, nullptr, value, shape);
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::FromCSRPointer(
    const std::shared_ptr<CSR>& csr, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  return c10::make_intrusive<SparseMatrix>(nullptr, csr, nullptr, value, shape);
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::FromCSCPointer(
    const std::shared_ptr<CSR>& csc, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  return c10::make_intrusive<SparseMatrix>(nullptr, nullptr, csc, value, shape);
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::FromCOO(
    torch::Tensor row, torch::Tensor col, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  auto coo =
      std::make_shared<COO>(COO{shape[0], shape[1], row, col, false, false});
  return SparseMatrix::FromCOOPointer(coo, value, shape);
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::FromCSR(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  auto csr = std::make_shared<CSR>(
      CSR{shape[0], shape[1], indptr, indices, torch::optional<torch::Tensor>(),
          false});
  return SparseMatrix::FromCSRPointer(csr, value, shape);
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::FromCSC(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  auto csc = std::make_shared<CSR>(
      CSR{shape[1], shape[0], indptr, indices, torch::optional<torch::Tensor>(),
          false});
  return SparseMatrix::FromCSCPointer(csc, value, shape);
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::ValLike(
    const c10::intrusive_ptr<SparseMatrix>& mat, torch::Tensor value) {
  TORCH_CHECK(
      mat->value().size(0) == value.size(0), "The first dimension of ",
      "the old values and the new values must be the same.");
  TORCH_CHECK(
      mat->value().device() == value.device(), "The device of the ",
      "old values and the new values must be the same.");
  auto shape = mat->shape();
  if (mat->HasCOO()) {
    return SparseMatrix::FromCOOPointer(mat->COOPtr(), value, shape);
  } else if (mat->HasCSR()) {
    return SparseMatrix::FromCSRPointer(mat->CSRPtr(), value, shape);
  } else {
    return SparseMatrix::FromCSCPointer(mat->CSCPtr(), value, shape);
  }
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

std::tuple<torch::Tensor, torch::Tensor> SparseMatrix::COOTensors() {
  auto coo = COOPtr();
  auto val = value();
  return std::make_tuple(coo->row, coo->col);
}

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
SparseMatrix::CSRTensors() {
  auto csr = CSRPtr();
  auto val = value();
  return std::make_tuple(csr->indptr, csr->indices, csr->value_indices);
}

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
SparseMatrix::CSCTensors() {
  auto csc = CSCPtr();
  return std::make_tuple(csc->indptr, csc->indices, csc->value_indices);
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::Transpose() const {
  auto shape = shape_;
  std::swap(shape[0], shape[1]);
  auto value = value_;
  if (HasCOO()) {
    auto coo = COOTranspose(coo_);
    return SparseMatrix::FromCOOPointer(coo, value, shape);
  } else if (HasCSR()) {
    return SparseMatrix::FromCSCPointer(csr_, value, shape);
  } else {
    return SparseMatrix::FromCSRPointer(csc_, value, shape);
  }
}

void SparseMatrix::_CreateCOO() {
  if (HasCOO()) return;
  if (HasCSR()) {
    coo_ = CSRToCOO(csr_);
  } else if (HasCSC()) {
    coo_ = CSCToCOO(csc_);
  } else {
    LOG(FATAL) << "SparseMatrix does not have any sparse format";
  }
}

void SparseMatrix::_CreateCSR() {
  if (HasCSR()) return;
  if (HasCOO()) {
    csr_ = COOToCSR(coo_);
  } else if (HasCSC()) {
    csr_ = CSCToCSR(csc_);
  } else {
    LOG(FATAL) << "SparseMatrix does not have any sparse format";
  }
}

void SparseMatrix::_CreateCSC() {
  if (HasCSC()) return;
  if (HasCOO()) {
    csc_ = COOToCSC(coo_);
  } else if (HasCSR()) {
    csc_ = CSRToCSC(csr_);
  } else {
    LOG(FATAL) << "SparseMatrix does not have any sparse format";
  }
}

}  // namespace sparse
}  // namespace dgl
