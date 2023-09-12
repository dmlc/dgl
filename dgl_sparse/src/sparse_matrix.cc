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

#include "./utils.h"

namespace dgl {
namespace sparse {

SparseMatrix::SparseMatrix(
    const std::shared_ptr<COO>& coo, const std::shared_ptr<CSR>& csr,
    const std::shared_ptr<CSR>& csc, const std::shared_ptr<Diag>& diag,
    torch::Tensor value, const std::vector<int64_t>& shape)
    : coo_(coo),
      csr_(csr),
      csc_(csc),
      diag_(diag),
      value_(value),
      shape_(shape) {
  TORCH_CHECK(
      coo != nullptr || csr != nullptr || csc != nullptr || diag != nullptr,
      "At least one of CSR/COO/CSC/Diag is required to construct a "
      "SparseMatrix.")
  TORCH_CHECK(
      shape.size() == 2, "The shape of a sparse matrix should be ",
      "2-dimensional.");
  // NOTE: Currently all the tensors of a SparseMatrix should on the same
  // device. Do we allow the graph structure and values are on different
  // devices?
  if (coo != nullptr) {
    TORCH_CHECK(coo->indices.dim() == 2);
    TORCH_CHECK(coo->indices.size(0) == 2);
    TORCH_CHECK(coo->indices.size(1) == value.size(0));
    TORCH_CHECK(coo->indices.device() == value.device());
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
  if (diag != nullptr) {
    TORCH_CHECK(value.size(0) == std::min(diag->num_rows, diag->num_cols));
  }
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::FromCOOPointer(
    const std::shared_ptr<COO>& coo, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  return c10::make_intrusive<SparseMatrix>(
      coo, nullptr, nullptr, nullptr, value, shape);
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::FromCSRPointer(
    const std::shared_ptr<CSR>& csr, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  return c10::make_intrusive<SparseMatrix>(
      nullptr, csr, nullptr, nullptr, value, shape);
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::FromCSCPointer(
    const std::shared_ptr<CSR>& csc, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  return c10::make_intrusive<SparseMatrix>(
      nullptr, nullptr, csc, nullptr, value, shape);
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::FromDiagPointer(
    const std::shared_ptr<Diag>& diag, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  return c10::make_intrusive<SparseMatrix>(
      nullptr, nullptr, nullptr, diag, value, shape);
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::FromCOO(
    torch::Tensor indices, torch::Tensor value,
    const std::vector<int64_t>& shape) {
  auto coo =
      std::make_shared<COO>(COO{shape[0], shape[1], indices, false, false});
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

c10::intrusive_ptr<SparseMatrix> SparseMatrix::FromDiag(
    torch::Tensor value, const std::vector<int64_t>& shape) {
  auto diag = std::make_shared<Diag>(Diag{shape[0], shape[1]});
  return SparseMatrix::FromDiagPointer(diag, value, shape);
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::IndexSelect(
    int64_t dim, torch::Tensor ids) {
  auto id_array = TorchTensorToDGLArray(ids);
  bool rowwise = dim == 0;
  auto csr = rowwise ? this->CSRPtr() : this->CSCPtr();
  auto slice_csr = dgl::aten::CSRSliceRows(CSRToOldDGLCSR(csr), id_array);
  auto slice_value =
      this->value().index_select(0, DGLArrayToTorchTensor(slice_csr.data));
  // To prevent potential errors in future conversions to the COO format,
  // where this array might be used as an initialization array for
  // constructing COO representations, it is necessary to clear this array.
  slice_csr.data = dgl::aten::NullArray();
  auto ret = CSRFromOldDGLCSR(slice_csr);
  if (rowwise) {
    return SparseMatrix::FromCSRPointer(
        ret, slice_value, {ret->num_rows, ret->num_cols});
  } else {
    return SparseMatrix::FromCSCPointer(
        ret, slice_value, {ret->num_cols, ret->num_rows});
  }
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::RangeSelect(
    int64_t dim, int64_t start, int64_t end) {
  bool rowwise = dim == 0;
  auto csr = rowwise ? this->CSRPtr() : this->CSCPtr();
  auto slice_csr = dgl::aten::CSRSliceRows(CSRToOldDGLCSR(csr), start, end);
  auto slice_value =
      this->value().index_select(0, DGLArrayToTorchTensor(slice_csr.data));
  // To prevent potential errors in future conversions to the COO format,
  // where this array might be used as an initialization array for
  // constructing COO representations, it is necessary to clear this array.
  slice_csr.data = dgl::aten::NullArray();
  auto ret = CSRFromOldDGLCSR(slice_csr);
  if (rowwise) {
    return SparseMatrix::FromCSRPointer(
        ret, slice_value, {ret->num_rows, ret->num_cols});
  } else {
    return SparseMatrix::FromCSCPointer(
        ret, slice_value, {ret->num_cols, ret->num_rows});
  }
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::Sample(
    int64_t dim, int64_t fanout, torch::Tensor ids, bool replace, bool bias) {
  bool rowwise = dim == 0;
  auto id_array = TorchTensorToDGLArray(ids);
  auto csr = rowwise ? this->CSRPtr() : this->CSCPtr();
  // Slicing matrix.
  auto slice_csr = dgl::aten::CSRSliceRows(CSRToOldDGLCSR(csr), id_array);
  auto slice_value =
      this->value().index_select(0, DGLArrayToTorchTensor(slice_csr.data));
  // Reset value indices.
  slice_csr.data = dgl::aten::NullArray();

  auto prob =
      bias ? TorchTensorToDGLArray(slice_value) : dgl::aten::NullArray();
  auto slice_id =
      dgl::aten::Range(0, id_array.NumElements(), 64, id_array->ctx);
  // Sampling all rows on sliced matrix.
  auto sample_coo =
      dgl::aten::CSRRowWiseSampling(slice_csr, slice_id, fanout, prob, replace);
  auto sample_value =
      slice_value.index_select(0, DGLArrayToTorchTensor(sample_coo.data));
  sample_coo.data = dgl::aten::NullArray();
  auto ret = COOFromOldDGLCOO(sample_coo);
  if (!rowwise) ret = COOTranspose(ret);
  return SparseMatrix::FromCOOPointer(
      ret, sample_value, {ret->num_rows, ret->num_cols});
}

c10::intrusive_ptr<SparseMatrix> SparseMatrix::ValLike(
    const c10::intrusive_ptr<SparseMatrix>& mat, torch::Tensor value) {
  TORCH_CHECK(
      mat->value().size(0) == value.size(0), "The first dimension of ",
      "the old values and the new values must be the same.");
  TORCH_CHECK(
      mat->value().device() == value.device(), "The device of the ",
      "old values and the new values must be the same.");
  const auto& shape = mat->shape();
  if (mat->HasDiag()) {
    return SparseMatrix::FromDiagPointer(mat->DiagPtr(), value, shape);
  }
  if (mat->HasCOO()) {
    return SparseMatrix::FromCOOPointer(mat->COOPtr(), value, shape);
  }
  if (mat->HasCSR()) {
    return SparseMatrix::FromCSRPointer(mat->CSRPtr(), value, shape);
  }
  TORCH_CHECK(mat->HasCSC(), "Invalid sparse format for ValLike.")
  return SparseMatrix::FromCSCPointer(mat->CSCPtr(), value, shape);
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

std::shared_ptr<Diag> SparseMatrix::DiagPtr() {
  TORCH_CHECK(
      diag_ != nullptr,
      "Cannot get Diag sparse format from a non-diagonal sparse matrix");
  return diag_;
}

std::tuple<torch::Tensor, torch::Tensor> SparseMatrix::COOTensors() {
  auto coo = COOPtr();
  return std::make_tuple(coo->indices.index({0}), coo->indices.index({1}));
}

torch::Tensor SparseMatrix::Indices() {
  auto coo = COOPtr();
  return coo->indices;
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
  if (HasDiag()) {
    return SparseMatrix::FromDiag(value, shape);
  } else if (HasCOO()) {
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
  if (HasDiag()) {
    auto indices_options = torch::TensorOptions()
                               .dtype(torch::kInt64)
                               .layout(torch::kStrided)
                               .device(this->device());
    coo_ = DiagToCOO(diag_, indices_options);
  } else if (HasCSR()) {
    coo_ = CSRToCOO(csr_);
  } else if (HasCSC()) {
    coo_ = CSCToCOO(csc_);
  } else {
    LOG(FATAL) << "SparseMatrix does not have any sparse format";
  }
}

void SparseMatrix::_CreateCSR() {
  if (HasCSR()) return;
  if (HasDiag()) {
    auto indices_options = torch::TensorOptions()
                               .dtype(torch::kInt64)
                               .layout(torch::kStrided)
                               .device(this->device());
    csr_ = DiagToCSR(diag_, indices_options);
  } else if (HasCOO()) {
    csr_ = COOToCSR(coo_);
  } else if (HasCSC()) {
    csr_ = CSCToCSR(csc_);
  } else {
    LOG(FATAL) << "SparseMatrix does not have any sparse format";
  }
}

void SparseMatrix::_CreateCSC() {
  if (HasCSC()) return;
  if (HasDiag()) {
    auto indices_options = torch::TensorOptions()
                               .dtype(torch::kInt64)
                               .layout(torch::kStrided)
                               .device(this->device());
    csc_ = DiagToCSC(diag_, indices_options);
  } else if (HasCOO()) {
    csc_ = COOToCSC(coo_);
  } else if (HasCSR()) {
    csc_ = CSRToCSC(csr_);
  } else {
    LOG(FATAL) << "SparseMatrix does not have any sparse format";
  }
}

}  // namespace sparse
}  // namespace dgl
