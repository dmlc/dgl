/**
 *  Copyright (c) 2022 by Contributors
 * @file sparse_format.cc
 * @brief DGL C++ sparse format implementations.
 */
// clang-format off
#include <sparse/dgl_headers.h>
// clang-format on

#include <sparse/sparse_format.h>

#include "./utils.h"

namespace dgl {
namespace sparse {

std::shared_ptr<COO> COOFromOldDGLCOO(const aten::COOMatrix& dgl_coo) {
  auto row = DGLArrayToTorchTensor(dgl_coo.row);
  auto col = DGLArrayToTorchTensor(dgl_coo.col);
  TORCH_CHECK(aten::IsNullArray(dgl_coo.data));
  auto indices = torch::stack({row, col});
  return std::make_shared<COO>(
      COO{dgl_coo.num_rows, dgl_coo.num_cols, indices, dgl_coo.row_sorted,
          dgl_coo.col_sorted});
}

aten::COOMatrix COOToOldDGLCOO(const std::shared_ptr<COO>& coo) {
  auto row = TorchTensorToDGLArray(coo->indices.index({0}));
  auto col = TorchTensorToDGLArray(coo->indices.index({1}));
  return aten::COOMatrix(
      coo->num_rows, coo->num_cols, row, col, aten::NullArray(),
      coo->row_sorted, coo->col_sorted);
}

std::shared_ptr<CSR> CSRFromOldDGLCSR(const aten::CSRMatrix& dgl_csr) {
  auto indptr = DGLArrayToTorchTensor(dgl_csr.indptr);
  auto indices = DGLArrayToTorchTensor(dgl_csr.indices);
  auto value_indices = DGLArrayToOptionalTorchTensor(dgl_csr.data);
  return std::make_shared<CSR>(
      CSR{dgl_csr.num_rows, dgl_csr.num_cols, indptr, indices, value_indices,
          dgl_csr.sorted});
}

aten::CSRMatrix CSRToOldDGLCSR(const std::shared_ptr<CSR>& csr) {
  auto indptr = TorchTensorToDGLArray(csr->indptr);
  auto indices = TorchTensorToDGLArray(csr->indices);
  auto data = OptionalTorchTensorToDGLArray(csr->value_indices);
  return aten::CSRMatrix(
      csr->num_rows, csr->num_cols, indptr, indices, data, csr->sorted);
}

torch::Tensor COOToTorchCOO(
    const std::shared_ptr<COO>& coo, torch::Tensor value) {
  torch::Tensor indices = coo->indices;
  if (value.ndimension() == 2) {
    return torch::sparse_coo_tensor(
        indices, value, {coo->num_rows, coo->num_cols, value.size(1)});
  } else {
    return torch::sparse_coo_tensor(
        indices, value, {coo->num_rows, coo->num_cols});
  }
}

std::shared_ptr<COO> CSRToCOO(const std::shared_ptr<CSR>& csr) {
  auto dgl_csr = CSRToOldDGLCSR(csr);
  auto dgl_coo = aten::CSRToCOO(dgl_csr, csr->value_indices.has_value());
  return COOFromOldDGLCOO(dgl_coo);
}

std::shared_ptr<COO> CSCToCOO(const std::shared_ptr<CSR>& csc) {
  auto dgl_csc = CSRToOldDGLCSR(csc);
  auto dgl_coo = aten::CSRToCOO(dgl_csc, csc->value_indices.has_value());
  dgl_coo = aten::COOTranspose(dgl_coo);
  return COOFromOldDGLCOO(dgl_coo);
}

std::shared_ptr<CSR> COOToCSR(const std::shared_ptr<COO>& coo) {
  auto dgl_coo = COOToOldDGLCOO(coo);
  auto dgl_csr = aten::COOToCSR(dgl_coo);
  return CSRFromOldDGLCSR(dgl_csr);
}

std::shared_ptr<CSR> CSCToCSR(const std::shared_ptr<CSR>& csc) {
  auto dgl_csc = CSRToOldDGLCSR(csc);
  auto dgl_csr = aten::CSRTranspose(dgl_csc);
  return CSRFromOldDGLCSR(dgl_csr);
}

std::shared_ptr<CSR> COOToCSC(const std::shared_ptr<COO>& coo) {
  auto dgl_coo = COOToOldDGLCOO(coo);
  auto dgl_coo_transpose = aten::COOTranspose(dgl_coo);
  auto dgl_csc = aten::COOToCSR(dgl_coo_transpose);
  return CSRFromOldDGLCSR(dgl_csc);
}

std::shared_ptr<CSR> CSRToCSC(const std::shared_ptr<CSR>& csr) {
  auto dgl_csr = CSRToOldDGLCSR(csr);
  auto dgl_csc = aten::CSRTranspose(dgl_csr);
  return CSRFromOldDGLCSR(dgl_csc);
}

std::shared_ptr<COO> DiagToCOO(
    const std::shared_ptr<Diag>& diag,
    const c10::TensorOptions& indices_options) {
  int64_t nnz = std::min(diag->num_rows, diag->num_cols);
  auto indices = torch::arange(nnz, indices_options).repeat({2, 1});
  return std::make_shared<COO>(
      COO{diag->num_rows, diag->num_cols, indices, true, true});
}

std::shared_ptr<CSR> DiagToCSR(
    const std::shared_ptr<Diag>& diag,
    const c10::TensorOptions& indices_options) {
  int64_t nnz = std::min(diag->num_rows, diag->num_cols);
  auto indptr = torch::full(diag->num_rows + 1, nnz, indices_options);
  auto nnz_range = torch::arange(nnz + 1, indices_options);
  indptr.index_put_({nnz_range}, nnz_range);
  auto indices = torch::arange(nnz, indices_options);
  return std::make_shared<CSR>(
      CSR{diag->num_rows, diag->num_cols, indptr, indices,
          torch::optional<torch::Tensor>(), true});
}

std::shared_ptr<CSR> DiagToCSC(
    const std::shared_ptr<Diag>& diag,
    const c10::TensorOptions& indices_options) {
  int64_t nnz = std::min(diag->num_rows, diag->num_cols);
  auto indptr = torch::full(diag->num_cols + 1, nnz, indices_options);
  auto nnz_range = torch::arange(nnz + 1, indices_options);
  indptr.index_put_({nnz_range}, nnz_range);
  auto indices = torch::arange(nnz, indices_options);
  return std::make_shared<CSR>(
      CSR{diag->num_cols, diag->num_rows, indptr, indices,
          torch::optional<torch::Tensor>(), true});
}

std::shared_ptr<COO> COOTranspose(const std::shared_ptr<COO>& coo) {
  auto dgl_coo = COOToOldDGLCOO(coo);
  auto dgl_coo_tr = aten::COOTranspose(dgl_coo);
  return COOFromOldDGLCOO(dgl_coo_tr);
}

std::pair<std::shared_ptr<COO>, torch::Tensor> COOSort(
    const std::shared_ptr<COO>& coo) {
  auto encoded_coo =
      coo->indices.index({0}) * coo->num_cols + coo->indices.index({1});
  torch::Tensor sorted, perm;
  std::tie(sorted, perm) = encoded_coo.sort();
  auto sorted_coo = std::make_shared<COO>(
      COO{coo->num_rows, coo->num_cols, coo->indices.index_select(1, perm),
          true, true});
  return {sorted_coo, perm};
}

}  // namespace sparse
}  // namespace dgl
