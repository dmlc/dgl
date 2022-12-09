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
  CHECK(aten::IsNullArray(dgl_coo.data));
  return std::make_shared<COO>(
      COO{dgl_coo.num_rows, dgl_coo.num_cols, row, col, dgl_coo.row_sorted,
          dgl_coo.col_sorted});
}

aten::COOMatrix COOToOldDGLCOO(const std::shared_ptr<COO>& coo) {
  auto row = TorchTensorToDGLArray(coo->row);
  auto col = TorchTensorToDGLArray(coo->col);
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

std::shared_ptr<COO> COOTranspose(const std::shared_ptr<COO>& coo) {
  auto dgl_coo = COOToOldDGLCOO(coo);
  auto dgl_coo_tr = aten::COOTranspose(dgl_coo);
  return COOFromOldDGLCOO(dgl_coo_tr);
}

}  // namespace sparse
}  // namespace dgl
