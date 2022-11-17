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

std::shared_ptr<COO> CSRToCOO(
    int64_t num_rows, int64_t num_cols, const std::shared_ptr<CSR> csr) {
  auto indptr = TorchTensorToDGLArray(csr->indptr);
  auto indices = TorchTensorToDGLArray(csr->indices);
  bool data_as_order = false;
  runtime::NDArray data = aten::NullArray();
  if (csr->value_indices.has_value()) {
    data_as_order = true;
    data = TorchTensorToDGLArray(csr->value_indices.value());
  }
  auto dgl_csr = aten::CSRMatrix(num_rows, num_cols, indptr, indices, data);
  auto dgl_coo = aten::CSRToCOO(dgl_csr, data_as_order);
  auto row = DGLArrayToTorchTensor(dgl_coo.row);
  auto col = DGLArrayToTorchTensor(dgl_coo.col);
  return std::make_shared<COO>(COO{row, col});
}

}  // namespace sparse
}  // namespace dgl
