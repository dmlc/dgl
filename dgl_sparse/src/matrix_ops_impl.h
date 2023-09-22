/**
 *  Copyright (c) 2023 by Contributors
 * @file matrix_ops_impl.h
 * @brief DGL C++ sparse matrix operator implementations.
 */
#ifndef DGL_SPARSE_MATRIX_OPS_IMPL_H_
#define DGL_SPARSE_MATRIX_OPS_IMPL_H_

#include <sparse/sparse_format.h>
#include <sparse/sparse_matrix.h>

#include <tuple>
#include <vector>

#include "./utils.h"

namespace dgl {
namespace sparse {

std::tuple<torch::Tensor, torch::Tensor> CompactId(
    torch::Tensor &row, torch::optional<torch::Tensor> &leading_indices);

template <c10::DeviceType XPU, typename IdType, typename ValType>
std::tuple<c10::intrusive_ptr<SparseMatrix>, torch::optional<torch::Tensor>>
CompactImplCOO(
    const c10::intrusive_ptr<SparseMatrix> &mat, int64_t dim,
    torch::optional<torch::Tensor> leading_indices) {
  torch::Tensor row, col;
  auto coo = mat->COOTensors();
  if (dim == 0)
    std::tie(row, col) = coo;
  else
    std::tie(col, row) = coo;

  torch::Tensor new_row, uniqued;
  std::tie(new_row, uniqued) = CompactId(row, leading_indices);

  if (dim == 0) {
    auto ret = SparseMatrix::FromCOO(
        torch::stack({new_row, col}, 0), mat->value(),
        std::vector<int64_t>{uniqued.numel(), mat->shape()[1]});
    auto ret_idx = torch::optional<torch::Tensor>(uniqued.flip(-1));
    return {ret, ret_idx};
  } else {
    auto ret = SparseMatrix::FromCOO(
        torch::stack({col, new_row}, 0), mat->value(),
        std::vector<int64_t>{mat->shape()[0], uniqued.numel()});
    auto ret_idx = torch::optional<torch::Tensor>(uniqued.flip(-1));
    return {ret, ret_idx};
  }
}

template <c10::DeviceType XPU, typename IdType, typename ValType>
std::tuple<c10::intrusive_ptr<SparseMatrix>, torch::optional<torch::Tensor>>
CompactImplCSC(
    const c10::intrusive_ptr<SparseMatrix> &mat, int64_t dim,
    torch::optional<torch::Tensor> leading_indices) {
  std::shared_ptr<dgl::sparse::CSR> csr;
  if (dim == 0)
    csr = mat->CSCPtr();
  else
    csr = mat->CSRPtr();

  torch::Tensor new_indices, uniqued;
  std::tie(new_indices, uniqued) = CompactId(csr->indices, leading_indices);

  if (dim == 0) {
    auto ret = SparseMatrix::FromCSC(
        csr->indptr, new_indices, mat->value(),
        std::vector<int64_t>{uniqued.numel(), mat->shape()[1]});
    auto ret_idx = torch::optional<torch::Tensor>(uniqued.flip(-1));
    return {ret, ret_idx};
  } else {
    auto ret = SparseMatrix::FromCSR(
        csr->indptr, new_indices, mat->value(),
        std::vector<int64_t>{mat->shape()[0], uniqued.numel()});
    auto ret_idx = torch::optional<torch::Tensor>(uniqued.flip(-1));
    return {ret, ret_idx};
  }
}

template <c10::DeviceType XPU, typename IdType, typename ValType>
std::tuple<c10::intrusive_ptr<SparseMatrix>, torch::optional<torch::Tensor>>
CompactImpl(
    const c10::intrusive_ptr<SparseMatrix> &mat, int64_t dim,
    torch::optional<torch::Tensor> leading_indices) {
  if (dim == 0) {
    if (mat->HasCSC())
      return CompactImplCSC<XPU, IdType, ValType>(mat, dim, leading_indices);
    else
      return CompactImplCOO<XPU, IdType, ValType>(mat, dim, leading_indices);
  } else {
    if (mat->HasCSR())
      return CompactImplCSC<XPU, IdType, ValType>(mat, dim, leading_indices);
    else
      return CompactImplCOO<XPU, IdType, ValType>(mat, dim, leading_indices);
  }
}

}  // namespace sparse
}  // namespace dgl

#endif  // DGL_SPARSE_MATRIX_OPS_IMPL_H_
