/**
 *  Copyright (c) 2022 by Contributors
 * @file sparse/sparse_format.h
 * @brief DGL C++ sparse format header.
 */
#ifndef SPARSE_SPARSE_FORMAT_H_
#define SPARSE_SPARSE_FORMAT_H_

// clang-format off
#include <sparse/dgl_headers.h>
// clang-format on

#include <torch/custom_class.h>
#include <torch/script.h>

#include <memory>

namespace dgl {
namespace sparse {

/** @brief SparseFormat enumeration */
enum SparseFormat { kCOO, kCSR, kCSC };

/** @brief CSR sparse structure */
struct CSR {
  // CSR format index pointer array of the matrix
  torch::Tensor indptr;
  // CSR format index array of the matrix
  torch::Tensor indices;
  // The element order of the sparse format. In the SparseMatrix, we have data
  // (value_) for each non-zero value. The order of non-zero values in (value_)
  // may differ from the order of non-zero entries in CSR. So we store
  // `value_indices` in CSR to indicate its relative non-zero value order to the
  // SparseMatrix. With `value_indices`, we can retrieve the correct value for
  // CSR, i.e., `value_[value_indices]`. If `value_indices` is not defined, this
  // CSR follows the same non-zero value order as the SparseMatrix.
  torch::optional<torch::Tensor> value_indices;
};

/** @brief COO sparse structure */
struct COO {
  // COO format row array of the matrix
  torch::Tensor row;
  // COO format column array of the matrix
  torch::Tensor col;
};

/**
 * @brief Convert a CSR format to COO format
 * @param num_rows Number of rows of the sparse format
 * @param num_cols Number of cols of the sparse format
 * @param csr CSR sparse format
 * @return COO sparse format
 */
std::shared_ptr<COO> CSRToCOO(
    int64_t num_rows, int64_t num_cols, const std::shared_ptr<CSR> csr);

}  // namespace sparse
}  // namespace dgl

#endif  // SPARSE_SPARSE_FORMAT_H_
