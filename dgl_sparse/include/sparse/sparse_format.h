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
#include <utility>

namespace dgl {
namespace sparse {

/** @brief SparseFormat enumeration. */
enum SparseFormat { kCOO, kCSR, kCSC, kDiag };

/** @brief COO sparse structure. */
struct COO {
  /** @brief The shape of the matrix. */
  int64_t num_rows = 0, num_cols = 0;
  /**
   * @brief COO tensor of shape (2, nnz), stacking the row and column indices.
   */
  torch::Tensor indices;
  /** @brief Whether the row indices are sorted. */
  bool row_sorted = false;
  /** @brief Whether the column indices per row are sorted. */
  bool col_sorted = false;
};

/** @brief CSR sparse structure. */
struct CSR {
  /** @brief The dense shape of the matrix. */
  int64_t num_rows = 0, num_cols = 0;
  /** @brief CSR format index pointer array of the matrix. */
  torch::Tensor indptr;
  /** @brief CSR format index array of the matrix. */
  torch::Tensor indices;
  /** @brief Data index tensor. When it is null, assume it is from 0 to NNZ - 1.
   */
  torch::optional<torch::Tensor> value_indices;
  /** @brief Whether the column indices per row are sorted. */
  bool sorted = false;
};

struct Diag {
  /** @brief The dense shape of the matrix. */
  int64_t num_rows = 0, num_cols = 0;
};

/** @brief Convert an old DGL COO format to a COO in the sparse library. */
std::shared_ptr<COO> COOFromOldDGLCOO(const aten::COOMatrix& dgl_coo);

/** @brief Convert a COO in the sparse library to an old DGL COO matrix. */
aten::COOMatrix COOToOldDGLCOO(const std::shared_ptr<COO>& coo);

/** @brief Convert an old DGL CSR format to a CSR in the sparse library. */
std::shared_ptr<CSR> CSRFromOldDGLCSR(const aten::CSRMatrix& dgl_csr);

/** @brief Convert a CSR in the sparse library to an old DGL CSR matrix. */
aten::CSRMatrix CSRToOldDGLCSR(const std::shared_ptr<CSR>& csr);

/**
 *  @brief Convert a COO and its nonzero values to a Torch COO matrix.
 *  @param coo The COO format in the sparse library
 *  @param value Values of the sparse matrix
 *
 *  @return Torch Sparse Tensor in COO format
 */
torch::Tensor COOToTorchCOO(
    const std::shared_ptr<COO>& coo, torch::Tensor value);

/** @brief Convert a CSR format to COO format. */
std::shared_ptr<COO> CSRToCOO(const std::shared_ptr<CSR>& csr);

/** @brief Convert a CSC format to COO format. */
std::shared_ptr<COO> CSCToCOO(const std::shared_ptr<CSR>& csc);

/** @brief Convert a COO format to CSR format. */
std::shared_ptr<CSR> COOToCSR(const std::shared_ptr<COO>& coo);

/** @brief Convert a CSC format to CSR format. */
std::shared_ptr<CSR> CSCToCSR(const std::shared_ptr<CSR>& csc);

/** @brief Convert a COO format to CSC format. */
std::shared_ptr<CSR> COOToCSC(const std::shared_ptr<COO>& coo);

/** @brief Convert a CSR format to CSC format. */
std::shared_ptr<CSR> CSRToCSC(const std::shared_ptr<CSR>& csr);

/** @brief Convert a Diag format to COO format. */
std::shared_ptr<COO> DiagToCOO(
    const std::shared_ptr<Diag>& diag,
    const c10::TensorOptions& indices_options);

/** @brief Convert a Diag format to CSR format. */
std::shared_ptr<CSR> DiagToCSR(
    const std::shared_ptr<Diag>& diag,
    const c10::TensorOptions& indices_options);

/** @brief Convert a Diag format to CSC format. */
std::shared_ptr<CSR> DiagToCSC(
    const std::shared_ptr<Diag>& diag,
    const c10::TensorOptions& indices_options);

/** @brief COO transposition. */
std::shared_ptr<COO> COOTranspose(const std::shared_ptr<COO>& coo);

/**
 * @brief Sort the COO matrix by row and column indices.
 * @return A pair of the sorted COO matrix and the permutation indices.
 */
std::pair<std::shared_ptr<COO>, torch::Tensor> COOSort(
    const std::shared_ptr<COO>& coo);

}  // namespace sparse
}  // namespace dgl

#endif  // SPARSE_SPARSE_FORMAT_H_
