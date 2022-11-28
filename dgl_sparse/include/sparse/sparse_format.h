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

/** @brief SparseFormat enumeration. */
enum SparseFormat { kCOO, kCSR, kCSC };

/** @brief COO sparse structure. */
struct COO {
  /** @brief The shape of the matrix. */
  int64_t num_rows = 0, num_cols = 0;
  /** @brief COO format row indices array of the matrix. */
  torch::Tensor row;
  /** @brief COO format column indices array of the matrix. */
  torch::Tensor col;
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

/** @brief Convert an old DGL COO format to a COO in the sparse library. */
std::shared_ptr<COO> COOFromOldDGLCOO(const aten::COOMatrix& dgl_coo);

/** @brief Convert a COO in the sparse library to an old DGL COO matrix. */
aten::COOMatrix COOToOldDGLCOO(const std::shared_ptr<COO>& coo);

/** @brief Convert an old DGL CSR format to a CSR in the sparse library. */
std::shared_ptr<CSR> CSRFromOldDGLCSR(const aten::CSRMatrix& dgl_csr);

/** @brief Convert a CSR in the sparse library to an old DGL CSR matrix. */
aten::CSRMatrix CSRToOldDGLCSR(const std::shared_ptr<CSR>& csr);

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

}  // namespace sparse
}  // namespace dgl

#endif  // SPARSE_SPARSE_FORMAT_H_
