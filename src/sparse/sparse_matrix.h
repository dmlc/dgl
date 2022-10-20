/*!
 *  Copyright (c) 2018 by Contributors
 * \file sparse_matrix.h
 * \brief DGL C++ sparse matrix implementations
 */

#ifndef DGL_SPARSE_SPARSE_MATRIX_H_
#define DGL_SPARSE_SPARSE_MATRIX_H_

#include <torch/custom_class.h>
#include <torch/script.h>

namespace dgl {
namespace sparse {

struct CSR {
  torch::Tensor indptr, indices;
  torch::optional<torch::Tensor> val;
};

struct COO {
  torch::Tensor row, col;
  torch::optional<torch::Tensor> val;
};

struct SparseMatrix : public torch::CustomClassHolder {
  SparseMatrix(torch::Tensor row, torch::Tensor col,
               torch::optional<torch::Tensor> val,
               const std::vector<int64_t> &shape)
      : shape(shape) {
    coo = std::make_unique<COO>(row, col, val);
    csr = nullptr;
  }

  torch::Tensor Row() const { return COO()->row; }

  std::unique_ptr<COO> COO() {
    _CreateCOOIfNotExist();
    return coo;
  }
  std::unique_ptr<CSR> CSR() {
    _CreateCSRIfNotExist();
    return csr;
  }

  void _CreateCOOIfNotExist() {
    if (coo == nullptr) {
      // Create COO
    }
  }

  void _CreateCSRIfNotExist() {
    if (csr == nullptr) {
      // Create CSR
    }
  }

  std::unique_ptr<CSR> csr;
  std::unique_ptr<COO> coo;
  std::vector<int64_t> shape;
};

} // namespace sparse
} // namespace dgl

#endif //  DGL_SPARSE_SPARSE_MATRIX_H_