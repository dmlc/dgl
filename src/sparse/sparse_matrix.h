/*!
 *  Copyright (c) 2022 by Contributors
 * \file sparse/sparse_matrix.h
 * \brief DGL C++ sparse matrix header
 */
#ifndef DGL_SPARSE_SPARSE_MATRIX_H_
#define DGL_SPARSE_SPARSE_MATRIX_H_

#include <torch/custom_class.h>
#include <torch/script.h>

#include <memory>

namespace dgl {
namespace sparse {

enum SparseFormat { kCOO, kCSR, kCSC };

struct CSR {
  torch::Tensor indptr, indices;
  torch::optional<torch::Tensor> eids;
};

struct COO {
  torch::Tensor row, col;
  torch::optional<torch::Tensor> eids;
};

class SparseMatrix : public torch::CustomClassHolder {
 public:
  SparseMatrix(
      const std::shared_ptr<COO>& coo, torch::Tensor value,
      const std::vector<int64_t>& shape)
      : coo_(coo), value_(value), shape_(shape) {
    csc_ = csr_ = nullptr;
  }

  torch::Tensor Value() const { return value_; }
  const std::vector<int64_t>& Shape() const { return shape_; }

  std::shared_ptr<COO> COOPtr() {
    _CreateCOOIfNotExist();
    return coo_;
  }
  std::shared_ptr<CSR> CSRPtr() {
    _CreateCSRIfNotExist();
    return csr_;
  }
  std::shared_ptr<CSR> CSCPtr() {
    _CreateCSCIfNotExist();
    return csc_;
  }
  bool HasCOO() const { return coo_ != nullptr; }
  bool HasCSR() const { return csr_ != nullptr; }
  bool HasCSC() const { return csc_ != nullptr; }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> COOTensors();
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CSRTensors();
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CSCTensors();

 private:
  void _CreateCOOIfNotExist() {
    if (coo_ == nullptr) {
      // TODO: Create COO
    }
  }

  void _CreateCSRIfNotExist() {
    if (csr_ == nullptr) {
      // TODO: Create CSR
    }
  }

  void _CreateCSCIfNotExist() {
    if (csc_ == nullptr) {
      // TODO: Create CSC
    }
  }

  std::shared_ptr<COO> coo_;
  std::shared_ptr<CSR> csr_, csc_;
  torch::Tensor value_;
  std::vector<int64_t> shape_;
};

c10::intrusive_ptr<SparseMatrix> CreateFromCOOPtr(
    const std::shared_ptr<COO>& coo, torch::Tensor value,
    const std::vector<int64_t>& shape);

c10::intrusive_ptr<SparseMatrix> CreateFromCOO(
    torch::Tensor row, torch::Tensor col, torch::Tensor value,
    const std::vector<int64_t>& shape);

}  // namespace sparse
}  // namespace dgl

#endif  //  DGL_SPARSE_SPARSE_MATRIX_H_