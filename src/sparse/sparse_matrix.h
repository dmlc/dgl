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

/*! \brief SparseFormat enumeration */
enum SparseFormat { kCOO, kCSR, kCSC };

/*! \brief CSR sparse structure */
struct CSR {
  torch::Tensor indptr, indices;
  // The element order of the sparse format. In the SparseMatrix, we have data
  // (value_) for each non-zero value. The order of non-zero values in (value_)
  // may differ from the order of non-zero entries in the sparse format. So we
  // store `e_order` in a sparse format to indicate its relative element order
  // to the SparseMatrix. With `e_order`, we can retrieve the correct data for a
  // sparse format, i.e., `value_[e_order]`. If `e_order` is not defined, this
  // sparse format follows the same non-zero value order as the SparseMatrix.
  torch::optional<torch::Tensor> e_order;
};

/*! \brief COO sparse structure */
struct COO {
  torch::Tensor row, col;
  // The element order of the sparse format. In the SparseMatrix, we have data
  // (value_) for each non-zero value. The order of non-zero values in (value_)
  // may differ from the order of non-zero entries in the sparse format. So we
  // store `e_order` in a sparse format to indicate its relative element order
  // to the SparseMatrix. With `e_order`, we can retrieve the correct data for a
  // sparse format, i.e., `value_[e_order]`. If `e_order` is not defined, this
  // sparse format follows the same non-zero value order as the SparseMatrix.
  torch::optional<torch::Tensor> e_order;
};

/*! \brief SparseMatrix bound to Python  */
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

  // COO/CSC/CSR pointers. Nullptr indicates non-existence.
  std::shared_ptr<COO> coo_;
  std::shared_ptr<CSR> csr_, csc_;
  // Value of the SparseMatrix
  torch::Tensor value_;
  // Shape of the SparseMatrix
  std::vector<int64_t> shape_;
};

/*!
 * \brief Constructing a SparseMatrix from a COO pointer.
 * \param coo COO pointer
 * \param value Values of the sparse matrix
 * \param shape Shape of the sparse matrix
 */
c10::intrusive_ptr<SparseMatrix> CreateFromCOOPtr(
    const std::shared_ptr<COO>& coo, torch::Tensor value,
    const std::vector<int64_t>& shape);

/*!
 * \brief Constructing a SparseMatrix from COO Pytorch tensors.
 * \param row Row indices of the COO
 * \param col Column indices of the COO
 * \param value Values of the sparse matrix
 * \param shape Shape of the sparse matrix
 */
c10::intrusive_ptr<SparseMatrix> CreateFromCOO(
    torch::Tensor row, torch::Tensor col, torch::Tensor value,
    const std::vector<int64_t>& shape);

}  // namespace sparse
}  // namespace dgl

#endif  //  DGL_SPARSE_SPARSE_MATRIX_H_