/**
 *  Copyright (c) 2022 by Contributors
 * @file sparse/sparse_matrix.h
 * @brief DGL C++ sparse matrix header
 */
#ifndef SPARSE_SPARSE_MATRIX_H_
#define SPARSE_SPARSE_MATRIX_H_

#include <torch/custom_class.h>
#include <torch/script.h>

#include <memory>
#include <vector>

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

/** @brief SparseMatrix bound to Python  */
class SparseMatrix : public torch::CustomClassHolder {
 public:
  /**
   * @brief General constructor to construct a sparse matrix for different
   * sparse formats. At least one of the sparse formats should be provided,
   * while others could be nullptrs.
   *
   * @param coo The COO format.
   * @param csr The CSR format.
   * @param csc The CSC format.
   * @param value Value of the sparse matrix.
   * @param shape Shape of the sparse matrix.
   */
  SparseMatrix(
      const std::shared_ptr<COO>& coo, const std::shared_ptr<CSR>& csr,
      const std::shared_ptr<CSR>& csc, torch::Tensor value,
      const std::vector<int64_t>& shape);

  /**
   * @brief Construct a SparseMatrix from a COO format.
   * @param coo The COO format
   * @param value Values of the sparse matrix
   * @param shape Shape of the sparse matrix
   *
   * @return SparseMatrix
   */
  static c10::intrusive_ptr<SparseMatrix> FromCOO(
      const std::shared_ptr<COO>& coo, torch::Tensor value,
      const std::vector<int64_t>& shape);

  /**
   * @brief Construct a SparseMatrix from a CSR format.
   * @param csr The CSR format
   * @param value Values of the sparse matrix
   * @param shape Shape of the sparse matrix
   *
   * @return SparseMatrix
   */
  static c10::intrusive_ptr<SparseMatrix> FromCSR(
      const std::shared_ptr<CSR>& csr, torch::Tensor value,
      const std::vector<int64_t>& shape);

  /**
   * @brief Construct a SparseMatrix from a CSC format.
   * @param csc The CSC format
   * @param value Values of the sparse matrix
   * @param shape Shape of the sparse matrix
   *
   * @return SparseMatrix
   */
  static c10::intrusive_ptr<SparseMatrix> FromCSC(
      const std::shared_ptr<CSR>& csc, torch::Tensor value,
      const std::vector<int64_t>& shape);

  /** @return Value of the sparse matrix. */
  inline torch::Tensor value() const { return value_; }
  /** @return Shape of the sparse matrix. */
  inline const std::vector<int64_t>& shape() const { return shape_; }
  /** @return Number of non-zero values */
  inline int64_t nnz() const { return value_.size(0); }
  /** @return Non-zero value data type */
  inline caffe2::TypeMeta dtype() const { return value_.dtype(); }
  /** @return Device of the sparse matrix */
  inline torch::Device device() const { return value_.device(); }

  /** @return COO of the sparse matrix. The COO is created if not exists. */
  std::shared_ptr<COO> COOPtr();
  /** @return CSR of the sparse matrix. The CSR is created if not exists. */
  std::shared_ptr<CSR> CSRPtr();
  /** @return CSC of the sparse matrix. The CSC is created if not exists. */
  std::shared_ptr<CSR> CSCPtr();

  /** @brief Check whether this sparse matrix has COO format. */
  inline bool HasCOO() const { return coo_ != nullptr; }
  /** @brief Check whether this sparse matrix has CSR format. */
  inline bool HasCSR() const { return csr_ != nullptr; }
  /** @brief Check whether this sparse matrix has CSC format. */
  inline bool HasCSC() const { return csc_ != nullptr; }

  /** @return {row, col, value} tensors in the COO format. */
  std::vector<torch::Tensor> COOTensors();
  /** @return {row, col, value} tensors in the CSR format. */
  std::vector<torch::Tensor> CSRTensors();
  /** @return {row, col, value} tensors in the CSC format. */
  std::vector<torch::Tensor> CSCTensors();

 private:
  /** @brief Create the COO format for the sparse matrix internally */
  void _CreateCOO();
  /** @brief Create the CSR format for the sparse matrix internally */
  void _CreateCSR();
  /** @brief Create the CSC format for the sparse matrix internally */
  void _CreateCSC();

  // COO/CSC/CSR pointers. Nullptr indicates non-existence.
  std::shared_ptr<COO> coo_;
  std::shared_ptr<CSR> csr_, csc_;
  // Value of the SparseMatrix
  torch::Tensor value_;
  // Shape of the SparseMatrix
  const std::vector<int64_t> shape_;
};

/**
 * @brief Create a SparseMatrix from tensors in COO format.
 * @param row Row indices of the COO.
 * @param col Column indices of the COO.
 * @param value Values of the sparse matrix.
 * @param shape Shape of the sparse matrix.
 *
 * @return SparseMatrix
 */
c10::intrusive_ptr<SparseMatrix> CreateFromCOO(
    torch::Tensor row, torch::Tensor col, torch::Tensor value,
    const std::vector<int64_t>& shape);

/**
 * @brief Create a SparseMatrix from tensors in CSR format.
 * @param indptr Index pointer array of the CSR
 * @param indices Indices array of the CSR
 * @param value Values of the sparse matrix
 * @param shape Shape of the sparse matrix
 *
 * @return SparseMatrix
 */
c10::intrusive_ptr<SparseMatrix> CreateFromCSR(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor value,
    const std::vector<int64_t>& shape);

/**
 * @brief Create a SparseMatrix from tensors in CSC format.
 * @param indptr Index pointer array of the CSC
 * @param indices Indices array of the CSC
 * @param value Values of the sparse matrix
 * @param shape Shape of the sparse matrix
 *
 * @return SparseMatrix
 */
c10::intrusive_ptr<SparseMatrix> CreateFromCSC(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor value,
    const std::vector<int64_t>& shape);

}  // namespace sparse
}  // namespace dgl

#endif  //  SPARSE_SPARSE_MATRIX_H_
