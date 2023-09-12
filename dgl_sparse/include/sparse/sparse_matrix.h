/**
 *  Copyright (c) 2022 by Contributors
 * @file sparse/sparse_matrix.h
 * @brief DGL C++ sparse matrix header.
 */
#ifndef SPARSE_SPARSE_MATRIX_H_
#define SPARSE_SPARSE_MATRIX_H_

// clang-format off
#include <sparse/dgl_headers.h>
// clang-format on

#include <sparse/sparse_format.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace dgl {
namespace sparse {

/** @brief SparseMatrix bound to Python.  */
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
      const std::shared_ptr<CSR>& csc, const std::shared_ptr<Diag>& diag,
      torch::Tensor value, const std::vector<int64_t>& shape);

  /**
   * @brief Construct a SparseMatrix from a COO format.
   * @param coo The COO format
   * @param value Values of the sparse matrix
   * @param shape Shape of the sparse matrix
   *
   * @return SparseMatrix
   */
  static c10::intrusive_ptr<SparseMatrix> FromCOOPointer(
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
  static c10::intrusive_ptr<SparseMatrix> FromCSRPointer(
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
  static c10::intrusive_ptr<SparseMatrix> FromCSCPointer(
      const std::shared_ptr<CSR>& csc, torch::Tensor value,
      const std::vector<int64_t>& shape);

  /**
   * @brief Construct a SparseMatrix from a Diag format.
   * @param diag The Diag format
   * @param value Values of the sparse matrix
   * @param shape Shape of the sparse matrix
   *
   * @return SparseMatrix
   */
  static c10::intrusive_ptr<SparseMatrix> FromDiagPointer(
      const std::shared_ptr<Diag>& diag, torch::Tensor value,
      const std::vector<int64_t>& shape);

  /**
   * @brief Create a SparseMatrix from tensors in COO format.
   * @param indices COO coordinates with shape (2, nnz).
   * @param value Values of the sparse matrix.
   * @param shape Shape of the sparse matrix.
   *
   * @return SparseMatrix
   */
  static c10::intrusive_ptr<SparseMatrix> FromCOO(
      torch::Tensor indices, torch::Tensor value,
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
  static c10::intrusive_ptr<SparseMatrix> FromCSR(
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
  static c10::intrusive_ptr<SparseMatrix> FromCSC(
      torch::Tensor indptr, torch::Tensor indices, torch::Tensor value,
      const std::vector<int64_t>& shape);

  /**
   * @brief Create a SparseMatrix with Diag format.
   * @param value Values of the sparse matrix
   * @param shape Shape of the sparse matrix
   *
   * @return SparseMatrix
   */
  static c10::intrusive_ptr<SparseMatrix> FromDiag(
      torch::Tensor value, const std::vector<int64_t>& shape);

  /**
   * @brief Create a SparseMatrix by selecting rows or columns based on provided
   * indices.
   *
   * This function allows you to create a new SparseMatrix by selecting specific
   * rows or columns from the original SparseMatrix based on the provided
   * indices. The selection can be performed either row-wise or column-wise,
   * determined by the 'dim' parameter.
   *
   * @param dim Select rows (dim=0) or columns (dim=1).
   * @param ids A tensor containing the indices of the selected rows or columns.
   *
   * @return A new SparseMatrix containing the selected rows or columns.
   *
   * @note The 'dim' parameter should be either 0 (for row-wise selection) or 1
   * (for column-wise selection).
   * @note The 'ids' tensor should contain valid indices within the range of the
   * original SparseMatrix's dimensions.
   */
  c10::intrusive_ptr<SparseMatrix> IndexSelect(int64_t dim, torch::Tensor ids);

  /**
   * @brief Create a SparseMatrix by selecting a range of rows or columns based
   * on provided indices.
   *
   * This function allows you to create a new SparseMatrix by selecting a range
   * of specific rows or columns from the original SparseMatrix based on the
   * provided indices. The selection can be performed either row-wise or
   * column-wise, determined by the 'dim' parameter.
   *
   * @param dim Select rows (dim=0) or columns (dim=1).
   * @param start The starting index (inclusive) of the range.
   * @param end The ending index (exclusive) of the range.
   *
   * @return A new SparseMatrix containing the selected range of rows or
   * columns.
   *
   * @note The 'dim' parameter should be either 0 (for row-wise selection) or 1
   * (for column-wise selection).
   * @note The 'start' and 'end' indices should be valid indices within
   * the valid range of the original SparseMatrix's dimensions.
   */
  c10::intrusive_ptr<SparseMatrix> RangeSelect(
      int64_t dim, int64_t start, int64_t end);

  /**
   * @brief Create a SparseMatrix by sampling elements based on the specified
   * dimension and sample count.
   *
   * If `ids` is provided, this function samples elements from the specified
   * set of row or column IDs, resulting in a sparse matrix containing only
   * the sampled rows or columns.
   *
   * @param dim Select rows (dim=0) or columns (dim=1) for sampling.
   * @param fanout The number of elements to randomly sample from each row or
   * column.
   * @param ids An optional tensor containing row or column IDs from which to
   * sample elements.
   * @param replace Indicates whether repeated sampling of the same element
   * is allowed. If True, repeated sampling is allowed; otherwise, it is not
   * allowed.
   * @param bias An optional boolean flag indicating whether to enable biasing
   * during sampling. If True, the values of the sparse matrix will be used as
   * bias weights, meaning that elements with higher values will be more likely
   * to be sampled. Otherwise, all elements will be sampled uniformly,
   * regardless of their value.
   *
   * @return A new SparseMatrix with the same shape as the original matrix
   * containing the sampled elements.
   *
   * @note If 'replace = false' and there are fewer elements than 'fanout',
   * all non-zero elements will be sampled.
   * @note If 'ids' is not provided, the function will sample from
   * all rows or columns.
   */
  c10::intrusive_ptr<SparseMatrix> Sample(
      int64_t dim, int64_t fanout, torch::Tensor ids, bool replace, bool bias);

  /**
   * @brief Create a SparseMatrix from a SparseMatrix using new values.
   * @param mat An existing sparse matrix
   * @param value New values of the sparse matrix
   *
   * @return SparseMatrix
   */
  static c10::intrusive_ptr<SparseMatrix> ValLike(
      const c10::intrusive_ptr<SparseMatrix>& mat, torch::Tensor value);

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
  /**
   * @return Diagonal format of the sparse matrix. An error will be raised if
   * it does not have a diagonal format.
   */
  std::shared_ptr<Diag> DiagPtr();

  /** @brief Check whether this sparse matrix has COO format. */
  inline bool HasCOO() const { return coo_ != nullptr; }
  /** @brief Check whether this sparse matrix has CSR format. */
  inline bool HasCSR() const { return csr_ != nullptr; }
  /** @brief Check whether this sparse matrix has CSC format. */
  inline bool HasCSC() const { return csc_ != nullptr; }
  /** @brief Check whether this sparse matrix has Diag format. */
  inline bool HasDiag() const { return diag_ != nullptr; }

  /** @return {row, col} tensors in the COO format. */
  std::tuple<torch::Tensor, torch::Tensor> COOTensors();
  /** @return Stacked row and col tensors in the COO format. */
  torch::Tensor Indices();
  /** @return {row, col, value_indices} tensors in the CSR format. */
  std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
  CSRTensors();
  /** @return {row, col, value_indices} tensors in the CSC format. */
  std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
  CSCTensors();

  /** @brief Return the transposition of the sparse matrix. It transposes the
   * first existing sparse format by checking COO, CSR, and CSC.
   */
  c10::intrusive_ptr<SparseMatrix> Transpose() const;

  /**
   * @brief Return a new coalesced matrix.
   *
   * A coalesced sparse matrix satisfies the following properties:
   *   - the indices of the non-zero elements are unique,
   *   - the indices are sorted in lexicographical order.
   *
   * @return A coalesced sparse matrix.
   */
  c10::intrusive_ptr<SparseMatrix> Coalesce();

  /**
   * @brief Return true if this sparse matrix contains duplicate indices.
   * @return A bool flag.
   */
  bool HasDuplicate();

 private:
  /** @brief Create the COO format for the sparse matrix internally */
  void _CreateCOO();
  /** @brief Create the CSR format for the sparse matrix internally */
  void _CreateCSR();
  /** @brief Create the CSC format for the sparse matrix internally */
  void _CreateCSC();

  // COO/CSC/CSR/Diag pointers. Nullptr indicates non-existence.
  std::shared_ptr<COO> coo_;
  std::shared_ptr<CSR> csr_, csc_;
  std::shared_ptr<Diag> diag_;
  // Value of the SparseMatrix
  torch::Tensor value_;
  // Shape of the SparseMatrix
  const std::vector<int64_t> shape_;
};
}  // namespace sparse
}  // namespace dgl

#endif  //  SPARSE_SPARSE_MATRIX_H_
