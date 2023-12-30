/**
 *  Copyright (c) 2022 by Contributors
 * @file matmul.h
 * @brief DGL sparse matrix multiplication functions.
 */
#ifndef DGL_SPARSE_MATMUL_H_
#define DGL_SPARSE_MATMUL_H_

#include <sparse/sparse_matrix.h>
#include <torch/script.h>

#include <string>

namespace dgl {
namespace sparse {

/**
 * @brief Perform a matrix multiplication of the sparse matrix and dense
 * matrix. It uses the sparse formats of `sparse_mat` and non-zero values of
 * `sparse_val` for SpMM. The `sparse_val` must be 1-dimensional. If the sparse
 * matrix has shape (n, m), the dense matrix must have shape (m, k). And
 * the returned dense matrix has shape (n, k).
 *
 * This function does not take care of autograd.
 *
 * @param sparse_mat The sparse matrix.
 * @param sparse_val Non-zero values of the sparse matrix.
 * @param dense_mat The dense matrix.
 * @param transpose_sparse Whether the sparse_mat is transposed.
 *
 * @return Dense tensor.
 */
torch::Tensor SpMMNoAutoGrad(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat,
    torch::Tensor sparse_val, torch::Tensor dense_mat, bool transpose_sparse);

/**
 * @brief Perform a sampled matrix multiplication of a sparse matrix and two
 * dense matrices. It calculates `(mat1 @ mat2_tr^T) * spy(A)` and does consider
 * the values of the sparse matrix. For efficiency, `mat2_tr` is the
 * transposition of the matrix to be multiplied. If the sparse matrix has shape
 * (n, m), `mat1` and `mat2_tr` must have shapes of `(n, k)` and `(m,
 * k)`respectively. And the returned tensor has shape
 * `(sparse_matrix->nnz(),)`.
 *
 * This function does not take care of autograd.
 *
 * @param sparse_mat The sparse matrix.
 * @param mat1 The first dense matrix.
 * @param mat2_tr Transposition of the second matrix.
 *
 * @return Dense tensor.
 */
torch::Tensor SDDMMNoAutoGrad(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat, torch::Tensor mat1,
    torch::Tensor mat2_tr);

/**
 * @brief Broadcast the dense feature to the nonzero entries and then compute
 * x_e = \phi(x_e, x_v) on the dimension dim, where x_e is the nonzero value,
 * x_v is the dense feature, and \phi is add, sub, mul, or div. dim = 0 or 1
 * means column-wise or row-wise broadcast respectively.
 *
 * This function does not take care of autograd.
 *
 * @param sparse_mat The sparse matrix with N rows and (nnz, D) nonzero values
 * @param dense_mat Dense feature of shape (N, D)
 * @param op Operator, can be add, sub, mul, or div
 * @param dim The dimension to broadcast.
 *
 * @return Dense tensor of shape (nnz, D)
 */
torch::Tensor BroadcastOpNoAutoGrad(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat, torch::Tensor dense_mat,
    const std::string& op, int64_t dim);

/**
 * @brief Broadcast the dense feature to the nonzero entries and then compute
 * x_e = x_e - x_v on the dimension dim, where x_e is the nonzero value, x_v is
 * the dense feature. dim = 0 or 1 means column-wise or row-wise broadcast
 * respectively.
 *
 * This function does not take care of autograd.
 *
 * @param sparse_mat The sparse matrix with N rows and (nnz, D) nonzero values
 * @param dense_mat Dense feature of shape (N, D)
 * @param dim The dimension to broadcast.
 *
 * @return Dense tensor of shape (nnz, D)
 */
torch::Tensor BroadcastSubNoAutoGrad(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat, torch::Tensor dense_mat,
    int64_t dim);

/**
 * @brief Broadcast the dense feature to the nonzero entries and then compute
 * x_e = x_e / x_v on the dimension dim, where x_e is the nonzero value, x_v is
 * the dense feature. dim = 0 or 1 means column-wise or row-wise broadcast
 * respectively.
 *
 * This function does not take care of autograd.
 *
 * @param sparse_mat The sparse matrix with N rows and (nnz, D) nonzero values
 * @param dense_mat Dense feature of shape (N, D)
 * @param dim The dimension to broadcast.
 *
 * @return Dense tensor of shape (nnz, D)
 */
torch::Tensor BroadcastDivNoAutoGrad(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat, torch::Tensor dense_mat,
    int64_t dim);

/**
 * @brief Broadcast the dense feature to the nonzero entries and then compute
 * x_e = x_e * x_v on the dimension dim, where x_e is the nonzero value, x_v is
 * the dense feature. dim = 0 or 1 means column-wise or row-wise broadcast
 * respectively.
 *
 * This function does not take care of autograd.
 *
 * @param sparse_mat The sparse matrix with N rows and (nnz, D) nonzero values
 * @param dense_mat Dense feature of shape (N, D)
 * @param dim The dimension to broadcast.
 *
 * @return Dense tensor of shape (nnz, D)
 */
torch::Tensor BroadcastMulNoAutoGrad(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat, torch::Tensor dense_mat,
    int64_t dim);

/**
 * @brief Perform a sparse-sparse matrix multiplication with possibly different
 * sparsities. The two sparse values must have 1-dimensional values. If the
 * first sparse matrix has shape (n, m), the second sparse matrix must have
 * shape (m, k), and the returned sparse matrix has shape (n, k).
 *
 * This function does not take care of autograd.
 *
 * @param lhs_mat The first sparse matrix of shape (n, m).
 * @param lhs_val Sparse value for the first sparse matrix.
 * @param rhs_mat The second sparse matrix of shape (m, k).
 * @param rhs_val Sparse value for the second sparse matrix.
 * @param lhs_transpose Whether the first matrix is transposed.
 * @param rhs_transpose Whether the second matrix is transposed.
 *
 * @return Sparse matrix of shape (n, k).
 */
c10::intrusive_ptr<SparseMatrix> SpSpMMNoAutoGrad(
    const c10::intrusive_ptr<SparseMatrix>& lhs_mat, torch::Tensor lhs_val,
    const c10::intrusive_ptr<SparseMatrix>& rhs_mat, torch::Tensor rhs_val,
    bool lhs_transpose, bool rhs_transpose);

}  // namespace sparse
}  // namespace dgl

#endif  // DGL_SPARSE_MATMUL_H_
