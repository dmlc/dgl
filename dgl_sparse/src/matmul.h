/**
 *  Copyright (c) 2022 by Contributors
 * @file matmul.h
 * @brief DGL sparse matrix multiplication functions.
 */
#ifndef DGL_SPARSE_MATMUL_H_
#define DGL_SPARSE_MATMUL_H_

#include <sparse/sparse_matrix.h>
#include <torch/script.h>

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

}  // namespace sparse
}  // namespace dgl

#endif  // DGL_SPARSE_MATMUL_H_
