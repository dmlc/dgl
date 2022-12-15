/**
 *  Copyright (c) 2022 by Contributors
 * @file sparse/spmm.h
 * @brief DGL C++ SpMM operator.
 */
#ifndef SPARSE_SPMM_H_
#define SPARSE_SPMM_H_

// clang-format off
#include <sparse/dgl_headers.h>
// clang-format on

#include <sparse/sparse_matrix.h>
#include <torch/script.h>

namespace dgl {
namespace sparse {

/**
 * @brief Perform a matrix multiplication of the sparse matrix and dense
 * matrix. It uses the sparse formats of `sparse_mat` and non-zero values of
 * `sparse_val` for SpMM. The `sparse_val` must be 1-dimensional. If the sparse
 * matrix has shape (n, m), the dense matrix must have shape (m, k) or (m,). and
 * the returned dense matrix has shape (n, k) or (n,
 * ).
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
torch::Tensor SpMMImpl(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat,
    torch::Tensor sparse_val, torch::Tensor dense_mat, bool transpose_sparse);

/**
 * @brief Perform a matrix multiplication of the sparse matrix and dense
 * matrix. The sparse matrix must have 1-dimensional values. If the sparse
 * matrix has shape (n, m), the dense matrix must have shape (m, k) or (m,), and
 * the returned dense matrix has shape (n, k) or (n,).
 *
 * This function supports autograd for both the sparse and dense matrix.
 *
 * @param sparse_mat The sparse matrix.
 * @param dense_mat The dense matrix.
 *
 * @return Dense matrix.
 */
torch::Tensor SpMM(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat,
    torch::Tensor dense_mat);

}  // namespace sparse
}  // namespace dgl

#endif  // SPARSE_SPMM_H_
