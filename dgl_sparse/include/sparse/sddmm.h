/**
 *  Copyright (c) 2022 by Contributors
 * @file sparse/sddmm.h
 * @brief DGL C++ SDDMM operator.
 */
#ifndef SPARSE_SDDMM_H_
#define SPARSE_SDDMM_H_

#include <sparse/sparse_matrix.h>
#include <torch/script.h>

namespace dgl {
namespace sparse {

/**
 * @brief Perform a sampled matrix multiplication of a sparse matrix and two
 * dense matrices. It calculates `(mat1 @ mat2) * sparse_mat`. If the sparse
 * matrix has shape (n, m), `mat1` and `mat2` must have shapes of `(n, k)` and
 * `(k, m)` or `(n,)` and `(m,)` respectively. And the returned tensor has shape
 * `(sparse_matrix->nnz(),)`.
 *
 * This function supports autograd for `mat1` and `mat2` but does not support
 * high order gradient.
 *
 *
 * @param sparse_mat The sparse matrix.
 * @param mat1 The first dense matrix.
 * @param mat2 The second dense matrix.
 *
 * @return SparseMatrix
 */
c10::intrusive_ptr<SparseMatrix> SDDMM(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat, torch::Tensor mat1,
    torch::Tensor mat2);

}  // namespace sparse
}  // namespace dgl

#endif  // SPARSE_SDDMM_H_
