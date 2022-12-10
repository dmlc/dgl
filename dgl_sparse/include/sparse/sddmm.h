/**
 *  Copyright (c) 2022 by Contributors
 * @file sparse/sddmm.h
 * @brief DGL C++ SDDMM operator.
 */
#ifndef SPARSE_SDDMM_H_
#define SPARSE_SDDMM_H_

// clang-format off
#include <sparse/dgl_headers.h>
// clang-format on

#include <sparse/sparse_matrix.h>
#include <torch/script.h>

namespace dgl {
namespace sparse {

/**
 * @brief Perform a sampled matrix multiplication of a sparse matrix and two
 * dense matrices. For efficiency, `mat2_tr` is the transposition of the matrix to
 * be multiplied. If the sparse matrix has shape (n, m), `mat1` and `mat2_tr`
 * must have shapes of `(n, k)` and `(m, k)` or
 * `(n,)` and `(m,)` respectively. And the returned tensor has shape
 * `(sparse_matrix->nnz())`.
 *
 * This function does not take care of autograd.
 *
 * @param sparse_mat The sparse matrix.
 * @param mat1 The first dense matrix.
 * @param mat2_tr Transposition of the second matrix.
 *
 * @return Dense tensor.
 */
torch::Tensor SDDMMImpl(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat, torch::Tensor mat1,
    torch::Tensor mat2_tr);

}  // namespace sparse
}  // namespace dgl

#endif  // SPARSE_SDDMM_H_
