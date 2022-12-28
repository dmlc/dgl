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
 * dense matrices. It calculates `sparse_mat * (mat1 @ mat2)`. The SDDMM can be
 * batched, where the batch dimension is the last dimension for all input
 * matrices.
 *
 * There are four cases for the input and output matrix shapes:
 *   (1) (n, m), (n, k), (k, m), and (n, m);
 *   (2) (n, m), (n,), and (m,), and (n, m);
 *   (3) (n, m, b), (n, k, b), (k, m, b), and (n, m, b);
 *   (4) (n, m), (n, k, b), (k, m, b), and (n, m, b);
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
