/**
 *  Copyright (c) 2022 by Contributors
 * @file sparse/spmm.h
 * @brief DGL C++ SpMM operator.
 */
#ifndef SPARSE_SPMM_H_
#define SPARSE_SPMM_H_

#include <sparse/sparse_matrix.h>
#include <torch/script.h>

namespace dgl {
namespace sparse {

/**
 * @brief Perform a matrix multiplication of the sparse matrix and dense
 * matrix. The SpMM can be batched, where the batch dimension is the last
 * dimension for both sparse and dense matrices.
 *
 * There are three cases for sparse, dense, and output matrix shapes:
 *   (1) (n, m), (m, k), and (n, k);
 *   (2) (n, m), (m,), and (n,);
 *   (3) (n, m, b), (m, k, b), and (n, k, b).
 *
 * This function supports autograd for both the sparse and dense matrix but does
 * not support higher order gradient.
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
