/**
 *  Copyright (c) 2022 by Contributors
 * @file sparse/spspmm.h
 * @brief DGL C++ SpSpMM operator.
 */
#ifndef SPARSE_SPSPMM_H_
#define SPARSE_SPSPMM_H_

#include <sparse/sparse_matrix.h>
#include <torch/script.h>

namespace dgl {
namespace sparse {

/**
 * @brief Perform a sparse-sparse matrix multiplication on matrices with
 * possibly different sparsities. The two sparse matrices must have
 * 1-D values. If the first sparse matrix has shape (n, m), the second
 * sparse matrix must have shape (m, k), and the returned sparse matrix has
 * shape (n, k).
 *
 * This function supports autograd for both sparse matrices but does
 * not support higher order gradient.
 *
 * @param lhs_mat The first sparse matrix of shape (n, m).
 * @param rhs_mat The second sparse matrix of shape (m, k).
 *
 * @return Sparse matrix of shape (n, k).
 */
c10::intrusive_ptr<SparseMatrix> SpSpMM(
    const c10::intrusive_ptr<SparseMatrix>& lhs_mat,
    const c10::intrusive_ptr<SparseMatrix>& rhs_mat);

}  // namespace sparse
}  // namespace dgl

#endif  // SPARSE_SPSPMM_H_
