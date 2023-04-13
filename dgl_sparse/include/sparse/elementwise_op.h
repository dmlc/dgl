/**
 *  Copyright (c) 2022 by Contributors
 * @file sparse/elementwise_op.h
 * @brief DGL C++ sparse elementwise operators.
 */
#ifndef SPARSE_ELEMENTWISE_OP_H_
#define SPARSE_ELEMENTWISE_OP_H_

#include <sparse/sparse_matrix.h>

namespace dgl {
namespace sparse {

/**
 * @brief Adds two sparse matrices possibly with different sparsities.
 *
 * @param A SparseMatrix
 * @param B SparseMatrix
 *
 * @return SparseMatrix
 */
c10::intrusive_ptr<SparseMatrix> SpSpAdd(
    const c10::intrusive_ptr<SparseMatrix>& A,
    const c10::intrusive_ptr<SparseMatrix>& B);

/**
 * @brief Multiplies two sparse matrices possibly with different sparsities.
 *
 * @param A SparseMatrix
 * @param B SparseMatrix
 *
 * @return SparseMatrix
 */
c10::intrusive_ptr<SparseMatrix> SpSpMul(
    const c10::intrusive_ptr<SparseMatrix>& lhs_mat,
    const c10::intrusive_ptr<SparseMatrix>& rhs_mat);

}  // namespace sparse
}  // namespace dgl

#endif  // SPARSE_ELEMENTWISE_OP_H_
