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

// TODO(zhenkun): support addition of matrices with different sparsity.
/**
 * @brief Adds two sparse matrices. Currently does not support two matrices with
 * different sparsity.
 *
 * @param A SparseMatrix
 * @param B SparseMatrix
 *
 * @return SparseMatrix
 */
c10::intrusive_ptr<SparseMatrix> SpSpAdd(
    const c10::intrusive_ptr<SparseMatrix>& A,
    const c10::intrusive_ptr<SparseMatrix>& B);

}  // namespace sparse
}  // namespace dgl

#endif  // SPARSE_ELEMENTWISE_OP_H_
