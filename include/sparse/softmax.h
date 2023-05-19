/**
 *  Copyright (c) 2022 by Contributors
 * @file sparse/softmax.h
 * @brief DGL C++ Softmax operator
 */
#ifndef SPARSE_SOFTMAX_H_
#define SPARSE_SOFTMAX_H_

#include <sparse/sparse_matrix.h>

namespace dgl {
namespace sparse {

/**
 * @brief Apply softmax to the non-zero entries of the sparse matrix on the
 * dimension dim. dim = 0 or 1 indicates column-wise or row-wise softmax
 * respectively.
 *
 * This function supports autograd for the sparse matrix, but it does not
 * support higher order gradient.
 *
 * @param sparse_mat The sparse matrix
 * @param dim The dimension to apply softmax
 *
 * @return Sparse matrix
 */
c10::intrusive_ptr<SparseMatrix> Softmax(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat, int64_t dim);

}  // namespace sparse
}  // namespace dgl

#endif  // SPARSE_SOFTMAX_H_
