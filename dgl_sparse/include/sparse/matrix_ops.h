/**
 *  Copyright (c) 2023 by Contributors
 * @file sparse/matrix_ops.h
 * @brief DGL C++ sparse matrix operators.
 */
#ifndef SPARSE_MATRIX_OPS_H_
#define SPARSE_MATRIX_OPS_H_

#include <sparse/sparse_format.h>
#include <sparse/sparse_matrix.h>

#include <tuple>

namespace dgl {
namespace sparse {

/**
 * @brief Compute the intersection of two COO matrices. Return the intersection
 * matrix, and the indices of the intersection in the left-hand-side and
 * right-hand-side matrices.
 *
 * @param lhs The left-hand-side COO matrix.
 * @param rhs The right-hand-side COO matrix.
 *
 * @return A tuple of COO matrix, lhs indices, and rhs indices.
 */
std::tuple<std::shared_ptr<COO>, torch::Tensor, torch::Tensor> COOIntersection(
    const std::shared_ptr<COO>& lhs, const std::shared_ptr<COO>& rhs);

/**
 * @brief Relabels indices of a dimension and removes rows or columns without
 * non-zero elements in the sparse matrix.
 *
 * This function serves a dual purpose: it allows you to reorganize the
 * indices within a specific dimension (rows or columns) of the sparse matrix
 * and, if needed, place certain 'leading_indices' at the beginning of the
 * relabeled dimension.
 *
 * @param mat The sparse matrix to be relabeled.
 * @param dim The dimension to relabel. Should be 0 or 1. Use 0 for row-wise
 *        relabeling and 1 for column-wise relabeling.
 * @param leading_indices An optional tensor containing row or column ids that
 *        should be placed at the beginning of the relabeled dimension.
 *
 * @return A tuple containing the relabeled sparse matrix and the index mapping
 *         of the relabeled dimension from the new index to the original index.
 */
std::tuple<c10::intrusive_ptr<SparseMatrix>, torch::Tensor> Relabel(
    const c10::intrusive_ptr<SparseMatrix>& mat, uint64_t dim,
    torch::Tensor leading_indices);

}  // namespace sparse
}  // namespace dgl

#endif  // SPARSE_MATRIX_OPS_H_
