/**
 *  Copyright (c) 2023 by Contributors
 * @file sparse/matrix_ops.h
 * @brief DGL C++ sparse matrix operators.
 */
#ifndef SPARSE_MATRIX_OPS_H_
#define SPARSE_MATRIX_OPS_H_

#include <sparse/sparse_format.h>

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

}  // namespace sparse
}  // namespace dgl

#endif  // SPARSE_MATRIX_OPS_H_
