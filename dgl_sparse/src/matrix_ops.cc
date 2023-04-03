/**
 *  Copyright (c) 2023 by Contributors
 * @file matrix_ops.cc
 * @brief DGL C++ matrix operators.
 */
#include <sparse/matrix_ops.h>
#include <torch/script.h>

#include "./macro.h"
#include "./matrix_ops_impl.h"

namespace dgl {
namespace sparse {

/**
 * @brief Compute the intersection of two COO matrices. Return the intersection
 * COO matrix, and the indices of the intersection in the left-hand-side and
 * right-hand-side COO matrices.
 *
 * @param lhs The left-hand-side COO matrix.
 * @param rhs The right-hand-side COO matrix.
 *
 * @return A tuple of COO matrix, lhs indices, and rhs indices.
 */
std::tuple<std::shared_ptr<COO>, torch::Tensor, torch::Tensor> COOIntersection(
    const std::shared_ptr<COO>& lhs, const std::shared_ptr<COO>& rhs) {
  DGL_SPARSE_COO_SWITCH(lhs, XPU, IdType, "COOIntersection", {
    return COOIntersectionImpl<XPU, IdType>(lhs, rhs);
  });
}

}  // namespace sparse
}  // namespace dgl
