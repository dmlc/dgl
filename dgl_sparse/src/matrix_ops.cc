/**
 *  Copyright (c) 2023 by Contributors
 * @file matrix_ops.cc
 * @brief DGL C++ matrix operators.
 */
#include <sparse/matrix_ops.h>
#include <torch/script.h>

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
  // 1. Encode the two COO matrices into arrays of integers.
  auto lhs_arr =
      lhs->indices.index({0}) * lhs->num_cols + lhs->indices.index({1});
  auto rhs_arr =
      rhs->indices.index({0}) * rhs->num_cols + rhs->indices.index({1});
  // 2. Concatenate the two arrays.
  auto arr = torch::cat({lhs_arr, rhs_arr});
  // 3. Unique the concatenated array.
  torch::Tensor unique, inverse, counts;
  std::tie(unique, inverse, counts) =
      torch::unique_dim(arr, 0, false, true, true);
  // 4. Find the indices of the counts greater than 1 in the unique array.
  auto mask = counts > 1;
  // 5. Map the inverse array to the original array to generate indices.
  auto lhs_inverse = inverse.slice(0, 0, lhs_arr.numel());
  auto rhs_inverse = inverse.slice(0, lhs_arr.numel(), arr.numel());
  auto map_to_original = torch::empty_like(unique);
  map_to_original.index_put_(
      {lhs_inverse},
      torch::arange(lhs_inverse.numel(), map_to_original.options()));
  auto lhs_indices = map_to_original.index({mask});
  map_to_original.index_put_(
      {rhs_inverse},
      torch::arange(rhs_inverse.numel(), map_to_original.options()));
  auto rhs_indices = map_to_original.index({mask});
  // 6. Decode the indices to get the intersection COO matrix.
  auto ret_arr = unique.index({mask});
  auto ret_indices = torch::stack(
      {ret_arr.floor_divide(lhs->num_cols), ret_arr % lhs->num_cols}, 0);
  auto ret_coo = std::make_shared<COO>(
      COO{lhs->num_rows, lhs->num_cols, ret_indices, false, false});
  return {ret_coo, lhs_indices, rhs_indices};
}

}  // namespace sparse
}  // namespace dgl
