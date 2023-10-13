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

/** @brief Return the reverted mapping of a permutation. */
static torch::Tensor RevertPermutation(const torch::Tensor& perm) {
  auto rev_tensor = torch::empty_like(perm);
  rev_tensor.index_put_(
      {perm}, torch::arange(0, perm.numel(), rev_tensor.options()));
  return rev_tensor;
}

/**
 * @brief Compute the compact indices of row indices and leading indices. Return
 * the compacted indices and the original row indices of compacted indices.
 *
 * @param row The row indices.
 * @param leading_indices The leading indices.
 *
 * @return A tuple of compact indices, original indices.
 */
static std::tuple<torch::Tensor, torch::Tensor> CompactIndices(
    const torch::Tensor& row,
    const torch::optional<torch::Tensor>& leading_indices) {
  torch::Tensor sorted, sort_indices, uniqued, unique_reverse_indices, counts;
  // 1. Sort leading indices and row indices in ascending order.
  int64_t n_leading_indices = 0;
  if (leading_indices.has_value()) {
    n_leading_indices = leading_indices.value().numel();
    std::tie(sorted, sort_indices) =
        torch::cat({leading_indices.value(), row}).sort();
  } else {
    std::tie(sorted, sort_indices) = row.sort();
  }
  // 2. Reverse sort indices.
  auto sort_rev_indices = RevertPermutation(sort_indices);
  // 3. Unique the sorted array.
  std::tie(uniqued, unique_reverse_indices, counts) =
      torch::unique_consecutive(sorted, true);
  auto reverse_indices = unique_reverse_indices.index({sort_rev_indices});
  auto n_uniqued = uniqued.numel();

  // 4. Relabel the indices and map the inverse array to the original array.
  auto split_indices = torch::full({n_uniqued}, -1, reverse_indices.options());

  split_indices.index_put_(
      {reverse_indices.slice(0, 0, n_leading_indices)},
      torch::arange(0, n_leading_indices, split_indices.options()));

  split_indices.index_put_(
      {(split_indices == -1).nonzero().view(-1)},
      torch::arange(n_leading_indices, n_uniqued, split_indices.options()));
  // 5. Decode the indices to get the compact indices.
  auto new_row = split_indices.index({reverse_indices.slice(
      0, n_leading_indices, n_leading_indices + row.numel())});
  return {new_row, uniqued.index({RevertPermutation(split_indices)})};
}

static std::tuple<c10::intrusive_ptr<SparseMatrix>, torch::Tensor> CompactCOO(
    const c10::intrusive_ptr<SparseMatrix>& mat, int64_t dim,
    const torch::optional<torch::Tensor>& leading_indices) {
  torch::Tensor row, col;
  auto coo = mat->COOTensors();
  if (dim == 0)
    std::tie(row, col) = coo;
  else
    std::tie(col, row) = coo;

  torch::Tensor new_row, uniqued;
  std::tie(new_row, uniqued) = CompactIndices(row, leading_indices);

  if (dim == 0) {
    auto ret = SparseMatrix::FromCOO(
        torch::stack({new_row, col}, 0), mat->value(),
        std::vector<int64_t>{uniqued.numel(), mat->shape()[1]});
    return {ret, uniqued};
  } else {
    auto ret = SparseMatrix::FromCOO(
        torch::stack({col, new_row}, 0), mat->value(),
        std::vector<int64_t>{mat->shape()[0], uniqued.numel()});
    return {ret, uniqued};
  }
}

static std::tuple<c10::intrusive_ptr<SparseMatrix>, torch::Tensor> CompactCSR(
    const c10::intrusive_ptr<SparseMatrix>& mat, int64_t dim,
    const torch::optional<torch::Tensor>& leading_indices) {
  std::shared_ptr<CSR> csr;
  if (dim == 0)
    csr = mat->CSCPtr();
  else
    csr = mat->CSRPtr();

  torch::Tensor new_indices, uniqued;
  std::tie(new_indices, uniqued) =
      CompactIndices(csr->indices, leading_indices);

  auto ret_value = mat->value();
  if (csr->value_indices.has_value())
    ret_value = mat->value().index_select(0, csr->value_indices.value());
  if (dim == 0) {
    auto ret = SparseMatrix::FromCSC(
        csr->indptr, new_indices, ret_value,
        std::vector<int64_t>{uniqued.numel(), mat->shape()[1]});
    return {ret, uniqued};
  } else {
    auto ret = SparseMatrix::FromCSR(
        csr->indptr, new_indices, ret_value,
        std::vector<int64_t>{mat->shape()[0], uniqued.numel()});
    return {ret, uniqued};
  }
}

std::tuple<c10::intrusive_ptr<SparseMatrix>, torch::Tensor> Compact(
    const c10::intrusive_ptr<SparseMatrix>& mat, int64_t dim,
    const torch::optional<torch::Tensor>& leading_indices) {
  if (mat->HasCOO()) {
    return CompactCOO(mat, dim, leading_indices);
  }
  return CompactCSR(mat, dim, leading_indices);
}

}  // namespace sparse
}  // namespace dgl
