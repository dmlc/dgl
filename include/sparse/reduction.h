/**
 *  Copyright (c) 2022 by Contributors
 * @file sparse/reduction.h
 * @brief DGL C++ sparse matrix reduction operators.
 */
#ifndef SPARSE_REDUCTION_H_
#define SPARSE_REDUCTION_H_

#include <sparse/sparse_matrix.h>

#include <string>

namespace dgl {
namespace sparse {

/**
 * @brief Reduces a sparse matrix along the specified sparse dimension.
 *
 * @param A The sparse matrix.
 * @param dim The sparse dimension to reduce along.  Must be either 0 (rows) or
 * 1 (columns).
 * @param reduce The reduce operator.  Must be either "sum", "smin", "smax",
 * "mean", or "sprod".
 *
 * @return Tensor
 */
torch::Tensor Reduce(
    const c10::intrusive_ptr<SparseMatrix>& A, const std::string& reduce,
    const torch::optional<int64_t>& dim = torch::nullopt);

inline torch::Tensor ReduceSum(
    const c10::intrusive_ptr<SparseMatrix>& A,
    const torch::optional<int64_t>& dim = torch::nullopt) {
  return Reduce(A, "sum", dim);
}

inline torch::Tensor ReduceMin(
    const c10::intrusive_ptr<SparseMatrix>& A,
    const torch::optional<int64_t>& dim = torch::nullopt) {
  return Reduce(A, "smin", dim);
}

inline torch::Tensor ReduceMax(
    const c10::intrusive_ptr<SparseMatrix>& A,
    const torch::optional<int64_t>& dim = torch::nullopt) {
  return Reduce(A, "smax", dim);
}

inline torch::Tensor ReduceMean(
    const c10::intrusive_ptr<SparseMatrix>& A,
    const torch::optional<int64_t>& dim = torch::nullopt) {
  return Reduce(A, "smean", dim);
}

inline torch::Tensor ReduceProd(
    const c10::intrusive_ptr<SparseMatrix>& A,
    const torch::optional<int64_t>& dim = torch::nullopt) {
  return Reduce(A, "sprod", dim);
}

}  // namespace sparse
}  // namespace dgl

#endif  // SPARSE_REDUCTION_H_
