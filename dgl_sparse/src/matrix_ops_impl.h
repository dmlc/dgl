/**
 *  Copyright (c) 2023 by Contributors
 * @file matrix_ops_impl.h
 * @brief DGL C++ sparse matrix operator implementations.
 */
#ifndef SPARSE_MATRIX_OPS_IMPL_H_
#define SPARSE_MATRIX_OPS_IMPL_H_

#include <sparse/sparse_format.h>

#include <tuple>

namespace dgl {
namespace sparse {

/**
 * @brief Compact sparse matrix by removing rows or columns without non-zero
 * elements in the sparse matrix and relabeling indices of the dimension.
 *
 * This function serves a dual purpose: it allows you to reorganize the
 * indices within a specific dimension (rows or columns) of the sparse matrix
 * and, if needed, place certain 'leading_indices' at the beginning of the
 * compact dimension.
 *
 * @param mat The sparse matrix to be compacted.
 * @param dim The dimension to compact. Should be 0 or 1. Use 0 for row-wise
 *        compaction and 1 for column-wise compaction.
 * @param leading_indices An optional tensor containing row or column ids that
 *        should be placed at the beginning of the compact dimension.
 *
 * @return A tuple containing the compacted sparse matrix and the index mapping
 *         of the compact dimension from the new index to the original index.
 */
template <c10::DeviceType XPU, typename IdType>
std::tuple<c10::intrusive_ptr<SparseMatrix>, torch::Tensor> CompactImpl(
    const c10::intrusive_ptr<SparseMatrix>& mat, int64_t dim,
    torch::Tensor leading_indices) {
  // Place holder only.
  return {mat, leading_indices};
}

}  // namespace sparse
}  // namespace dgl

#endif  // SPARSE_MATRIX_OPS_IMPL_H_
