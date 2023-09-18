/**
 *  Copyright (c) 2023 by Contributors
 * @file matrix_ops_impl.h
 * @brief DGL C++ sparse matrix operator implementations.
 */
#ifndef DGL_SPARSE_MATRIX_OPS_IMPL_H_
#define DGL_SPARSE_MATRIX_OPS_IMPL_H_

#include <sparse/sparse_format.h>

#include <tuple>

namespace dgl {
namespace sparse {

template <c10::DeviceType XPU, typename IdType>
std::tuple<c10::intrusive_ptr<SparseMatrix>, torch::Tensor> CompactImpl(
    const c10::intrusive_ptr<SparseMatrix>& mat, int64_t dim,
    torch::Tensor leading_indices) {
  // Place holder only.
  return {mat, leading_indices};
}

}  // namespace sparse
}  // namespace dgl

#endif  // DGL_SPARSE_MATRIX_OPS_IMPL_H_
