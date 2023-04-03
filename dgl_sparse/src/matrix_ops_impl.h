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
std::tuple<std::shared_ptr<COO>, torch::Tensor, torch::Tensor>
COOIntersectionImpl(
    const std::shared_ptr<COO>& lhs, const std::shared_ptr<COO>& rhs);

}
}  // namespace dgl

#endif  // DGL_SPARSE_MATRIX_OPS_IMPL_H_
