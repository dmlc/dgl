/**
 *  Copyright (c) 2019 by Contributors
 * @file array/check.h
 * @brief DGL check utilities
 */
#ifndef DGL_ARRAY_CHECK_H_
#define DGL_ARRAY_CHECK_H_

#include <dgl/array.h>
#include <dgl/runtime/ndarray.h>

#include <string>
#include <vector>

namespace dgl {
namespace aten {

// Check whether the given arguments have the same context.
inline void CheckCtx(
    const DGLContext& ctx, const std::vector<NDArray>& arrays,
    const std::vector<std::string>& names) {
  for (size_t i = 0; i < arrays.size(); ++i) {
    if (IsNullArray(arrays[i])) continue;
    CHECK_EQ(ctx, arrays[i]->ctx)
        << "Expected device context " << ctx << ". But got " << arrays[i]->ctx
        << " for " << names[i] << ".";
  }
}

// Check whether input tensors are contiguous.
inline void CheckContiguous(
    const std::vector<NDArray>& arrays, const std::vector<std::string>& names) {
  for (size_t i = 0; i < arrays.size(); ++i) {
    if (IsNullArray(arrays[i])) continue;
    CHECK(arrays[i].IsContiguous())
        << "Expect " << names[i] << " to be a contiguous tensor";
  }
}

// Check whether input tensors have valid shape.
inline void CheckShape(
    const std::vector<uint64_t>& gdim, const std::vector<int>& uev_idx,
    const std::vector<NDArray>& arrays, const std::vector<std::string>& names) {
  for (size_t i = 0; i < arrays.size(); ++i) {
    if (IsNullArray(arrays[i])) continue;
    CHECK_GE(arrays[i]->ndim, 2)
        << "Expect " << names[i] << " to have ndim >= 2, "
        << "Note that for scalar feature we expand its "
        << "dimension with an additional dimension of "
        << "length one.";
    CHECK_EQ(gdim[uev_idx[i]], arrays[i]->shape[0])
        << "Expect " << names[i] << " to have size " << gdim[uev_idx[i]]
        << " on the first dimension, "
        << "but got " << arrays[i]->shape[0];
  }
}

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CHECK_H_
