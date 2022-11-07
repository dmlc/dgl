/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/array_nonzero.cc
 * @brief Array nonzero CPU implementation
 */
#include <dgl/array.h>

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
IdArray NonZero(IdArray array) {
  std::vector<int64_t> ret;
  const IdType* data = array.Ptr<IdType>();
  for (int64_t i = 0; i < array->shape[0]; ++i)
    if (data[i] != 0) ret.push_back(i);
  return NDArray::FromVector(ret, array->ctx);
}

template IdArray NonZero<kDGLCPU, int32_t>(IdArray);
template IdArray NonZero<kDGLCPU, int64_t>(IdArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
