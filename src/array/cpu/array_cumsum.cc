/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/array_cumsum.cc
 * @brief Array cumsum CPU implementation
 */
#include <dgl/array.h>

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
IdArray CumSum(IdArray array, bool prepend_zero) {
  const int64_t len = array.NumElements();
  if (len == 0)
    return !prepend_zero ? array
                         : aten::Full(0, 1, array->dtype.bits, array->ctx);
  if (prepend_zero) {
    IdArray ret = aten::NewIdArray(len + 1, array->ctx, array->dtype.bits);
    const IdType* in_d = array.Ptr<IdType>();
    IdType* out_d = ret.Ptr<IdType>();
    out_d[0] = 0;
    for (int64_t i = 0; i < len; ++i) out_d[i + 1] = out_d[i] + in_d[i];
    return ret;
  } else {
    IdArray ret = aten::NewIdArray(len, array->ctx, array->dtype.bits);
    const IdType* in_d = array.Ptr<IdType>();
    IdType* out_d = ret.Ptr<IdType>();
    out_d[0] = in_d[0];
    for (int64_t i = 1; i < len; ++i) out_d[i] = out_d[i - 1] + in_d[i];
    return ret;
  }
}

template IdArray CumSum<kDGLCPU, int32_t>(IdArray, bool);
template IdArray CumSum<kDGLCPU, int64_t>(IdArray, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
