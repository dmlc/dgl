/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/array_op_impl.cc
 * \brief Array operator CPU implementation
 */
#include <dgl/array.h>
#include <numeric>

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename DType, typename IdType>
IdArray FPS(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx) {
  const auto max_size = array->shape[0];
  const DType* array_data = static_cast<DType*>(array->data);
  const IdType* batch_data = static_cast<IdType*>(batch_ptr->data);
  IdArray ret = NewIdArray(npoints, ctx, sizeof(IdType) * 8);
  IdType* ret_data = static_cast<IdType*>(ret->data);
  IdType val = static_cast<IdType>(0);
  std::fill(ret_data, ret_data + npoints, val);
  return ret;
}

template IdArray FPS<kDLCPU, int32_t, int32_t>(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx);
template IdArray FPS<kDLCPU, int64_t, int32_t>(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx);
template IdArray FPS<kDLCPU, float, int32_t>(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx);
template IdArray FPS<kDLCPU, double, int32_t>(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx);
template IdArray FPS<kDLCPU, int32_t, int64_t>(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx);
template IdArray FPS<kDLCPU, int64_t, int64_t>(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx);
template IdArray FPS<kDLCPU, float, int64_t>(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx);
template IdArray FPS<kDLCPU, double, int64_t>(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
