/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/array_cumsum.cu
 * @brief Array cumsum GPU implementation
 */
#include <dgl/array.h>

#include <cub/cub.cuh>

#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

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

  auto device = runtime::DeviceAPI::Get(array->ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const IdType* in_d = array.Ptr<IdType>();
  IdArray ret;
  IdType* out_d = nullptr;
  if (prepend_zero) {
    ret = aten::Full(0, len + 1, array->dtype.bits, array->ctx);
    out_d = ret.Ptr<IdType>() + 1;
  } else {
    ret = aten::NewIdArray(len, array->ctx, array->dtype.bits);
    out_d = ret.Ptr<IdType>();
  }
  // Allocate workspace
  size_t workspace_size = 0;
  CUDA_CALL(cub::DeviceScan::InclusiveSum(
      nullptr, workspace_size, in_d, out_d, len, stream));
  void* workspace = device->AllocWorkspace(array->ctx, workspace_size);

  // Compute cumsum
  CUDA_CALL(cub::DeviceScan::InclusiveSum(
      workspace, workspace_size, in_d, out_d, len, stream));

  device->FreeWorkspace(array->ctx, workspace);

  return ret;
}

template IdArray CumSum<kDGLCUDA, int32_t>(IdArray, bool);
template IdArray CumSum<kDGLCUDA, int64_t>(IdArray, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
