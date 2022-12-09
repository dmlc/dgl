/**
 *  Copyright (c) 2019 by Contributors
 * @file array/cpu/array_index_select.cu
 * @brief Array index select GPU implementation
 */
#include <dgl/array.h>

#include "../../runtime/cuda/cuda_common.h"
#include "./array_index_select.cuh"
#include "./utils.h"

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename DType, typename IdType>
NDArray IndexSelect(NDArray array, IdArray index) {
  const int64_t arr_len = array->shape[0];
  const int64_t len = index->shape[0];
  int64_t num_feat = 1;
  std::vector<int64_t> shape{len};
  for (int d = 1; d < array->ndim; ++d) {
    num_feat *= array->shape[d];
    shape.emplace_back(array->shape[d]);
  }

  // use index->ctx for pinned array
  NDArray ret = NDArray::Empty(shape, array->dtype, index->ctx);
  if (len == 0 || arr_len * num_feat == 0) return ret;
  DType* ret_data = static_cast<DType*>(ret->data);

  const DType* array_data = static_cast<DType*>(cuda::GetDevicePointer(array));
  const IdType* idx_data = static_cast<IdType*>(index->data);

  cudaStream_t stream = runtime::getCurrentCUDAStream();
  if (num_feat == 1) {
    const int nt = cuda::FindNumThreads(len);
    const int nb = (len + nt - 1) / nt;
    CUDA_KERNEL_CALL(
        IndexSelectSingleKernel, nb, nt, 0, stream, array_data, idx_data, len,
        arr_len, ret_data);
  } else {
    dim3 block(256, 1);
    while (static_cast<int64_t>(block.x) >= 2 * num_feat) {
      block.x /= 2;
      block.y *= 2;
    }
    const dim3 grid((len + block.y - 1) / block.y);
    CUDA_KERNEL_CALL(
        IndexSelectMultiKernel, grid, block, 0, stream, array_data, num_feat,
        idx_data, len, arr_len, ret_data);
  }
  return ret;
}

template NDArray IndexSelect<kDGLCUDA, int32_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLCUDA, int32_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLCUDA, int64_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLCUDA, int64_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLCUDA, __half, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLCUDA, __half, int64_t>(NDArray, IdArray);
#if BF16_ENABLED
template NDArray IndexSelect<kDGLCUDA, __nv_bfloat16, int32_t>(
    NDArray, IdArray);
template NDArray IndexSelect<kDGLCUDA, __nv_bfloat16, int64_t>(
    NDArray, IdArray);
#endif  // BF16_ENABLED
template NDArray IndexSelect<kDGLCUDA, float, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLCUDA, float, int64_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLCUDA, double, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLCUDA, double, int64_t>(NDArray, IdArray);

template <DGLDeviceType XPU, typename DType>
DType IndexSelect(NDArray array, int64_t index) {
  auto device = runtime::DeviceAPI::Get(array->ctx);
  DType ret = static_cast<DType>(0.0f);
  device->CopyDataFromTo(
      static_cast<DType*>(array->data) + index, 0, &ret, 0, sizeof(DType),
      array->ctx, DGLContext{kDGLCPU, 0}, array->dtype);
  return ret;
}

template int32_t IndexSelect<kDGLCUDA, int32_t>(NDArray array, int64_t index);
template int64_t IndexSelect<kDGLCUDA, int64_t>(NDArray array, int64_t index);
template uint32_t IndexSelect<kDGLCUDA, uint32_t>(NDArray array, int64_t index);
template uint64_t IndexSelect<kDGLCUDA, uint64_t>(NDArray array, int64_t index);
template __half IndexSelect<kDGLCUDA, __half>(NDArray array, int64_t index);
#if BF16_ENABLED
template __nv_bfloat16 IndexSelect<kDGLCUDA, __nv_bfloat16>(
    NDArray array, int64_t index);
#endif  // BF16_ENABLED
template float IndexSelect<kDGLCUDA, float>(NDArray array, int64_t index);
template double IndexSelect<kDGLCUDA, double>(NDArray array, int64_t index);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
