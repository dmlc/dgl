/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/array_index_select_uvm.cu
 * \brief Array index select GPU implementation
 */
#include <dgl/array.h>
#include "../../../runtime/cuda/cuda_common.h"
#include "../array_index_select.cuh"
#include "./array_index_select_uvm.cuh"
#include "../utils.h"

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template<typename DType, typename IdType>
NDArray IndexSelectCPUFromGPU(NDArray array, IdArray index) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const DType* array_data = static_cast<DType*>(array->data);
  const IdType* idx_data = static_cast<IdType*>(index->data);
  const int64_t arr_len = array->shape[0];
  const int64_t len = index->shape[0];
  int64_t num_feat = 1;
  std::vector<int64_t> shape{len};

  CHECK(array.IsPinned());
  CHECK_EQ(index->ctx.device_type, kDLGPU);

  for (int d = 1; d < array->ndim; ++d) {
    num_feat *= array->shape[d];
    shape.emplace_back(array->shape[d]);
  }

  NDArray ret = NDArray::Empty(shape, array->dtype, index->ctx);
  if (len == 0)
    return ret;
  DType* ret_data = static_cast<DType*>(ret->data);

  if (num_feat == 1) {
      const int nt = cuda::FindNumThreads(len);
      const int nb = (len + nt - 1) / nt;
      CUDA_KERNEL_CALL(IndexSelectSingleKernel, nb, nt, 0,
          thr_entry->stream, array_data, idx_data, len, arr_len, ret_data);
  } else {
      dim3 block(256, 1);
      while (static_cast<int64_t>(block.x) >= 2*num_feat) {
          block.x /= 2;
          block.y *= 2;
      }
      const dim3 grid((len+block.y-1)/block.y);
      if (num_feat * sizeof(DType) < 2 * CACHE_LINE_SIZE) {
        CUDA_KERNEL_CALL(IndexSelectMultiKernel, grid, block, 0,
            thr_entry->stream, array_data, num_feat, idx_data,
            len, arr_len, ret_data);
      } else {
        CUDA_KERNEL_CALL(IndexSelectMultiKernelAligned, grid, block, 0,
            thr_entry->stream, array_data, num_feat, idx_data,
            len, arr_len, ret_data);
      }
  }
  return ret;
}

template NDArray IndexSelectCPUFromGPU<int8_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelectCPUFromGPU<int8_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelectCPUFromGPU<int16_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelectCPUFromGPU<int16_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelectCPUFromGPU<int32_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelectCPUFromGPU<int32_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelectCPUFromGPU<int64_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelectCPUFromGPU<int64_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelectCPUFromGPU<float, int32_t>(NDArray, IdArray);
template NDArray IndexSelectCPUFromGPU<float, int64_t>(NDArray, IdArray);
template NDArray IndexSelectCPUFromGPU<double, int32_t>(NDArray, IdArray);
template NDArray IndexSelectCPUFromGPU<double, int64_t>(NDArray, IdArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
