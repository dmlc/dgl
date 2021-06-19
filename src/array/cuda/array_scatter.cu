/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cuda/array_scatter.cu
 * \brief Array scatter GPU implementation
 */
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <typename DType, typename IdType>
__global__ void _ScatterKernel(const IdType* index, const DType* value,
                               int64_t length, DType* out) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    out[index[tx]] = value[tx];
    tx += stride_x;
  }
}

template <DLDeviceType XPU, typename DType, typename IdType>
void Scatter_(IdArray index, NDArray value, NDArray out) {
  const int64_t len = index->shape[0];
  const IdType* idx = index.Ptr<IdType>();
  const DType* val = value.Ptr<DType>();
  DType* outd = out.Ptr<DType>();

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const int nt = cuda::FindNumThreads(len);
  const int nb = (len + nt - 1) / nt;
  CUDA_KERNEL_CALL(_ScatterKernel, nb, nt, 0, thr_entry->stream,
      idx, val, len, outd);
}

template void Scatter_<kDLGPU, int32_t, int32_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDLGPU, int64_t, int32_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDLGPU, float, int32_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDLGPU, double, int32_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDLGPU, int32_t, int64_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDLGPU, int64_t, int64_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDLGPU, float, int64_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDLGPU, double, int64_t>(IdArray, NDArray, NDArray);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
