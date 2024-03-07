/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/array_nonzero.cc
 * @brief Array nonzero CPU implementation
 */

#include <dgl/array.h>

#include <cub/cub.cuh>

#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <typename IdType>
struct IsNonZeroIndex {
  explicit IsNonZeroIndex(const IdType* array) : array_(array) {}

  __device__ bool operator()(const int64_t index) { return array_[index] != 0; }

  const IdType* array_;
};

template <DGLDeviceType XPU, typename IdType>
IdArray NonZero(IdArray array) {
  const auto& ctx = array->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);

  const int64_t len = array->shape[0];
  IdArray ret = NewIdArray(len, ctx, 64);

  cudaStream_t stream = runtime::getCurrentCUDAStream();

  const IdType* const in_data = static_cast<const IdType*>(array->data);
  int64_t* const out_data = static_cast<int64_t*>(ret->data);

  IsNonZeroIndex<IdType> comp(in_data);
  cub::CountingInputIterator<int64_t> counter(0);

  // room for cub to output on GPU
  int64_t* d_num_nonzeros =
      static_cast<int64_t*>(device->AllocWorkspace(ctx, sizeof(int64_t)));

  size_t temp_size = 0;
  CUDA_CALL(cub::DeviceSelect::If(
      nullptr, temp_size, counter, out_data, d_num_nonzeros, len, comp,
      stream));
  void* temp = device->AllocWorkspace(ctx, temp_size);
  CUDA_CALL(cub::DeviceSelect::If(
      temp, temp_size, counter, out_data, d_num_nonzeros, len, comp, stream));
  device->FreeWorkspace(ctx, temp);

  // copy number of selected elements from GPU to CPU
  int64_t num_nonzeros = cuda::GetCUDAScalar(device, ctx, d_num_nonzeros);
  device->FreeWorkspace(ctx, d_num_nonzeros);
  device->StreamSync(ctx, stream);

  // truncate array to size
  return ret.CreateView({num_nonzeros}, ret->dtype, 0);
}

template IdArray NonZero<kDGLCUDA, int32_t>(IdArray);
template IdArray NonZero<kDGLCUDA, int64_t>(IdArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
