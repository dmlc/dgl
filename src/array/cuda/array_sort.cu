/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/array_sort.cu
 * @brief Array sort GPU implementation
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
std::pair<IdArray, IdArray> Sort(IdArray array, int num_bits) {
  const auto& ctx = array->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  const int64_t nitems = array->shape[0];
  IdArray orig_idx = Range(0, nitems, 64, ctx);
  IdArray sorted_array = NewIdArray(nitems, ctx, array->dtype.bits);
  IdArray sorted_idx = NewIdArray(nitems, ctx, 64);

  const IdType* keys_in = array.Ptr<IdType>();
  const int64_t* values_in = orig_idx.Ptr<int64_t>();
  IdType* keys_out = sorted_array.Ptr<IdType>();
  int64_t* values_out = sorted_idx.Ptr<int64_t>();

  cudaStream_t stream = runtime::getCurrentCUDAStream();
  if (num_bits == 0) {
    num_bits = sizeof(IdType) * 8;
  }

  // Allocate workspace
  size_t workspace_size = 0;
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      nullptr, workspace_size, keys_in, keys_out, values_in, values_out, nitems,
      0, num_bits, stream));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);

  // Compute
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      workspace, workspace_size, keys_in, keys_out, values_in, values_out,
      nitems, 0, num_bits, stream));

  device->FreeWorkspace(ctx, workspace);

  return std::make_pair(sorted_array, sorted_idx);
}

template std::pair<IdArray, IdArray> Sort<kDGLCUDA, int32_t>(
    IdArray, int num_bits);
template std::pair<IdArray, IdArray> Sort<kDGLCUDA, int64_t>(
    IdArray, int num_bits);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
