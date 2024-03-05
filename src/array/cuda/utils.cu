/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/utils.cu
 * @brief Utilities for CUDA kernels.
 */

#include <cub/cub.cuh>

#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {
namespace cuda {

bool AllTrue(int8_t* flags, int64_t length, const DGLContext& ctx) {
  auto device = runtime::DeviceAPI::Get(ctx);
  int8_t* rst = static_cast<int8_t*>(device->AllocWorkspace(ctx, 1));
  // Call CUB's reduction
  size_t workspace_size = 0;
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  CUDA_CALL(cub::DeviceReduce::Min(
      nullptr, workspace_size, flags, rst, length, stream));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);
  CUDA_CALL(cub::DeviceReduce::Min(
      workspace, workspace_size, flags, rst, length, stream));
  int8_t cpu_rst = GetCUDAScalar(device, ctx, rst);
  device->FreeWorkspace(ctx, workspace);
  device->FreeWorkspace(ctx, rst);
  return cpu_rst == 1;
}

}  // namespace cuda
}  // namespace dgl
