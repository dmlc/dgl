/*!
 *  Copyright (c) 2020 by Contributors
 * \file torch/torch.cpp
 * \brief Implementation of PyTorch adapter library.
 */

#include <tensoradapter_exports.h>
#include <torch/torch.h>
#include <ATen/DLConvertor.h>
#ifdef DGL_USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#endif  // DGL_USE_CUDA
#include <vector>
#include <iostream>

#if DLPACK_VERSION > 040
// Compatibility across DLPack - note that this assumes that the ABI stays the same.
#define kDLGPU kDLCUDA
#define DLContext DLDevice
#endif

namespace tensoradapter {

static at::Device get_device(DLContext ctx) {
  switch (ctx.device_type) {
   case kDLCPU:
    return at::Device(torch::kCPU);
    break;
   case kDLGPU:
    return at::Device(torch::kCUDA, ctx.device_id);
    break;
   default:
    // fallback to CPU
    return at::Device(torch::kCPU);
    break;
  }
}

extern "C" {

TA_EXPORTS DLManagedTensor* TAempty(
    std::vector<int64_t> shape,
    DLDataType dtype,
    DLContext ctx,
    void* stream) {
#ifdef DGL_USE_CUDA
  // allocate tensors on DGL's current stream
  auto alloc_stream = c10::cuda::getStreamFromExternal(
    static_cast<cudaStream_t>(stream), ctx.device_id);
  auto prev_stream = c10::cuda::getCurrentCUDAStream(ctx.device_id);
  c10::cuda::setCurrentCUDAStream(alloc_stream);
#endif  // DGL_USE_CUDA
  auto options = torch::TensorOptions()
    .layout(torch::kStrided)
    .device(get_device(ctx))
    .dtype(at::toScalarType(dtype));
  torch::Tensor tensor = torch::empty(shape, options);
#ifdef DGL_USE_CUDA
  c10::cuda::setCurrentCUDAStream(prev_stream);
#endif  // DGL_USE_CUDA
  return at::toDLPack(tensor);
}

#ifdef DGL_USE_CUDA
TA_EXPORTS void* RawAlloc(size_t nbytes, void* stream) {
  return c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(
    nbytes, static_cast<cudaStream_t>(stream));
}

TA_EXPORTS void RawDelete(void* ptr) {
  c10::cuda::CUDACachingAllocator::raw_delete(ptr);
}
#endif  // DGL_USE_CUDA

};

};  // namespace tensoradapter
