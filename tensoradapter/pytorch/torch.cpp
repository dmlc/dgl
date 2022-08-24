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
#endif  // DGL_USE_CUDA
#include <vector>
#include <iostream>

#if DLPACK_VERSION > 040
// Compatibility across DLPack - note that this assumes that the ABI stays the same.
#define kDLCUDA kDLCUDA
#endif

namespace tensoradapter {

static at::Device get_device(DLDevice device) {
  switch (device.device_type) {
   case kDLCPU:
    return at::Device(torch::kCPU);
    break;
   case kDLCUDA:
    return at::Device(torch::kCUDA, device.device_id);
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
    DLDevice device) {
  auto options = torch::TensorOptions()
    .layout(torch::kStrided)
    .device(get_device(device))
    .dtype(at::toScalarType(dtype));
  torch::Tensor tensor = torch::empty(shape, options);
  return at::toDLPack(tensor);
}

#ifdef DGL_USE_CUDA
TA_EXPORTS void* RawAlloc(size_t nbytes) {
  return c10::cuda::CUDACachingAllocator::raw_alloc(nbytes);
}

TA_EXPORTS void RawDelete(void* ptr) {
  c10::cuda::CUDACachingAllocator::raw_delete(ptr);
}
#endif  // DGL_USE_CUDA

};

};  // namespace tensoradapter
