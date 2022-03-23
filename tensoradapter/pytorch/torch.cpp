/*!
 *  Copyright (c) 2020 by Contributors
 * \file torch/torch.cpp
 * \brief Implementation of PyTorch adapter library.
 */

#include <tensoradapter_exports.h>
#include <torch/torch.h>
#include <ATen/DLConvertor.h>
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
    DLContext ctx) {
  auto options = torch::TensorOptions()
    .layout(torch::kStrided)
    .device(get_device(ctx))
    .dtype(at::toScalarType(dtype));
  torch::Tensor tensor = torch::empty(shape, options);
  return at::toDLPack(tensor);
}

};

};  // namespace tensoradapter
