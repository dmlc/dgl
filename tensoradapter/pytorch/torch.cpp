/*!
 *  Copyright (c) 2020 by Contributors
 * \file torch/torch.cpp
 * \brief Implementation of PyTorch adapter library.
 */

#include <tensoradapter.h>
#include <torch/torch.h>
#include <ATen/DLConvertor.h>
#include <vector>
#include <iostream>

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

DLManagedTensor* TAempty(
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
