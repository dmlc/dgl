#include <tensoradapter.h>
#include <torch/torch.h>
#include <ATen/DLConvertor.h>
#include <vector>

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

DLManagedTensor* TAclone(const DLManagedTensor* tensor) {
  return at::toDLPack(at::fromDLPack(tensor).clone());
}

DLManagedTensor* TAcopyto(const DLManagedTensor* tensor, DLContext ctx) {
  return at::toDLPack(at::fromDLPack(tensor).to(get_device(ctx)));
}

};
