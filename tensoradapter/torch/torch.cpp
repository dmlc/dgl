#include <tensoradapter.h>
#include <torch/extension.h>
#include <ATen/DLConvertor.h>

extern "C" DLManagedTensor *empty(int64_t n) {
  auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64).layout(torch::kStrided);
  torch::Tensor tensor = torch::empty({n}, options);
  return at::toDLPack(tensor);
}
