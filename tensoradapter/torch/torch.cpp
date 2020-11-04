#include <tensoradapter.h>
#include <torch/extension.h>
#include <ATen/DLConvertor.h>
#include <vector>

extern "C" DLManagedTensor *empty(const std::vector<int64_t> &shape) {
  auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64).layout(torch::kStrided);
  torch::Tensor tensor = torch::empty(shape, options);
  return at::toDLPack(tensor);
}
