/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file expand_indptr.cc
 * @brief ExpandIndptr operators.
 */
#include <graphbolt/cuda_ops.h>
#include <torch/autograd.h>

#include "./macro.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

torch::Tensor Cat(const std::vector<torch::Tensor>& tensors) {
  bool all_on_gpu = true;
  for (const auto& tensor : tensors) {
    all_on_gpu = all_on_gpu && utils::is_on_gpu(tensor);
    if (!all_on_gpu) break;
  }
  if (all_on_gpu) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "unique_and_compact",
        { return ops::CatImpl(tensors); });
  }
  return torch::cat(tensors, 0);
}

TORCH_LIBRARY_IMPL(graphbolt, CPU, m) { m.impl("cat", &Cat); }

#ifdef GRAPHBOLT_USE_CUDA
TORCH_LIBRARY_IMPL(graphbolt, CUDA, m) { m.impl("cat", &CatImpl); }
#endif

TORCH_LIBRARY_IMPL(graphbolt, Autograd, m) {
  m.impl("cat", torch::autograd::autogradNotImplementedFallback());
}

}  // namespace ops
}  // namespace graphbolt
