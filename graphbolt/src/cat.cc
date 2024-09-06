/**
 *   Copyright (c) 2024, mfbalin (Muhammed Fatih Balin)
 *   All rights reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 * @file cat.cc
 * @brief Concatenation operation.
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
