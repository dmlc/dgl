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

torch::Tensor ExpandIndptr(
    torch::Tensor indptr, torch::ScalarType dtype,
    torch::optional<torch::Tensor> node_ids,
    torch::optional<int64_t> output_size) {
  if (utils::is_on_gpu(indptr) &&
      (!node_ids.has_value() || utils::is_on_gpu(node_ids.value()))) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(c10::DeviceType::CUDA, "ExpandIndptr", {
      return ExpandIndptrImpl(indptr, dtype, node_ids, output_size);
    });
  }
  if (!node_ids.has_value()) {
    node_ids = torch::arange(indptr.size(0) - 1, indptr.options().dtype(dtype));
  }
  return node_ids.value().to(dtype).repeat_interleave(
      indptr.diff(), 0, output_size);
}

TORCH_LIBRARY_IMPL(graphbolt, CPU, m) {
  m.impl("expand_indptr", &ExpandIndptr);
}

TORCH_LIBRARY_IMPL(graphbolt, CUDA, m) {
  m.impl("expand_indptr", &ExpandIndptrImpl);
}

TORCH_LIBRARY_IMPL(graphbolt, Autograd, m) {
  m.impl("expand_indptr", torch::autograd::autogradNotImplementedFallback());
}

}  // namespace ops
}  // namespace graphbolt
