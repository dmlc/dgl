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
    return torch::repeat_interleave(indptr.diff(), output_size).to(dtype);
  }
  return node_ids.value().to(dtype).repeat_interleave(
      indptr.diff(), 0, output_size);
}

torch::Tensor IndptrEdgeIds(
    torch::Tensor indptr, torch::ScalarType dtype,
    torch::optional<torch::Tensor> offset,
    torch::optional<int64_t> output_size) {
  if (utils::is_on_gpu(indptr) &&
      (!offset.has_value() || utils::is_on_gpu(offset.value()))) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "IndptrEdgeIds",
        { return IndptrEdgeIdsImpl(indptr, dtype, offset, output_size); });
  }
  TORCH_CHECK(false, "CPU implementation of IndptrEdgeIds is not available.");
}

TORCH_LIBRARY_IMPL(graphbolt, CPU, m) {
  m.impl("expand_indptr", &ExpandIndptr);
}

#ifdef GRAPHBOLT_USE_CUDA
TORCH_LIBRARY_IMPL(graphbolt, CUDA, m) {
  m.impl("expand_indptr", &ExpandIndptrImpl);
}
#endif

TORCH_LIBRARY_IMPL(graphbolt, Autograd, m) {
  m.impl("expand_indptr", torch::autograd::autogradNotImplementedFallback());
}

TORCH_LIBRARY_IMPL(graphbolt, CPU, m) {
  m.impl("indptr_edge_ids", &IndptrEdgeIds);
}

#ifdef GRAPHBOLT_USE_CUDA
TORCH_LIBRARY_IMPL(graphbolt, CUDA, m) {
  m.impl("indptr_edge_ids", &IndptrEdgeIdsImpl);
}
#endif

TORCH_LIBRARY_IMPL(graphbolt, Autograd, m) {
  m.impl("indptr_edge_ids", torch::autograd::autogradNotImplementedFallback());
}

}  // namespace ops
}  // namespace graphbolt
