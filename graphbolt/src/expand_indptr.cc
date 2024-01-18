/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file expand_indptr.cc
 * @brief ExpandIndptr operators.
 */
#include <graphbolt/cuda_ops.h>

#include "./macro.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

torch::Tensor ExpandIndptr(
    torch::Tensor indptr, torch::ScalarType output_dtype,
    torch::optional<int64_t> num_edges,
    torch::optional<torch::Tensor> original_column_node_ids) {
  if (utils::is_on_gpu(indptr) &&
      (!original_column_node_ids.has_value() ||
       utils::is_on_gpu(original_column_node_ids.value()))) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(c10::DeviceType::CUDA, "ExpandIndptr", {
      return ExpandIndptrImpl(
          indptr, output_dtype, num_edges, original_column_node_ids);
    });
  }
  if (!original_column_node_ids.has_value()) {
    original_column_node_ids =
        torch::arange(indptr.size(0) - 1, indptr.options().dtype(output_dtype));
  }
  return original_column_node_ids.value()
      .to(output_dtype)
      .repeat_interleave(indptr.diff(), 0, num_edges);
}

}  // namespace ops
}  // namespace graphbolt
