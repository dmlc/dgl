/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file csr_to_coo.cc
 * @brief CSRToCOO operators.
 */
#include <graphbolt/cuda_ops.h>

#include "./macro.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

torch::Tensor CSRToCOO(
    torch::Tensor indptr, torch::ScalarType output_dtype,
    torch::optional<int64_t> num_edges,
    torch::optional<torch::Tensor> original_row_node_ids) {
  if (utils::is_accessible_from_gpu(indptr) &&
      (!original_row_node_ids.has_value() ||
       utils::is_accessible_from_gpu(original_row_node_ids.value()))) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(c10::DeviceType::CUDA, "CSRToCOO", {
      return CSRToCOOImpl(
          indptr, output_dtype, num_edges, original_row_node_ids);
    });
  }
  if (!original_row_node_ids.has_value()) {
    original_row_node_ids =
        torch::arange(indptr.size(0) - 1, indptr.options().dtype(output_dtype));
  }
  return original_row_node_ids.value().repeat_interleave(indptr.diff());
}

}  // namespace ops
}  // namespace graphbolt
