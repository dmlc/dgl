/**
 *  Copyright (c) 2023 by Contributors
 * @file index_select.cc
 * @brief Index select operators.
 */
#include "./index_select.h"

#include "./macro.h"

namespace graphbolt {
namespace ops {

torch::Tensor UVAIndexSelect(torch::Tensor input, torch::Tensor index) {
  TORCH_CHECK(
      input.device().type() == c10::DeviceType::CPU && input.is_pinned(),
      "Input tensor must be on pinned CPU memory.");
  TORCH_CHECK(
      index.device().type() == c10::DeviceType::CUDA ||
          (index.device().type() == c10::DeviceType::CPU && index.is_pinned()),
      "Index tensor must be on CUDA memory or pinned CPU memory.");
  TORCH_CHECK(
      input.is_contiguous() && index.is_contiguous(),
      "Input and index tensors must be contiguous in memory.");
  GRAPHBOLT_DTYPE_SWITCH(input.scalar_type(), DType, "input", {
    GRAPHBOLT_ID_TYPE_SWITCH(index.scalar_type(), IdType, "index", {
      return UVAIndexSelectImpl<c10::DeviceType::CUDA, DType, IdType>(
          input, index);
    });
  });
}

}  // namespace ops
}  // namespace graphbolt
