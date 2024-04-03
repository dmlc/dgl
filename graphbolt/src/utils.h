/**
 *  Copyright (c) 2023 by Contributors
 * @file utils.h
 * @brief Graphbolt utils.
 */

#ifndef GRAPHBOLT_UTILS_H_
#define GRAPHBOLT_UTILS_H_

#include <torch/script.h>

namespace graphbolt {
namespace utils {

/**
 * @brief Checks whether the tensor is stored on the GPU.
 */
inline bool is_on_gpu(torch::Tensor tensor) {
  return tensor.device().is_cuda();
}

/**
 * @brief Checks whether the tensor is stored on the GPU or the pinned memory.
 */
inline bool is_accessible_from_gpu(torch::Tensor tensor) {
  return is_on_gpu(tensor) || tensor.is_pinned();
}

/**
 * @brief Parses the destination node type from a given edge type triple
 * seperated with ":".
 */
inline std::string parse_dst_ntype_from_etype(std::string etype) {
  auto first_seperator_it = std::find(etype.begin(), etype.end(), ':');
  auto second_seperator_pos =
      std::find(first_seperator_it + 1, etype.end(), ':') - etype.begin();
  return etype.substr(second_seperator_pos + 1);
}

/**
 * @brief Retrieves the value of the tensor at the given index.
 *
 * @note If the tensor is not contiguous, it will be copied to a contiguous
 * tensor.
 *
 * @tparam T The type of the tensor.
 * @param tensor The tensor.
 * @param index The index.
 *
 * @return T The value of the tensor at the given index.
 */
template <typename T>
T GetValueByIndex(const torch::Tensor& tensor, int64_t index) {
  TORCH_CHECK(
      index >= 0 && index < tensor.numel(),
      "The index should be within the range of the tensor, but got index ",
      index, " and tensor size ", tensor.numel());
  auto contiguous_tensor = tensor.contiguous();
  auto data_ptr = contiguous_tensor.data_ptr<T>();
  return data_ptr[index];
}

}  // namespace utils
}  // namespace graphbolt

#endif  // GRAPHBOLT_UTILS_H_
