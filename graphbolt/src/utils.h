/**
 *  Copyright (c) 2023 by Contributors
 * @file utils.h
 * @brief Graphbolt utils.
 */

#ifndef GRAPHBOLT_UTILS_H_
#define GRAPHBOLT_UTILS_H_

#include <torch/script.h>

#include <optional>

namespace graphbolt {
namespace utils {

/**
 * @brief If this process is a worker part as part of a DataLoader, then returns
 * the assigned worker id less than the # workers.
 */
std::optional<int64_t> GetWorkerId();

/**
 * @brief If this process is a worker part as part of a DataLoader, then this
 * function is called to initialize its worked id to be less than the # workers.
 */
void SetWorkerId(int64_t worker_id_value);

/**
 * @brief Checks whether the tensor is stored on the GPU.
 */
inline bool is_on_gpu(const torch::Tensor& tensor) {
  return tensor.device().is_cuda();
}

/**
 * @brief Checks whether the tensor is stored on the GPU or the pinned memory.
 */
inline bool is_accessible_from_gpu(const torch::Tensor& tensor) {
  return is_on_gpu(tensor) || tensor.is_pinned();
}

/**
 * @brief Checks whether the tensor is stored on the pinned memory.
 */
inline bool is_pinned(const torch::Tensor& tensor) {
  // If this process is a worker, we should avoid initializing the CUDA context.
  return !GetWorkerId() && tensor.is_pinned();
}

/**
 * @brief Checks whether the tensors are all stored on the GPU or the pinned
 * memory.
 */
template <typename TensorContainer>
inline bool are_accessible_from_gpu(const TensorContainer& tensors) {
  for (auto& tensor : tensors) {
    if (!is_accessible_from_gpu(tensor)) return false;
  }
  return true;
}

/**
 * @brief Parses the source and destination node type from a given edge type
 * triple seperated with ":".
 */
inline std::pair<std::string, std::string> parse_src_dst_ntype_from_etype(
    std::string etype) {
  auto first_seperator_it = std::find(etype.begin(), etype.end(), ':');
  auto second_seperator_pos =
      std::find(first_seperator_it + 1, etype.end(), ':') - etype.begin();
  return {
      etype.substr(0, first_seperator_it - etype.begin()),
      etype.substr(second_seperator_pos + 1)};
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
