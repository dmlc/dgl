/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file utils.h
 * @brief Graphbolt utils.
 */

#ifndef GRAPHBOLT_UTILS_H_
#define GRAPHBOLT_UTILS_H_

#include <torch/script.h>

namespace graphbolt {
namespace utils {

/**
 * @brief Checks whether the tensor is stored on the GPU or the pinned memory.
 */
inline bool is_accessible_from_gpu(torch::Tensor tensor) {
  return tensor.is_pinned() || tensor.device().type() == c10::DeviceType::CUDA;
}

}  // namespace utils
}  // namespace graphbolt

#endif  // GRAPHBOLT_UTILS_H_
