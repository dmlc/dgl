/**
 *  Copyright (c) 2023 by Contributors
 * @file utils.h
 * @brief Graphbolt utils.
 */

#ifndef GRAPHBOLT_UTILS_H_
#define GRAPHBOLT_UTILS_H_

#include <torch/script.h>

namespace graphbolt {
namespace cuda {

/**
 * @brief Checks whether the tensor is stored on the GPU or the pinned memory.
 */
inline bool is_accessible(torch::Tensor tensor) {
  return tensor.is_pinned() || tensor.device().type() == c10::DeviceType::CUDA;
}

}  // namespace cuda

}  // namespace graphbolt

#endif  // GRAPHBOLT_UTILS_H_
