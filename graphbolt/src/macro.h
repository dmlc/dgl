/**
 *  Copyright (c) 2023 by Contributors
 * @file macro.h
 * @brief Graphbolt macros.
 */

#ifndef GRAPHBOLT_MACRO_H_
#define GRAPHBOLT_MACRO_H_

#include <torch/script.h>

namespace graphbolt {

// Dispatch operator implementation function to CUDA device only.
#ifdef GRAPHBOLT_USE_CUDA
#define GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(device_type, name, ...) \
  if (device_type == c10::DeviceType::CUDA) {                       \
    [[maybe_unused]] auto XPU = c10::DeviceType::CUDA;              \
    __VA_ARGS__                                                     \
  } else {                                                          \
    TORCH_CHECK(false, name, " is only available on CUDA device."); \
  }
#else
#define GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(device_type, name, ...) \
  TORCH_CHECK(false, name, " is only available on CUDA device.");
#endif

// This includes all integer, float and boolean types.
#define GRAPHBOLT_DISPATCH_CASE_ALL_TYPES(...)            \
  AT_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__)                 \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)     \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Bool, __VA_ARGS__)

#define GRAPHBOLT_DISPATCH_ALL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, GRAPHBOLT_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__))

}  // namespace graphbolt

#endif  // GRAPHBOLT_MACRO_H_
