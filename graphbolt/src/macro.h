/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/src/macro.h
 * @brief Common macros for graphbolt package.
 */

#ifndef GRAPHBOLT_MACRO_H_
#define GRAPHBOLT_MACRO_H_

#include <torch/torch.h>

/**
 * Dispatch according to torch scalar type.
 *
 * ATEN_ID_TYPE_SWITCH(type_per_edge.dtype(), DataType, {
 *   // Now DataType is the type corresponding to data type in type_per_edge.
 *   // For instance, one can do this for a CPU array:
 *   DataType *data = static_cast<DataType *>(type_per_edge.data_ptr());
 * });
 */
#define ATEN_DATA_TYPE_SWITCH(val, DataType, ...)                           \
  do {                                                                      \
    if ((val) == torch::kInt8 || (val) == torch::kUInt8 ||                  \
        (val) == torch::kBool) {                                            \
      typedef int8_t DataType;                                              \
      { __VA_ARGS__ }                                                       \
    } else if ((val) == torch::kInt16) {                                    \
      typedef int16_t DataType;                                             \
      { __VA_ARGS__ }                                                       \
    } else if ((val) == torch::kInt32) {                                    \
      typedef uint32_t DataType;                                            \
      { __VA_ARGS__ }                                                       \
    } else if ((val) == torch::kInt64) {                                    \
      typedef uint64_t DataType;                                            \
      { __VA_ARGS__ }                                                       \
    }                                                                       \
  }                                                                         \
  else if ((val) == torch::kFloat) {                                        \
    typedef float ProbType;                                                 \
    { __VA_ARGS__ }                                                         \
  }                                                                         \
  else if ((val) == torch::kDouble) {                                       \
    typedef double ProbType;                                                \
    { __VA_ARGS__ }                                                         \
    else {                                                                  \
      LOG(FATAL) << "Probs can only be bool, uint8, int8, float or double"; \
    }                                                                       \
  }                                                                         \
  while (0)

#endif  // GRAPHBOLT_MACRO_H_
