/**
 *  Copyright (c) 2023 by Contributors
 * @file macro.h
 * @brief Graphbolt macros.
 */

#ifndef GRAPHBOLT_MACRO_H_
#define GRAPHBOLT_MACRO_H_

#include <torch/script.h>

namespace graphbolt {

/**
 * Dispatch Pytorch data type to template functions. Available data types are
 * float32, float64, int32, and int64.
 *
 * GRAPHBOLT_DTYPE_SWITCH(tensor.scalar_type(), DType, {
 *   // Now DType is the type corresponding to data type in the tensor.
 *   // Then you can use DType to template your function.
 *   DType *data = tensor.data_ptr<DType>();
 * });
 */
#define GRAPHBOLT_DTYPE_SWITCH(val, DType, val_name, ...)           \
  do {                                                              \
    if ((val) == torch::kFloat32) {                                 \
      typedef float DType;                                          \
      { __VA_ARGS__ }                                               \
    } else if ((val) == torch::kFloat64) {                          \
      typedef double DType;                                         \
      { __VA_ARGS__ }                                               \
    } else if ((val) == torch::kInt32) {                            \
      typedef int32_t DType;                                        \
      { __VA_ARGS__ }                                               \
    } else if ((val) == torch::kInt64) {                            \
      typedef int64_t DType;                                        \
      { __VA_ARGS__ }                                               \
    } else {                                                        \
      TORCH_CHECK(false, (val_name), " must be float or int type"); \
    }                                                               \
  } while (0)

/**
 * Dispatch Pytorch index type to template functions. Available index types are
 * int32 and int64.
 *
 * GRAPHBOLT_ID_TYPE_SWITCH(tensor.scalar_type(), IdType, {
 *  // Now IdType is the type corresponding to index type in the tensor.
 *  // Then you can use IdType to template your function.
 *  IdType *data = tensor.data_ptr<IdType>();
 * });
 */
#define GRAPHBOLT_ID_TYPE_SWITCH(val, IdType, val_name, ...)          \
  do {                                                                \
    TORCH_CHECK(                                                      \
        (val) == torch::kInt32 || (val) == torch::kInt64, (val_name), \
        " must be int type");                                         \
    if ((val) == torch::kInt32) {                                     \
      typedef int32_t IdType;                                         \
      { __VA_ARGS__ }                                                 \
    } else {                                                          \
      typedef int64_t IdType;                                         \
      { __VA_ARGS__ }                                                 \
    }                                                                 \
  } while (0)

}  // namespace graphbolt

#endif  // GRAPHBOLT_MACRO_H_
