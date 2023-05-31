/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/src/macro.h
 * @brief Common macros for graphbolt package.
 */

#ifndef GRAPHBOLT_MACRO_H_
#define GRAPHBOLT_MACRO_H_

#include <torch/torch.h>

/**
 * Dispatch according to integral type (either int8 or int16):
 *
 * ATEN_ID_TYPE_SWITCH(type_per_edge.dtype(), EtypeType, {
 *   // Now EtypeType is the type corresponding to data type in type_per_edge.
 *   // For instance, one can do this for a CPU array:
 *   EtypeType *data = static_cast<EtypeType *>(type_per_edge.data_ptr());
 * });
 */
#define ATEN_ETYPE_TYPE_SWITCH(val, EtypeType, ...)    \
  do {                                                 \
    if ((val) == torch::kInt8 ||         \
        (val) == torch::kUInt8) {        \
      typedef uint8_t EtypeType;                       \
      { __VA_ARGS__ }                                  \
    } else if ((val) == torch::kInt16) { \
      typedef uint16_t EtypeType;                      \
      { __VA_ARGS__ }                                  \
    } else {                                           \
      typedef uint64_t EtypeType;                      \
      { __VA_ARGS__ }                                  \
    }                                                  \
  } while (0)

#endif  // GRAPHBOLT_MACRO_H_