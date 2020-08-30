/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/common.h
 * \brief Kernel common utilities
 */
#ifndef DGL_KERNEL_COMMON_H_
#define DGL_KERNEL_COMMON_H_

#include <dgl/runtime/ndarray.h>

#include <cstdint>
#include "../c_api_common.h"

namespace dgl {
namespace kernel {

#ifdef __CUDACC__
#define DGLDEVICE __device__
#define DGLINLINE __forceinline__
#else
#define DGLDEVICE
#define DGLINLINE inline
#endif  // __CUDACC__

// Macro for dispatch device flag to template function calls
#ifdef DGL_USE_CUDA
#define DGL_XPU_SWITCH(val, Method, ...)  \
  if (val == kDLCPU) {                    \
    Method<kDLCPU>(__VA_ARGS__);          \
  } else if (val == kDLGPU) {             \
    Method<kDLGPU>(__VA_ARGS__);          \
  } else {                                \
    LOG(FATAL) << "Unsupported device type: " << val;  \
  }
#else  // DGL_USE_CUDA
#define DGL_XPU_SWITCH(val, Method, ...)  \
  if (val == kDLCPU) {                    \
    Method<kDLCPU>(__VA_ARGS__);          \
  } else {                                \
    LOG(FATAL) << "Unsupported device type: " << val;  \
  }
#endif  // DGL_USE_CUDA

// MSVC does not expand __VA_ARGS__ correctly, and needs this expand hack
#define MSVC_EXPAND(x) x

// Macro for dispatch dtype flag to template argument. Currently only
// support float32.
#define DGL_DTYPE_SWITCH(val, DType, ...)       \
  if (val.code == kDLFloat && val.bits == 32) { \
    typedef float DType;                        \
    { __VA_ARGS__ }                             \
  } else {                                      \
    LOG(FATAL) << "Unsupported dtype: " << val; \
  }

// Macro for unrolling with data type arguments.
#define GEN_DTYPE(GEN, ...)  \
  MSVC_EXPAND(GEN(__VA_ARGS__, float))

// Macro for dispatch index nbits to template argument.
#ifdef __CUDACC__
#define DGL_IDX_TYPE_SWITCH(bits, Idx, ...)            \
  if (bits == 32) {                                    \
    typedef int32_t Idx;                               \
    {__VA_ARGS__}                                      \
  } else {                                             \
    LOG(FATAL) << "Unsupported idx bits: " << bits;    \
  }
#else
#define DGL_IDX_TYPE_SWITCH(bits, Idx, ...)            \
  if (bits == 32) {                                    \
    typedef int32_t Idx;                               \
    {__VA_ARGS__}                                      \
  } else if (bits == 64) {                             \
    typedef int64_t Idx;                               \
    {__VA_ARGS__}                                      \
  } else {                                             \
    LOG(FATAL) << "Unsupported idx bits: " << bits;    \
  }
#endif

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_COMMON_H_
