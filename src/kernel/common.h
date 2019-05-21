/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/common.h
 * \brief Kernel common utilities
 */
#ifndef DGL_KERNEL_COMMON_H_
#define DGL_KERNEL_COMMON_H_

#include <dgl/runtime/ndarray.h>

#include <cstdint>

namespace dgl {
namespace kernel {

#ifdef __CUDACC__
#define DGLDEVICE __device__
#define DGLINLINE __forceinline__
#else
#define DGLDEVICE
#define DGLINLINE inline
#endif  // __CUDACC__

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

#if 0
#define GEN_DTYPE(GEN, ...)  \
  GEN(__VA_ARGS__, float)    \
  GEN(__VA_ARGS__, double)   \
  GEN(__VA_ARGS__, int32_t)  \
  GEN(__VA_ARGS__, int64_t)

#define DGL_DTYPE_SWITCH(val, DType, ...)                   \
  if (val.code == kDLInt && val.bits == 32) {               \
    typedef int32_t DType;                                  \
    {__VA_ARGS__}                                           \
  } else if (val.code == kDLInt && val.bits == 64) {        \
    typedef int64_t DType;                                  \
    {__VA_ARGS__}                                           \
  } else if (val.code == kDLFloat && val.bits == 32) {      \
    typedef float DType;                                    \
    {__VA_ARGS__}                                           \
  } else if (val.code == kDLFloat && val.bits == 64) {      \
    typedef double DType;                                   \
    {__VA_ARGS__}                                           \
  } else {                                                  \
    LOG(FATAL) << "Unsupported dtype: " << val.code << "_"  \
               << val.bits;                                 \
  }
#else

// MSVC does not expand __VA_ARGS__ correctly, and needs this expand hack
#define MSVC_EXPAND(x) x

#define GEN_DTYPE(GEN, ...)  \
  MSVC_EXPAND(GEN(__VA_ARGS__, float))

#define DGL_DTYPE_SWITCH(val, DType, ...)                   \
  if (val.code == kDLFloat && val.bits == 32) {             \
    typedef float DType;                                    \
    {__VA_ARGS__}                                           \
  } else {                                                  \
    LOG(FATAL) << "Unsupported dtype: " << val.code << "_"  \
               << val.bits;                                 \
  }
#endif

inline bool IsValidCsr(runtime::NDArray indptr, runtime::NDArray indices) {
  return (indptr->ndim == 1) && (indptr->dtype.code == kDLInt) && (indptr->dtype.bits == 32)
    && (indices->ndim == 1) && (indices->dtype.code == kDLInt) && (indices->dtype.bits == 32);
}

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_COMMON_H_
