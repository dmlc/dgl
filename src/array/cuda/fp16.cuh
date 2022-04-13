/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/fp16.cuh
 * \brief float16 related functions.
 * \note this file is modified from TVM project:
 *       https://github.com/apache/tvm/blob/e561007f0c330e3d14c2bc8a3ef40fb741db9004/src/target/source/literal/cuda_half_t.h.
 */
#ifndef DGL_ARRAY_FP16_CUH_
#define DGL_ARRAY_FP16_CUH_


#ifdef USE_FP16
#include <cuda_fp16.h>

static __device__ __forceinline__ half max(half a, half b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hgt(__half(a), __half(b)) ? a : b;
#else
  return a;
#endif
}

static __device__ __forceinline__ half min(half a, half b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hlt(__half(a), __half(b)) ? a : b;
#else
  return a;
#endif
}
#endif  // USE_FP16

#endif  // DGL_ARRAY_FP16_CUH_
