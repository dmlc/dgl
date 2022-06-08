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
  return __half(max(float(a), float(b)));
#endif
}

static __device__ __forceinline__ half min(half a, half b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hlt(__half(a), __half(b)) ? a : b;
#else
  return __half(min(float(a), float(b)));
#endif
}

#ifdef __CUDACC__
// Arithmetic FP16 operations for architecture >= 5.3 are already defined in cuda_fp16.h
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530)
__device__ __forceinline__ __half operator+(const __half& lh, const __half& rh) { return __half(float(lh) + float(rh)); }
__device__ __forceinline__ __half operator-(const __half& lh, const __half& rh) { return __half(float(lh) - float(rh)); }
__device__ __forceinline__ __half operator*(const __half& lh, const __half& rh) { return __half(float(lh) * float(rh)); }
__device__ __forceinline__ __half operator/(const __half& lh, const __half& rh) { return __half(float(lh) / float(rh)); }

__device__ __forceinline__ __half& operator+=(__half& lh, const __half& rh) { lh = __half(float(lh) + float(rh)); return lh; }
__device__ __forceinline__ __half& operator-=(__half& lh, const __half& rh) { lh = __half(float(lh) - float(rh)); return lh; }
__device__ __forceinline__ __half& operator*=(__half& lh, const __half& rh) { lh = __half(float(lh) * float(rh)); return lh; }
__device__ __forceinline__ __half& operator/=(__half& lh, const __half& rh) { lh = __half(float(lh) / float(rh)); return lh; }

__device__ __forceinline__ __half& operator++(__half& h)      { h = __half(float(h) + 1.0f); return h; }
__device__ __forceinline__ __half& operator--(__half& h)      { h = __half(float(h) - 1.0f); return h; }
__device__ __forceinline__ __half  operator++(__half& h, int) { __half ret = h; h = __half(float(h) + 1.0f); return ret; }
__device__ __forceinline__ __half  operator--(__half& h, int) { __half ret = h; h = __half(float(h) - 1.0f); return ret; }

__device__ __forceinline__ __half operator+(const __half& h) { return h; }
__device__ __forceinline__ __half operator-(const __half& h) { return __half(-float(h)); }

__device__ __forceinline__ bool operator==(const __half& lh, const __half& rh) { return float(lh) == float(rh); }
__device__ __forceinline__ bool operator!=(const __half& lh, const __half& rh) { return float(lh) != float(rh); }
__device__ __forceinline__ bool operator> (const __half& lh, const __half& rh) { return float(lh) >  float(rh); }
__device__ __forceinline__ bool operator< (const __half& lh, const __half& rh) { return float(lh) <  float(rh); }
__device__ __forceinline__ bool operator>=(const __half& lh, const __half& rh) { return float(lh) >= float(rh); }
__device__ __forceinline__ bool operator<=(const __half& lh, const __half& rh) { return float(lh) <= float(rh); }
#endif  // __CUDA_ARCH__ < 530
#endif  // __CUDACC__

#endif  // USE_FP16

#endif  // DGL_ARRAY_FP16_CUH_
