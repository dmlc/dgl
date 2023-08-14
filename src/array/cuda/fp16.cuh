/**
 *  Copyright (c) 2020-2022 by Contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * @file array/cuda/fp16.cuh
 * @brief float16 related functions.
 * @note this file is modified from TVM project:
 *       https://github.com/apache/tvm/blob/e561007f0c330e3d14c2bc8a3ef40fb741db9004/src/target/source/literal/cuda_half_t.h.
 */
#ifndef DGL_ARRAY_CUDA_FP16_CUH_
#define DGL_ARRAY_CUDA_FP16_CUH_

#include <cuda_fp16.h>

#include <algorithm>

static __device__ __forceinline__ half max(half a, half b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hgt(__half(a), __half(b)) ? a : b;
#else
  return __half(max(float(a), float(b)));  // NOLINT
#endif
}

static __device__ __forceinline__ half min(half a, half b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hlt(__half(a), __half(b)) ? a : b;
#else
  return __half(min(float(a), float(b)));  // NOLINT
#endif
}

#ifdef __CUDACC__
// Arithmetic FP16 operations for architecture >= 5.3 are already defined in
// cuda_fp16.h
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530)
// CUDA 12.2 adds "emulated" support for older architectures.
#if defined(CUDART_VERSION) && (CUDART_VERSION < 12020)
__device__ __forceinline__ __half
operator+(const __half& lh, const __half& rh) {
  return __half(float(lh) + float(rh));  // NOLINT
}
__device__ __forceinline__ __half
operator-(const __half& lh, const __half& rh) {
  return __half(float(lh) - float(rh));  // NOLINT
}
__device__ __forceinline__ __half
operator*(const __half& lh, const __half& rh) {
  return __half(float(lh) * float(rh));  // NOLINT
}
__device__ __forceinline__ __half
operator/(const __half& lh, const __half& rh) {
  return __half(float(lh) / float(rh));  // NOLINT
}

__device__ __forceinline__ __half& operator+=(
    __half& lh, const __half& rh) {    // NOLINT
  lh = __half(float(lh) + float(rh));  // NOLINT
  return lh;
}
__device__ __forceinline__ __half& operator-=(
    __half& lh, const __half& rh) {    // NOLINT
  lh = __half(float(lh) - float(rh));  // NOLINT
  return lh;
}
__device__ __forceinline__ __half& operator*=(
    __half& lh, const __half& rh) {    // NOLINT
  lh = __half(float(lh) * float(rh));  // NOLINT
  return lh;
}
__device__ __forceinline__ __half& operator/=(
    __half& lh, const __half& rh) {    // NOLINT
  lh = __half(float(lh) / float(rh));  // NOLINT
  return lh;
}

__device__ __forceinline__ __half& operator++(__half& h) {  // NOLINT
  h = __half(float(h) + 1.0f);                              // NOLINT
  return h;
}
__device__ __forceinline__ __half& operator--(__half& h) {  // NOLINT
  h = __half(float(h) - 1.0f);                              // NOLINT
  return h;
}
__device__ __forceinline__ __half operator++(__half& h, int) {  // NOLINT
  __half ret = h;
  h = __half(float(h) + 1.0f);  // NOLINT
  return ret;
}
__device__ __forceinline__ __half operator--(__half& h, int) {  // NOLINT
  __half ret = h;
  h = __half(float(h) - 1.0f);  // NOLINT
  return ret;
}

__device__ __forceinline__ __half operator+(const __half& h) { return h; }
__device__ __forceinline__ __half operator-(const __half& h) {
  return __half(-float(h));  // NOLINT
}

__device__ __forceinline__ bool operator==(const __half& lh, const __half& rh) {
  return float(lh) == float(rh);  // NOLINT
}
__device__ __forceinline__ bool operator!=(const __half& lh, const __half& rh) {
  return float(lh) != float(rh);  // NOLINT
}
__device__ __forceinline__ bool operator>(const __half& lh, const __half& rh) {
  return float(lh) > float(rh);  // NOLINT
}
__device__ __forceinline__ bool operator<(const __half& lh, const __half& rh) {
  return float(lh) < float(rh);  // NOLINT
}
__device__ __forceinline__ bool operator>=(const __half& lh, const __half& rh) {
  return float(lh) >= float(rh);  // NOLINT
}
__device__ __forceinline__ bool operator<=(const __half& lh, const __half& rh) {
  return float(lh) <= float(rh);  // NOLINT
}
#endif  // defined(CUDART_VERSION) && (CUDART_VERSION < 12020)
#endif  // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530)
#endif  // __CUDACC__

#endif  // DGL_ARRAY_CUDA_FP16_CUH_
