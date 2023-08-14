/**
 *  Copyright (c) 2022 by Contributors
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
 * @file array/cuda/bf16.cuh
 * @brief bfloat16 related functions.
 */
#ifndef DGL_ARRAY_CUDA_BF16_CUH_
#define DGL_ARRAY_CUDA_BF16_CUH_

#if BF16_ENABLED
#include <cuda_bf16.h>

#include <algorithm>

static __device__ __forceinline__ __nv_bfloat16
max(__nv_bfloat16 a, __nv_bfloat16 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hmax(a, b);
#else
  return __nv_bfloat16(max(float(a), float(b)));  // NOLINT
#endif
}

static __device__ __forceinline__ __nv_bfloat16
min(__nv_bfloat16 a, __nv_bfloat16 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hmin(a, b);
#else
  return __nv_bfloat16(min(float(a), float(b)));  // NOLINT
#endif
}

#ifdef __CUDACC__
// Arithmetic BF16 operations for architecture >= 8.0 are already defined in
// cuda_bf16.h
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
// CUDA 12.2 adds "emulated" support for older architectures.
#if defined(CUDART_VERSION) && (CUDART_VERSION < 12020)
__device__ __forceinline__ __nv_bfloat16
operator+(const __nv_bfloat16& lh, const __nv_bfloat16& rh) {
  return __nv_bfloat16(float(lh) + float(rh));  // NOLINT
}
__device__ __forceinline__ __nv_bfloat16
operator-(const __nv_bfloat16& lh, const __nv_bfloat16& rh) {
  return __nv_bfloat16(float(lh) - float(rh));  // NOLINT
}
__device__ __forceinline__ __nv_bfloat16
operator*(const __nv_bfloat16& lh, const __nv_bfloat16& rh) {
  return __nv_bfloat16(float(lh) * float(rh));  // NOLINT
}
__device__ __forceinline__ __nv_bfloat16
operator/(const __nv_bfloat16& lh, const __nv_bfloat16& rh) {
  return __nv_bfloat16(float(lh) / float(rh));  // NOLINT
}

__device__ __forceinline__ __nv_bfloat16& operator+=(
    __nv_bfloat16& lh, const __nv_bfloat16& rh) {  // NOLINT
  lh = __nv_bfloat16(float(lh) + float(rh));       // NOLINT
  return lh;
}
__device__ __forceinline__ __nv_bfloat16& operator-=(
    __nv_bfloat16& lh, const __nv_bfloat16& rh) {  // NOLINT
  lh = __nv_bfloat16(float(lh) - float(rh));       // NOLINT
  return lh;
}
__device__ __forceinline__ __nv_bfloat16& operator*=(
    __nv_bfloat16& lh, const __nv_bfloat16& rh) {  // NOLINT
  lh = __nv_bfloat16(float(lh) * float(rh));       // NOLINT
  return lh;
}
__device__ __forceinline__ __nv_bfloat16& operator/=(
    __nv_bfloat16& lh, const __nv_bfloat16& rh) {  // NOLINT
  lh = __nv_bfloat16(float(lh) / float(rh));       // NOLINT
  return lh;
}

__device__ __forceinline__ __nv_bfloat16& operator++(
    __nv_bfloat16& h) {                // NOLINT
  h = __nv_bfloat16(float(h) + 1.0f);  // NOLINT
  return h;
}
__device__ __forceinline__ __nv_bfloat16& operator--(
    __nv_bfloat16& h) {                // NOLINT
  h = __nv_bfloat16(float(h) - 1.0f);  // NOLINT
  return h;
}
__device__ __forceinline__ __nv_bfloat16
operator++(__nv_bfloat16& h, int) {  // NOLINT
  __nv_bfloat16 ret = h;
  h = __nv_bfloat16(float(h) + 1.0f);  // NOLINT
  return ret;
}
__device__ __forceinline__ __nv_bfloat16
operator--(__nv_bfloat16& h, int) {  // NOLINT
  __nv_bfloat16 ret = h;
  h = __nv_bfloat16(float(h) - 1.0f);  // NOLINT
  return ret;
}

__device__ __forceinline__ __nv_bfloat16 operator+(const __nv_bfloat16& h) {
  return h;
}
__device__ __forceinline__ __nv_bfloat16 operator-(const __nv_bfloat16& h) {
  return __nv_bfloat16(-float(h));  // NOLINT
}

__device__ __forceinline__ bool operator==(
    const __nv_bfloat16& lh, const __nv_bfloat16& rh) {
  return float(lh) == float(rh);  // NOLINT
}
__device__ __forceinline__ bool operator!=(
    const __nv_bfloat16& lh, const __nv_bfloat16& rh) {
  return float(lh) != float(rh);  // NOLINT
}
__device__ __forceinline__ bool operator>(
    const __nv_bfloat16& lh, const __nv_bfloat16& rh) {
  return float(lh) > float(rh);  // NOLINT
}
__device__ __forceinline__ bool operator<(
    const __nv_bfloat16& lh, const __nv_bfloat16& rh) {
  return float(lh) < float(rh);  // NOLINT
}
__device__ __forceinline__ bool operator>=(
    const __nv_bfloat16& lh, const __nv_bfloat16& rh) {
  return float(lh) >= float(rh);  // NOLINT
}
__device__ __forceinline__ bool operator<=(
    const __nv_bfloat16& lh, const __nv_bfloat16& rh) {
  return float(lh) <= float(rh);  // NOLINT
}
#endif  // defined(CUDART_VERSION) && (CUDART_VERSION < 12020)
#endif  // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
#endif  // __CUDACC__

#endif  // BF16_ENABLED

#endif  // DGL_ARRAY_CUDA_BF16_CUH_
