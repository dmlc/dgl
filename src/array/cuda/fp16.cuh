/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/fp16.cuh
 * \brief float16 related functions.
 * \note this file is modified from TVM project:
 *       https://github.com/apache/tvm/blob/e561007f0c330e3d14c2bc8a3ef40fb741db9004/src/target/source/literal/cuda_half_t.h.
 */
#ifndef DGL_ARRAY_FP16_CUH_
#define DGL_ARRAY_FP16_CUH_

#include <cuda_fp16.h>

#if CUDART_VERSION < 11000  // CUDA 11 already implements max/min

__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}

__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}

#endif  // CUDART_VERSION < 11000

#endif  // DGL_ARRAY_FP16_CUH_