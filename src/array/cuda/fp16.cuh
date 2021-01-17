#ifndef DGL_ARRAY_FP16_CUH_
#define DGL_ARRAY_FP16_CUH_

#include <cuda_fp16.h>

/*
__device__ half max(half a, half b) {
  return __hgt(__half(a), __half(b)) ? a : b;
}

__device__ half min(half a, half b) {
  return __hlt(__half(a), __half(b)) ? a : b;
}
*/

#endif  // DGL_ARRAY_FP16_CUH_