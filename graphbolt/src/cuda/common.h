/**
 *  Copyright (c) 2017-2023 by Contributors
 * @file cuda_common.h
 * @brief Common utilities for CUDA
 */
#ifndef GRAPHBOLT_CUDA_COMMON_H_
#define GRAPHBOLT_CUDA_COMMON_H_

#include <cuda_runtime.h>

namespace graphbolt {
namespace cuda {

template <typename T>
inline bool is_zero(T size) {
  return size == 0;
}

template <>
inline bool is_zero<dim3>(dim3 size) {
  return size.x == 0 || size.y == 0 || size.z == 0;
}

#define CUDA_DRIVER_CALL(x)                                             \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) { \
      const char* msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      LOG(FATAL) << "CUDAError: " #x " failed with error: " << msg;     \
    }                                                                   \
  }

#define CUDA_CALL(func)                                      \
  {                                                          \
    cudaError_t e = (func);                                  \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                \
  }

#define CUDA_KERNEL_CALL(kernel, nblks, nthrs, shmem, stream, ...)    \
  {                                                                   \
    if (!graphbolt::cuda::is_zero((nblks)) &&                         \
        !graphbolt::cuda::is_zero((nthrs))) {                         \
      (kernel)<<<(nblks), (nthrs), (shmem), (stream)>>>(__VA_ARGS__); \
      cudaError_t e = cudaGetLastError();                             \
      CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)        \
          << "CUDA kernel launch error: " << cudaGetErrorString(e);   \
    }                                                                 \
  }

}  // namespace cuda
}  // namespace graphbolt
#endif  // GRAPHBOLT_CUDA_COMMON_H_
