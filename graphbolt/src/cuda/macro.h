/**
 *  Copyright (c) 2023 by Contributors
 * @file cuda/macro.h
 * @brief CUDA macro definitions.
 */
#ifndef GRAPHBOLT_CUDA_MACRO_H_
#define GRAPHBOLT_CUDA_MACRO_H_

namespace graphbolt {

/**
 * @brief CUDA kernel launch macro.
 *
 * @param kernel Kernel function name.
 * @param nblks Number of blocks.
 * @param nthrs Number of threads per block.
 * @param shmem Shared memory size.
 * @param stream CUDA stream.
 * @param ... Kernel arguments.
 */
#define GRAPHBOLT_CUDA_KERNEL_CALL(kernel, nblks, nthrs, shmem, stream, ...) \
  {                                                                          \
    (kernel)<<<(nblks), (nthrs), (shmem), (stream)>>>(__VA_ARGS__);          \
    cudaError_t e = cudaGetLastError();                                      \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)                 \
        << "CUDA kernel launch error: " << cudaGetErrorString(e);            \
  }

}  // namespace graphbolt

#endif  // GRAPHBOLT_CUDA_MACRO_H_
