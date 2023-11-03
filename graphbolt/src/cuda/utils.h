/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file utils.h
 * @brief CUDA utilities.
 */

#ifndef GRAPHBOLT_CUDA_UTILS_H_
#define GRAPHBOLT_CUDA_UTILS_H_

namespace graphbolt {
namespace cuda {

// The cache line size of GPU.
#define GPU_CACHE_LINE_SIZE 128
// The max number of threads per block.
#define CUDA_MAX_NUM_THREADS 1024

/**
 * @brief Calculate the number of threads needed given the size of the dimension
 * to be processed.
 *
 * It finds the largest power of two that is less than or equal to the minimum
 * of size and CUDA_MAX_NUM_THREADS.
 */
inline int FindNumThreads(int size) {
  int ret = 1;
  while ((ret << 1) <= std::min(size, CUDA_MAX_NUM_THREADS)) {
    ret <<= 1;
  }
  return ret;
}

}  // namespace cuda
}  // namespace graphbolt

#endif  // GRAPHBOLT_CUDA_UTILS_H_
