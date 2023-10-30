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

// The cache line size of GPU
#define GPU_CACHE_LINE_SIZE 128
// The max number of threads per block
#define CUDA_MAX_NUM_THREADS 1024

/** @brief Calculate the number of threads needed given the dimension length.
 *
 * It finds the biggest number that is smaller than min(dim, max_nthrs)
 * and is also power of two.
 */
inline int FindNumThreads(int dim, int max_num_threads = CUDA_MAX_NUM_THREADS) {
  if (dim == 0) return 1;
  int ret = max_num_threads;
  while (ret > dim) {
    ret = ret >> 1;
  }
  return ret;
}

}  // namespace cuda
}  // namespace graphbolt

#endif  // GRAPHBOLT_CUDA_UTILS_H_