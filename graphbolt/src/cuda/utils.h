/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file utils.h
 * @brief CUDA utilities.
 */

#ifndef GRAPHBOLT_CUDA_UTILS_H_
#define GRAPHBOLT_CUDA_UTILS_H_

// The cache line size of GPU.
constexpr int GPU_CACHE_LINE_SIZE = 128;
// The max number of threads per block.
constexpr int CUDA_MAX_NUM_THREADS = 1024;

namespace graphbolt {
namespace cuda {

/**
 * @brief Returns the compute capability of the cuda device, e.g. 70 for Volta.
 */
inline int compute_capability(
    int device = cuda::GetCurrentStream().device_index()) {
  int sm_version;
  CUDA_RUNTIME_CHECK(cub::SmVersion(sm_version, device));
  return sm_version / 10;
};

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

/**
 * @brief Calculate the smallest number of bits needed to represent a given
 * range of integers [0, range).
 */
template <typename T>
int NumberOfBits(const T& range) {
  if (range <= 1) {
    // ranges of 0 or 1 require no bits to store
    return 0;
  }

  int bits = 1;
  const auto urange = static_cast<std::make_unsigned_t<T>>(range);
  while (bits < static_cast<int>(sizeof(T) * 8) && (1ull << bits) < urange) {
    ++bits;
  }

  return bits;
}

}  // namespace cuda
}  // namespace graphbolt

#endif  // GRAPHBOLT_CUDA_UTILS_H_
