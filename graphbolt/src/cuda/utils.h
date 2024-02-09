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

/**
 * @brief Given a sorted array and a value this function returns the index
 * of the first element which compares greater than or equal to value.
 *
 * This function assumes 0-based index
 * @param A: ascending sorted array
 * @param n: size of the A
 * @param x: value to search in A
 * @return index, i, of the first element st. A[i]>=x. If x>A[n-1] returns n.
 * if x<A[0] then it returns 0.
 */
template <typename indptr_t, typename indices_t>
__device__ indices_t LowerBound(const indptr_t* A, indices_t n, indptr_t x) {
  indices_t l = 0, r = n;
  while (l < r) {
    const auto m = l + (r - l) / 2;
    if (x > A[m]) {
      l = m + 1;
    } else {
      r = m;
    }
  }
  return l;
}

/**
 * @brief Given a sorted array and a value this function returns the index
 * of the first element which compares greater than value.
 *
 * This function assumes 0-based index
 * @param A: ascending sorted array
 * @param n: size of the A
 * @param x: value to search in A
 * @return index, i, of the first element st. A[i]>x. If x>=A[n-1] returns n.
 * if x<A[0] then it returns 0.
 */
template <typename indptr_t, typename indices_t>
__device__ indices_t UpperBound(const indptr_t* A, indices_t n, indptr_t x) {
  indices_t l = 0, r = n;
  while (l < r) {
    const auto m = l + (r - l) / 2;
    if (x >= A[m]) {
      l = m + 1;
    } else {
      r = m;
    }
  }
  return l;
}

}  // namespace cuda
}  // namespace graphbolt

#endif  // GRAPHBOLT_CUDA_UTILS_H_
