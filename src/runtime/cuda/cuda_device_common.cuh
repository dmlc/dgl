/*!
 *  Copyright (c) 2021 by Contributors
 * \file runtime/cuda/cuda_device_common.cuh 
 * \brief Device level functions for within cuda kernels. 
 */

#ifndef DGL_RUNTIME_CUDA_CUDA_DEVICE_COMMON_CUH_
#define DGL_RUNTIME_CUDA_CUDA_DEVICE_COMMON_CUH_

#include <cuda_runtime.h>

namespace dgl {
namespace runtime {
namespace cuda {

/**
* \brief Performs an atomic compare-and-swap on 64 bit integers. That is,
* it the word `old` at the memory location `address`, computes
* `(old == compare ? val : old)` , and stores the result back to memory at
* the same address.
*
* \param address The address to perform the atomic operation on.
* \param compare The value to compare to.
* \param val The new value to conditionally store.
*
* \return The old value at the address.
*/
inline __device__ int64_t AtomicCAS(
    int64_t * const address,
    const int64_t compare,
    const int64_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = unsigned long long int; // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return ::atomicCAS(reinterpret_cast<Type*>(address),
                   static_cast<Type>(compare),
                   static_cast<Type>(val));
}

/**
* \brief Performs an atomic compare-and-swap on 32 bit integers. That is,
* it the word `old` at the memory location `address`, computes
* `(old == compare ? val : old)` , and stores the result back to memory at
* the same address.
*
* \param address The address to perform the atomic operation on.
* \param compare The value to compare to.
* \param val The new value to conditionally store.
*
* \return The old value at the address.
*/
inline __device__ int32_t AtomicCAS(
    int32_t * const address,
    const int32_t compare,
    const int32_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = int; // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return ::atomicCAS(reinterpret_cast<Type*>(address),
                   static_cast<Type>(compare),
                   static_cast<Type>(val));
}

}  // cuda
}  // runtime
}  // dgl

#endif

