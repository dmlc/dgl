/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/functor.cuh
 * \brief Functors for template on CUDA
 */
#ifndef DGL_KERNEL_CUDA_FUNCTOR_CUH_
#define DGL_KERNEL_CUDA_FUNCTOR_CUH_

#include "../binary_reduce_common.h"
#include "./atomic.cuh"

namespace dgl {
namespace kernel {
namespace cuda {

// Cache load from global memory
template <typename DType>
struct LDGReader {
  static __device__ __forceinline__ DType Call(DType* addr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(addr);
#else
    return *addr;
#endif
  }
};

}  // namespace cuda

// Reducer functor specialization
template <typename DType>
struct ReduceSum<kDLGPU, DType> {
  static __device__ __forceinline__ void Call(DType* addr, DType val) {
    cuda::AtomicAdd(addr, val);
  }
  static __device__ __forceinline__ DType BackwardCall(DType val, DType accum) {
    return 1;
  }
};

template <typename DType>
struct ReduceMax<kDLGPU, DType> {
  static __device__ __forceinline__ void Call(DType* addr, DType val) {
    cuda::AtomicMax(addr, val);
  }
  static __device__ __forceinline__ DType BackwardCall(DType val, DType accum) {
    return static_cast<DType>(val == accum);
  }
};

template <typename DType>
struct ReduceMin<kDLGPU, DType> {
  static __device__ __forceinline__ void Call(DType* addr, DType val) {
    cuda::AtomicMin(addr, val);
  }
  static __device__ __forceinline__ DType BackwardCall(DType val, DType accum) {
    return static_cast<DType>(val == accum);
  }
};

template <typename DType>
struct ReduceProd<kDLGPU, DType> {
  static __device__ __forceinline__ void Call(DType* addr, DType val) {
    cuda::AtomicMul(addr, val);
  }
  static __device__ __forceinline__ DType BackwardCall(DType val, DType accum) {
    return accum / val;
  }
};

template <typename DType>
struct ReduceNone<kDLGPU, DType> {
  static __device__ __forceinline__ void Call(DType* addr, DType val) {
    *addr = val;
  }
  static __device__ __forceinline__ DType BackwardCall(DType val, DType accum) {
    return 1;
  }
};

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CUDA_FUNCTOR_CUH_
