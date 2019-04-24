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
};

template <typename DType>
struct ReduceMax<kDLGPU, DType> {
  static __device__ __forceinline__ void Call(DType* addr, DType val) {
    cuda::AtomicMax(addr, val);
  }
};

template <typename DType>
struct ReduceMin<kDLGPU, DType> {
  static __device__ __forceinline__ void Call(DType* addr, DType val) {
    cuda::AtomicMin(addr, val);
  }
};

template <typename DType>
struct ReduceMean<kDLGPU, DType> {
  static __device__ __forceinline__ void Call(DType* addr, DType val) {
    cuda::AtomicAdd(addr, val);
  }
};

template <typename DType>
struct ReduceProd<kDLGPU, DType> {
  static __device__ __forceinline__ void Call(DType* addr, DType val) {
    cuda::AtomicMul(addr, val);
  }
};

template <typename DType>
struct ReduceNone<kDLGPU, DType> {
  static __device__ __forceinline__ void Call(DType* addr, DType val) {
    *addr = val;
  }
};

template <typename IdxType>
struct IndirectId<kDLGPU, IdxType> {
  static __device__ __forceinline__ IdxType Call(IdxType id, IdxType* shuffle_ids) {
    if (shuffle_ids) {
      return cuda::LDGReader<IdxType>::Call(shuffle_ids + id);
    } else {
      return id;
    }
  }
};

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CUDA_FUNCTOR_CUH_
