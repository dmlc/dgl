#ifndef DGL_KERNEL_CUDA_FUNCTOR_CUH_
#define DGL_KERNEL_CUDA_FUNCTOR_CUH_

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

// Direct write
template <typename DType>
struct StoreWriter {
  static __device__ __forceinline__ void Call(DType* addr, DType val) {
    *addr = val;
  }
};

// Reducer
template <typename DType>
struct AtomicAddWriter {
  static __device__ __forceinline__ void Call(DType* addr, DType val) {
    AtomicAdd(addr, val);
  }
};

template <typename DType>
struct AtomicMaxWriter {
  static __device__ __forceinline__ void Call(DType* addr, DType val) {
    AtomicMax(addr, val);
  }
};

template <typename DType>
struct AtomicMinWriter {
  static __device__ __forceinline__ void Call(DType* addr, DType val) {
    AtomicMin(addr, val);
  }
};

template <typename DType>
struct AtomicMulWriter {
  static __device__ __forceinline__ void Call(DType* addr, DType val) {
    AtomicMul(addr, val);
  }
};

}  // namespace cuda
}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CUDA_FUNCTOR_CUH_
