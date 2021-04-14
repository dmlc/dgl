/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/utils.h
 * \brief Utilities for CUDA kernels.
 */
#ifndef DGL_ARRAY_CUDA_UTILS_H_
#define DGL_ARRAY_CUDA_UTILS_H_

#include <dmlc/logging.h>
#include <dlpack/dlpack.h>
#include "../../runtime/cuda/cuda_common.h"

namespace dgl {
namespace cuda {

#define CUDA_MAX_NUM_BLOCKS_X 0x7FFFFFFF
#define CUDA_MAX_NUM_BLOCKS_Y 0xFFFF
#define CUDA_MAX_NUM_BLOCKS_Z 0xFFFF
#define CUDA_MAX_NUM_THREADS 1024

#ifdef USE_FP16
#define SWITCH_BITS(bits, DType, ...)                           \
  do {                                                          \
    if ((bits) == 16) {                                         \
      typedef half DType;                                       \
      { __VA_ARGS__ }                                           \
    } else if ((bits) == 32) {                                  \
      typedef float DType;                                      \
      { __VA_ARGS__ }                                           \
    } else if ((bits) == 64) {                                  \
      typedef double DType;                                     \
      { __VA_ARGS__ }                                           \
    } else {                                                    \
      LOG(FATAL) << "Data type not recognized with bits " << bits; \
    }                                                           \
  } while (0)
#else  // USE_FP16
#define SWITCH_BITS(bits, DType, ...)                           \
  do {                                                          \
    if ((bits) == 32) {                                         \
      typedef float DType;                                      \
      { __VA_ARGS__ }                                           \
    } else if ((bits) == 64) {                                  \
      typedef double DType;                                     \
      { __VA_ARGS__ }                                           \
    } else {                                                    \
      LOG(FATAL) << "Data type not recognized with bits " << bits; \
    }                                                           \
  } while (0)
#endif  // USE_FP16

/*! \brief Calculate the number of threads needed given the dimension length.
 *
 * It finds the biggest number that is smaller than min(dim, max_nthrs)
 * and is also power of two.
 */
inline int FindNumThreads(int dim, int max_nthrs = CUDA_MAX_NUM_THREADS) {
  CHECK_GE(dim, 0);
  if (dim == 0)
    return 1;
  int ret = max_nthrs;
  while (ret > dim) {
    ret = ret >> 1;
  }
  return ret;
}

/*
 * !\brief Find number of blocks is smaller than nblks and max_nblks
 * on the given axis ('x', 'y' or 'z').
 */
template <char axis>
inline int FindNumBlocks(int nblks, int max_nblks = -1) {
  int default_max_nblks = -1;
  switch (axis) {
    case 'x':
      default_max_nblks = CUDA_MAX_NUM_BLOCKS_X;
      break;
    case 'y':
      default_max_nblks = CUDA_MAX_NUM_BLOCKS_Y;
      break;
    case 'z':
      default_max_nblks = CUDA_MAX_NUM_BLOCKS_Z;
      break;
    default:
      LOG(FATAL) << "Axis " << axis << " not recognized";
      break;
  }
  if (max_nblks == -1)
    max_nblks = default_max_nblks;
  CHECK_NE(nblks, 0);
  if (nblks < max_nblks)
    return nblks;
  return max_nblks;
}

template <typename T>
__device__ __forceinline__ T _ldg(T* addr) {
#if __CUDA_ARCH__ >= 350
  return __ldg(addr);
#else
  return *addr;
#endif
}

/*!
 * \brief Return true if the given bool flag array is all true.
 * The input bool array is in int8_t type so it is aligned with byte address.
 *
 * \param flags The bool array.
 * \param length The length.
 * \param ctx Device context.
 * \return True if all the flags are true.
 */
bool AllTrue(int8_t* flags, int64_t length, const DLContext& ctx);

/*!
 * \brief CUDA Kernel of filling the vector started from ptr of size length
 *        with val.
 * \note internal use only.
 */
template <typename DType>
__global__ void _FillKernel(DType* ptr, size_t length, DType val) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    ptr[tx] = val;
    tx += stride_x;
  }
}

/*! \brief Fill the vector started from ptr of size length with val */
template <typename DType>
void _Fill(DType* ptr, size_t length, DType val) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int nt = FindNumThreads(length);
  int nb = (length + nt - 1) / nt;  // on x-axis, no need to worry about upperbound.
  CUDA_KERNEL_CALL(cuda::_FillKernel, nb, nt, 0, thr_entry->stream, ptr, length, val);
}

}  // namespace cuda
}  // namespace dgl

#endif  // DGL_ARRAY_CUDA_UTILS_H_
