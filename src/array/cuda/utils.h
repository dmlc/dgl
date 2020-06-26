/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/utils.h
 * \brief Utilities for CUDA kernels.
 */
#ifndef DGL_ARRAY_CUDA_UTILS_H_
#define DGL_ARRAY_CUDA_UTILS_H_

#include <dmlc/logging.h>
#include <dlpack/dlpack.h>

namespace dgl {
namespace cuda {

#define CUDA_MAX_NUM_BLOCKS_X 0x7FFFFFFF
#define CUDA_MAX_NUM_BLOCKS_Y 0xFFFF
#define CUDA_MAX_NUM_BLOCKS_Z 0xFFFF
#define CUDA_MAX_NUM_THREADS 1024

/*! \brief Calculate the number of threads needed given the dimension length.
 *
 * It finds the biggest number that is smaller than min(dim, max_nthrs)
 * and is also power of two.
 */
inline int FindNumThreads(int dim, int max_nthrs = CUDA_MAX_NUM_THREADS) {
  CHECK_NE(dim, 0);
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

}  // namespace cuda
}  // namespace dgl

#endif  // DGL_ARRAY_CUDA_UTILS_H_
