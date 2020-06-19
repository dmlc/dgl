/*!
 *  Copyright (c) 2020 by Contributors
 * \file cuda_utils.h
 * \brief Utilities for CUDA kernels.
 */
#ifndef DGL_CUDA_UTILS_H_
#define DGL_CUDA_UTILS_H_

#include <dmlc/logging.h>

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

}  // namespace cuda
}  // namespace dgl

#endif  // DGL_CUDA_UTILS_H_
