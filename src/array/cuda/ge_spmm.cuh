/*!
 * Copyright (c) 2020 by Contributors
 * \file array/cuda/ge_spmm.cuh
 * \brief GE-SpMM CUDA kernel function header.
 */
#ifndef DGL_ARRAY_CUDA_GE_SPMM_CUH_
#define DGL_ARRAY_CUDA_GE_SPMM_CUH_

#include "macro.cuh"
#include "atomic.cuh"
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {

using namespace cuda;

namespace aten {
namespace cuda {

/*! 
 * \brief: TODO(zihao)
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
__global__ void GESpMMCsrKernel(
    const DType* __restrict__ ufeat,
    const DType* __restrict__ efeat,
    DType* __restrict__ out,
    Idx* __restrict__ arg_u,
    Idx* __restrict__ arg_e,
    int64_t N, int64_t M, int64_t E,
    int64_t feat_len) {
  // TODO(zihao)
}


}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif
