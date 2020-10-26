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
          typename BinaryOp, typename ReduceOp,
	  bool UseEdgeFeat = false>
__global__ void GESpMMSumKernel(
    const DType* __restrict__ ufeat,
    const DType* __restrict__ efeat,
    DType* __restrict__ out,
    const Idx* __restrict__ indptr,
    const Idx* __restrict__ indices,
    int64_t num_rows, int64_t num_cols, int64_t nnz,
    int64_t feat_len) {
  extern __shared__ DType array[];
  Idx* col = (Idx*)array;
  DType* val = array + 32 * sizeof(Idx);
  int sm_offset = 32 * threadIdx.y;

  int ty = blockIdx.y * blockDim.y + threadIdx.y;  // iterate over destination nodes
  const Idx stride_x = blockDim.x * gridDim.x * 64;
  const Idx stride_x = blockDim.y * gridDim.y;
  while (ty < num_rows) {
    const Idx rid = ty;
    const Idx low = indptr[ty], high = indptr[ty + 1];
    const Idx tx = threadIdx.x;
    int cid = (blockIdx.x << 6) + tx;  // iterate over feature dimension
    while (cid < feat_dim) {
      DType accum_0 = ReduceOp::zero,
            accum_1 = ReduceOp::zero;
      for (int left = low; left < high; left += 32) {
        col[sm_offset + tx] = indices[low + tx]; 
	if (UseEdgeFeat)
	  val[sm_offset + tx] = efeat[low + tx];
        __syncwarp();

#pragma unroll
        for (int i = 0; i < 32 && left + i < high; ++i) {
          const Idx offset = feat_len * col[sm_offset + i] + cid;
	  if (UseEdgeFeat) {
	    const DType weight = val[sm_offset + i];
	    accum_0 += ufeat[offset] * weight;
	    if (cid + 32 < feat_len)
	      accum_1 += ufeat[offset + 32] * weight;
	  } else {
	    accum_0 += ufeat[offset];
	    if (cid + 32 < feat_len)
	      accum_1 += ufeat[offset + 32];
	  }
        }
        __syncwarp();
      } 
      out[feat_len * rid + cid] = accum_0;
      if (cid + 32 < feat_len)
        out[feat_len * rid + cid + 32] = accum_1;
      cid += stride_x;
    }
    ty += stride_y;
  }
}


}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif
