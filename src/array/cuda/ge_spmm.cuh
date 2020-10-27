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
__global__ void GESpMMSumKernel(
    const DType* __restrict__ ufeat,
    const DType* __restrict__ efeat,
    DType* __restrict__ out,
    Idx* __restrict__ arg_u,
    Idx* __restrict__ arg_e,
    const Idx* __restrict__ indptr,
    const Idx* __restrict__ indices,
    const int64_t num_rows, const int64_t num_cols,
    const int64_t feat_len) {
  /*
  extern __shared__ char smem[];
  Idx* col = nullptr;
  DType* val = nullptr;
  if (BinaryOp::use_rhs) {  // use edge feature.
    if (sizeof(Idx) >= sizeof(DType)) {
      // handle alignment issue: https://forums.developer.nvidia.com/t/dynamic-shared-memory-allocation/21671/3
      col = (Idx*) smem;
      val = (DType*) &col[blockDim.y * blockDim.x];
    } else {
      val = (DType*) smem;
      col = (Idx*) &val[blockDim.y * blockDim.x];
    }
  } else {
    col = (Idx*) smem;
  }
*/
  __shared__ Idx col[8 * 32];
  //__shared__ DType val[BinaryOp::use_rhs ? 8 * 32 : 1];
  DType *val = nullptr;

  int ty = blockIdx.y * blockDim.y + threadIdx.y;  // iterate over destination nodes
  const Idx stride_y = blockDim.y * gridDim.y;
  const Idx sm_offset = 32 * threadIdx.y;

  while (ty < num_rows) {
    const Idx rid = ty;
    const Idx low = indptr[ty], high = indptr[ty + 1];
    const Idx tx = threadIdx.x;
    int fid = (blockIdx.x * 64) + tx;  // iterate over feature dimension

    DType accum_0 = ReduceOp::zero,
          accum_1 = ReduceOp::zero;
    Idx argu_0 = 0, arge_0 = 0,
        argu_1 = 0, arge_1 = 0;

    if (blockIdx.x < gridDim.x) {
      for (int left = low; left < high; left += 32) {
        if (left + tx < high) {
          col[sm_offset + tx] = indices[left + tx]; 
          if (BinaryOp::use_rhs)
            val[sm_offset + tx] = efeat[left + tx];
        }

#pragma unroll
        for (int i = 0; i < 32; ++i) {
          const Idx eid = left + i; 
          if (eid < high) {
            const Idx cid = col[sm_offset + i];
            const Idx offset = feat_len * cid + fid;
            if (BinaryOp::use_rhs) {
              const DType weight = val[sm_offset + i];
              ReduceOp::Call(&accum_0, &argu_0, &arge_0,
                BinaryOp::Call(ufeat + offset, &weight), cid, eid);
              ReduceOp::Call(&accum_1, &argu_1, &arge_1,
                BinaryOp::Call(ufeat + offset + 32, &weight), cid, eid);
            } else {
              ReduceOp::Call(&accum_0, &argu_0, &arge_0,
                ufeat[offset], fid, eid);
              ReduceOp::Call(&accum_1, &argu_1, &arge_1,
                ufeat[offset + 32], cid, eid);
            }
          }
        }

        out[feat_len * rid + fid] = accum_0;
        if (ReduceOp::require_arg && BinaryOp::use_lhs)
          arg_u[feat_len * rid + fid] = argu_0;
        if (ReduceOp::require_arg && BinaryOp::use_rhs)
          arg_e[feat_len * rid + fid] = arge_0;

        out[feat_len * rid + fid + 32] = accum_1;
        if (ReduceOp::require_arg && BinaryOp::use_rhs)
          arg_u[feat_len * rid + fid + 32] = argu_1;
        if (ReduceOp::require_arg && BinaryOp::use_rhs)
          arg_e[feat_len * rid + fid + 32] = arge_1; 
      }
    } else {
      bool left_inbound = fid < feat_len,
           right_inbound = fid + 32 < feat_len;
      for (int left = low; left < high; left += 32) {
        if (left + tx < high) {
          col[sm_offset + tx] = indices[left + tx]; 
          if (BinaryOp::use_rhs)
            val[sm_offset + tx] = efeat[left + tx];
        }

#pragma unroll
        for (int i = 0; i < 32; ++i) {
          const Idx eid = left + i; 
          if (eid < high) {
            const Idx cid = col[sm_offset + i];
            const Idx offset = feat_len * cid + fid;
            if (BinaryOp::use_rhs) {
              const DType weight = val[sm_offset + i];
              if (left_inbound)
                ReduceOp::Call(&accum_0, &argu_0, &arge_0,
                  BinaryOp::Call(ufeat + offset, &weight), cid, eid);
              if (right_inbound)
                ReduceOp::Call(&accum_1, &argu_1, &arge_1,
                  BinaryOp::Call(ufeat + offset + 32, &weight), cid, eid);
            } else {
              if (left_inbound)
                ReduceOp::Call(&accum_0, &argu_0, &arge_0,
                  ufeat[offset], fid, eid);
              if (right_inbound)
                ReduceOp::Call(&accum_1, &argu_1, &arge_1,
                  ufeat[offset + 32], cid, eid);
            }
          }
        }

        if (left_inbound) {
          out[feat_len * rid + fid] = accum_0;
          if (ReduceOp::require_arg && BinaryOp::use_lhs)
            arg_u[feat_len * rid + fid] = argu_0;
          if (ReduceOp::require_arg && BinaryOp::use_rhs)
            arg_e[feat_len * rid + fid] = arge_0;
        }

        if (right_inbound) {
          out[feat_len * rid + fid + 32] = accum_1;
          if (ReduceOp::require_arg && BinaryOp::use_rhs)
            arg_u[feat_len * rid + fid + 32] = argu_1;
          if (ReduceOp::require_arg && BinaryOp::use_rhs)
            arg_e[feat_len * rid + fid + 32] = arge_1; 
        }
      }
    }
    ty += stride_y;
  }
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void GESpMMCsr(
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge,
    int64_t feat_len) {
  const Idx *indptr = csr.indptr.Ptr<Idx>();
  const Idx *indices = csr.indices.Ptr<Idx>();
  const DType *ufeat_data = ufeat.Ptr<DType>();
  const DType *efeat_data = efeat.Ptr<DType>();
  DType *out_data = out.Ptr<DType>();
  Idx *argu_data = argu.Ptr<Idx>();
  Idx *arge_data = arge.Ptr<Idx>();

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  
  const int ntx = 32;
  const int nty = 8;
  const int nbx = (feat_len + (ntx * 2) - 1) / (ntx * 2);
  const int nby = FindNumBlocks<'y'>((csr.num_rows + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const int sh_mem_size = 8 * 32 * sizeof(Idx) + (BinaryOp::use_rhs ? 8 * 32 * sizeof(DType) : 0);

  CUDA_KERNEL_CALL((GESpMMSumKernel<Idx, DType, BinaryOp, ReduceOp>),
      nblks, nthrs, 0, thr_entry->stream,
      ufeat_data, efeat_data, out_data, argu_data, arge_data,
      indptr, indices,
      csr.num_rows, csr.num_cols,
      feat_len);
}

}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif
