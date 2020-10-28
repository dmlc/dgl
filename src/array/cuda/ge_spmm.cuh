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
  const Idx rid = blockIdx.x * blockDim.y + threadIdx.y;
  const Idx tx = threadIdx.x;

  if (rid < num_rows) {
    const Idx low = __ldg(indptr + rid), high = __ldg(indptr + rid + 1);
    Idx fid = (blockIdx.y * 64) + tx;  // iterate over feature dimension
    DType accum_0 = ReduceOp::zero,
          accum_1 = ReduceOp::zero;
    Idx argu_0 = 0, arge_0 = 0,
        argu_1 = 0, arge_1 = 0;

    if (blockIdx.y != gridDim.y - 1) {
      for (Idx left = low; left < high; left += 32) {
        for (Idx i = 0; i < 32 && left + i < high; ++i) {
          const Idx eid = left + i; 
          const Idx cid = __ldg(indices + eid);
          const Idx offset = feat_len * cid + fid;
          if (BinaryOp::use_rhs) {
            const DType weight = __ldg(efeat + eid);
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
        for (int i = 0; i < 32 && left + i < high; ++i) {
          const Idx eid = left + i; 
          const Idx cid = __ldg(indices + eid); 
          const Idx offset = feat_len * cid + fid;
          if (BinaryOp::use_rhs) {
            const DType weight = __ldg(efeat + eid);
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
  const int nty = 16;
  const int nby = (feat_len + (ntx * 2) - 1) / (ntx * 2);
  const int nbx = (csr.num_rows + nty - 1) / nty;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const int sh_mem_size = 0;

  CUDA_KERNEL_CALL((GESpMMSumKernel<Idx, DType, BinaryOp, ReduceOp>),
      nblks, nthrs, sh_mem_size, thr_entry->stream,
      ufeat_data, efeat_data, out_data, argu_data, arge_data,
      indptr, indices,
      csr.num_rows, csr.num_cols,
      feat_len);
}

}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif
