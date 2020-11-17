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
 * \brief CUDA kernel of GE-SpMM on Csr.
 * \note GE-SpMM: https://arxiv.org/pdf/2007.03179.pdf
 *       The grid dimension x and y are reordered for better performance.
 */
template <typename Idx, typename DType, typename BinaryOp>
__global__ void GESpMMKernel(
    const DType* __restrict__ ufeat,
    const DType* __restrict__ efeat,
    DType* __restrict__ out,
    const Idx* __restrict__ indptr,
    const Idx* __restrict__ indices,
    const int64_t num_rows, const int64_t num_cols,
    const int64_t feat_len) {
  const Idx ty = threadIdx.y, tx = threadIdx.x;
  const Idx rid = blockIdx.x * blockDim.y + ty;  // over vertices dimension
  const Idx fid = (blockIdx.y * 64) + tx;        // over feature dimension

  if (rid < num_rows) {
    const Idx u_fid_0 = fid < feat_len ? fid : 0;
    const Idx u_fid_1 = (fid + 32 < feat_len) ? fid + 32 : 0;
    const Idx low = __ldg(indptr + rid), high = __ldg(indptr + rid + 1);
    DType accum_0 = 0., accum_1 = 0.;

    for (Idx left = low; left < high; left += 32) {
      if (left + 32 <= high) {
#pragma unroll
        for (Idx i = 0; i < 32; ++i) {
          const Idx eid = left + i;
          const Idx cid = __ldg(indices + eid);
          const Idx offset_u = feat_len * cid;
          if (BinaryOp::use_rhs) {
            accum_0 += BinaryOp::Call(ufeat + offset_u + u_fid_0, efeat + eid);
            accum_1 += BinaryOp::Call(ufeat + offset_u + u_fid_1, efeat + eid);
          } else {
            accum_0 += ufeat[offset_u + u_fid_0];
            accum_1 += ufeat[offset_u + u_fid_1];
          }
        }
      } else {
        for (Idx i = 0; left + i < high; ++i) {
          const Idx eid = left + i;
          const Idx cid = __ldg(indices + eid);
          const Idx offset_u = feat_len * cid;
          if (BinaryOp::use_rhs) {
            accum_0 += BinaryOp::Call(ufeat + offset_u + u_fid_0, efeat + eid);
            accum_1 += BinaryOp::Call(ufeat + offset_u + u_fid_1, efeat + eid);
          } else {
            accum_0 += ufeat[offset_u + u_fid_0];
            accum_1 += ufeat[offset_u + u_fid_1];
          }
        }
      }

      if (fid < feat_len)
        out[feat_len * rid + fid] = accum_0;
      if (fid + 32 < feat_len)
        out[feat_len * rid + fid + 32] = accum_1;
    }
  }
}

/*!
 * \brief CUDA implementation of GE-SpMM on Csr format, note that this
 *        function only supports sum reducer.
 * \param bcast Broadcast information.
 * \param csr The Csr matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 */
template <typename Idx, typename DType,
          typename BinaryOp>
void GESpMMCsr(
    const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out) {
  const Idx *indptr = csr.indptr.Ptr<Idx>();
  const Idx *indices = csr.indices.Ptr<Idx>();
  const DType *ufeat_data = ufeat.Ptr<DType>();
  const DType *efeat_data = efeat.Ptr<DType>();
  DType *out_data = out.Ptr<DType>();

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int64_t feat_len = bcast.out_len;
  const int ntx = 32;
  const int nty = 32;
  const int nby = (feat_len + (ntx * 2) - 1) / (ntx * 2);
  const int nbx = (csr.num_rows + nty - 1) / nty;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const int sh_mem_size = 0;

  CUDA_KERNEL_CALL((GESpMMKernel<Idx, DType, BinaryOp>),
      nblks, nthrs, sh_mem_size, thr_entry->stream,
      ufeat_data, efeat_data, out_data,
      indptr, indices,
      csr.num_rows, csr.num_cols, feat_len);
}

}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif
