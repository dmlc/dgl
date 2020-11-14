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
template <typename Idx, typename DType, typename BinaryOp,
          bool UseBcast = false,
          bool UseIdx = false,
          bool IsScalar = false>
__global__ void GESpMMKernel(
    const DType* __restrict__ ufeat,
    const DType* __restrict__ efeat,
    DType* __restrict__ out,
    const Idx* __restrict__ indptr,
    const Idx* __restrict__ indices,
    const Idx* __restrict__ edge_map,
    const int64_t num_rows, const int64_t num_cols,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off,
    const int64_t ufeat_len,
    const int64_t efeat_len,
    const int64_t feat_len) {
  extern __shared__ char smem[];
  DType *weight = (DType *)smem;
  const Idx ty = threadIdx.y, tx = threadIdx.x;
  const Idx rid = blockIdx.x * blockDim.y + ty;  // over vertices dimension
  const Idx fid = (blockIdx.y * 64) + tx;        // over feature dimension

  if (rid < num_rows) {
    const Idx u_fid_0 = UseBcast ? (fid < feat_len ? __ldg(ubcast_off + fid) : 0) : fid;
    const Idx u_fid_1 = UseBcast ? (fid + 32 < feat_len ? __ldg(ubcast_off + fid + 32) : 0) : fid + 32;
    const Idx e_fid_0 = IsScalar ? 0 : (UseBcast ? (fid < feat_len ? __ldg(ebcast_off + fid) : 0) : fid);
    const Idx e_fid_1 = IsScalar ? 0 : (UseBcast ? (fid + 32 < feat_len ? __ldg(ebcast_off + fid + 32) : 0) : fid + 32);
    const Idx low = __ldg(indptr + rid), high = __ldg(indptr + rid + 1);
    DType accum_0 = 0., accum_1 = 0.;

    for (Idx left = low; left < high; left += 32) {
      // copy edge weight to shared memory
      for (Idx i = 0; i < efeat_len; ++i) {
        const Idx eid = left + tx;
        if (eid < high) {
          const Idx real_eid = UseIdx ? __ldg(edge_map + eid) : eid;
          weight[ty * (32 * efeat_len) + tx * efeat_len + i] =\
            efeat[real_eid * efeat_len + i];
        }
      }
      // NOTE(zihao): no need to sync thread here.

      if (left + 32 <= high) {
#pragma unroll
        for (Idx i = 0; i < 32; ++i) {
          const Idx eid = left + i;
          const Idx cid = __ldg(indices + eid);
          const Idx offset_u = ufeat_len * cid;
          const Idx offset_e = ty * 32 * efeat_len + i * efeat_len;
          if (BinaryOp::use_rhs) {
            accum_0 += BinaryOp::Call(ufeat + offset_u + u_fid_0,
                                      weight + offset_e + e_fid_0);
            accum_1 += BinaryOp::Call(ufeat + offset_u + u_fid_1,
                                      weight + offset_e + e_fid_1);
          } else {
            accum_0 += ufeat[offset_u + u_fid_0];
            accum_1 += ufeat[offset_u + u_fid_1];
          }
        }
      } else {
        for (Idx i = 0; left + i < high; ++i) {
          const Idx eid = left + i;
          const Idx cid = __ldg(indices + eid);
          const Idx offset_u = ufeat_len * cid;
          const Idx offset_e = ty * 32 * efeat_len + i * efeat_len;
          if (BinaryOp::use_rhs) {
            accum_0 += BinaryOp::Call(ufeat + offset_u + u_fid_0,
                                      weight + offset_e + e_fid_0);
            accum_1 += BinaryOp::Call(ufeat + offset_u + u_fid_1,
                                      weight + offset_e + e_fid_1);
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

#define SWITCH_USE_IDX(use_idx, UseIdx, ...)         \
  do {                                               \
    if ((use_idx)) {                                 \
      constexpr bool UseIdx = true;                  \
      { __VA_ARGS__ }                                \
    } else {                                         \
      constexpr bool UseIdx = false;                 \
      { __VA_ARGS__}                                 \
    }                                                \
  } while (0)

#define SWITCH_USE_BCAST(use_bcast, UseBcast, ...)   \
  do {                                               \
    if ((use_bcast)) {                               \
      constexpr bool UseBcast = true;                \
      { __VA_ARGS__ }                                \
    } else {                                         \
      constexpr bool UseBcast = false;               \
      { __VA_ARGS__ }                                \
    }                                                \
  } while (0)

#define SWITCH_IS_SCALAR(is_scalar, IsScalar, ...)   \
  do {                                               \
    if ((is_scalar)) {                               \
      constexpr bool IsScalar = true;                \
      { __VA_ARGS__ }                                \
    } else {                                         \
      constexpr bool IsScalar = false;               \
      { __VA_ARGS__ }                                \
    }                                                \
  } while (0)

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
  const Idx *edge_map = csr.data.Ptr<Idx>();
  const DType *ufeat_data = ufeat.Ptr<DType>();
  const DType *efeat_data = efeat.Ptr<DType>();
  DType *out_data = out.Ptr<DType>();

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t feat_len = bcast.out_len,
          lhs_len = bcast.lhs_len,
          rhs_len = bcast.rhs_len;

  const int ntx = 32;
  const int nty = std::min<int>(32, 128 / (rhs_len * sizeof(DType)));
  const int nby = (feat_len + (ntx * 2) - 1) / (ntx * 2);
  const int nbx = (csr.num_rows + nty - 1) / nty;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const int sh_mem_size = ntx * nty * rhs_len * sizeof(DType);
  const bool use_idx = !IsNullArray(csr.data),
             use_bcast = bcast.use_bcast,
             is_scalar = rhs_len == 1;

//  LOG(INFO) << ntx << " " << nty << " " << sh_mem_size;

  if (use_bcast) {
    const DLContext ctx = ufeat->ctx;
    const auto device = runtime::DeviceAPI::Get(ctx);
    ubcast_off = static_cast<int64_t *>(
        device->AllocWorkspace(ctx, sizeof(int64_t) * bcast.lhs_offset.size()));
    CUDA_CALL(cudaMemcpy(ubcast_off, &bcast.lhs_offset[0],
                         sizeof(int64_t) * bcast.lhs_offset.size(),
                         cudaMemcpyHostToDevice));
    ebcast_off = static_cast<int64_t *>(
        device->AllocWorkspace(ctx, sizeof(int64_t) * bcast.rhs_offset.size()));
    CUDA_CALL(cudaMemcpy(ebcast_off, &bcast.rhs_offset[0],
                         sizeof(int64_t) * bcast.rhs_offset.size(),
                         cudaMemcpyHostToDevice));
  }

  SWITCH_USE_BCAST(use_bcast, UseBcast, {
    SWITCH_USE_IDX(use_idx, UseIdx, {
      SWITCH_IS_SCALAR(is_scalar, IsScalar, {
        CUDA_KERNEL_CALL((GESpMMKernel<Idx, DType, BinaryOp, UseBcast, UseIdx, IsScalar>),
            nblks, nthrs, sh_mem_size, thr_entry->stream,
            ufeat_data, efeat_data, out_data,
            indptr, indices, edge_map,
            csr.num_rows, csr.num_cols,
            ubcast_off, ebcast_off,
            lhs_len, rhs_len, feat_len);
      });
    });
  });
}

}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif
