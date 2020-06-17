/*!
 *  Copyright (c) 2020 by Contributors
 * \file kernel/cuda/spmm.cuh
 * \brief SPMM CUDA kernel function header.
 */
#ifndef DGL_KERNEL_CUDA_SPMM_CUH_
#define DGL_KERNEL_CUDA_SPMM_CUH_

#include "../utils.h"
#include "../bcast.h"
#include "atomic.cuh"
#include "../../runtime/cuda/cuda_common.h"

namespace dgl {
namespace kernel {

namespace cuda {

template <typename T>
__device__ __forceinline__ T _ldg(T* addr) {
#if __CUDA_ARCH__ >= 350
  return __ldg(addr);
#else
  return *addr;
#endif
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCooKernel(
  const DType *ufeat, const DType *efeat, DType *out, Idx *arg_u, Idx *arg_e,
  const Idx *row, const Idx *col, const Idx* edge_map,
  int64_t N, int64_t M, int64_t E,
  int64_t *ubcast_off, int64_t *ebcast_off,
  int64_t ufeat_len, int64_t efeat_len, int64_t out_len) {
  // SPMM with COO.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* uoff = BinaryOp::use_lhs ? (ufeat + src * ufeat_len): nullptr;
    const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len): nullptr;
    DType* outoff = out + dst * out_len;
    while (tx < out_len) {
      const int64_t lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int64_t rhs_add = UseBcast ? ebcast_off[tx] : tx;
      DType val = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
      Idx* arguoff = (ReduceOp::require_arg && BinaryOp::use_lhs) ?
        (arg_u + dst * out_len + tx): nullptr;
      Idx* argeoff = (ReduceOp::require_arg && BinaryOp::use_rhs) ?
        (arg_e + dst * out_len + tx): nullptr;
      ReduceOp::Call(outoff + tx, arguoff, argeoff, val, src, eid);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void ArgSpMMCooKernel(
  const DType *ufeat, const DType *efeat, DType *out, Idx *arg_u, Idx *arg_e,
  const Idx *row, const Idx *col, const Idx* edge_map,
  int64_t N, int64_t M, int64_t E,
  int64_t *ubcast_off, int64_t *ebcast_off,
  int64_t ufeat_len, int64_t efeat_len, int64_t out_len) {
  // SPMM with COO arg max/min.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* uoff = BinaryOp::use_lhs ? (ufeat + src * ufeat_len): nullptr;
    const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len): nullptr;
    const DType* outoff = out + dst * out_len;
    Idx* arguoff = BinaryOp::use_lhs ? (arg_u + dst * out_len): nullptr;
    Idx* argeoff = BinaryOp::use_rhs ? (arg_e + dst * out_len): nullptr;
    while (tx < out_len) {
      int64_t lhs_add = UseBcast ? ubcast_off[tx] : tx;
      int64_t rhs_add = UseBcast ? ebcast_off[tx] : tx;
      DType val = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
      ReduceOp::CallArg(tx, arguoff, argeoff, val, outoff[tx], src, eid);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCsrKernel(
  const DType *ufeat, const DType *efeat, DType *out, Idx *arg_u, Idx *arg_e,
  const Idx *indptr, const Idx *indices, const Idx *edge_map,
  int64_t num_rows, int64_t num_cols, int64_t nnz,
  int64_t *ubcast_off, int64_t *ebcast_off,
  int64_t ufeat_len, int64_t efeat_len, int64_t out_len) {
  // SPMM with CSR.
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  const int stride_x = blockDim.x * gridDim.x;
  while (ty < num_rows) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    while (tx < out_len) {
      DType local_accum = ReduceOp::zero;
      Idx local_argu = 0, local_arge = 0;
      const int lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int rhs_add = UseBcast ? ebcast_off[tx] : tx;
      for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const Idx eid = UseIdx ? _ldg(edge_map + i) : i;
        const Idx cid = _ldg(indices + i);
        const DType* uoff = BinaryOp::use_lhs ? (ufeat + cid * ufeat_len): nullptr;
        const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len): nullptr;
        DType out = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
        ReduceOp::Call(&local_accum, &local_argu, &local_arge, out, cid, eid);
      }
      out[ty * out_len + tx] = local_accum;
      if (ReduceOp::require_arg && BinaryOp::use_lhs)
        arg_u[ty * out_len + tx] = local_argu;
      if (ReduceOp::require_arg && BinaryOp::use_rhs)
        arg_e[ty * out_len + tx] = local_arge;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

#define BCAST_IDX_CTX_SWITCH(BCAST, EDGE_MAP, CTX, LHS_OFF, RHS_OFF, ...) do { \
  const BcastOff &info = (BCAST);                                              \
  if (!info.use_bcast) {                                                       \
    constexpr bool UseBcast = false;                                           \
    if ((EDGE_MAP)) {                                                          \
      constexpr bool UseIdx = true;                                            \
      { __VA_ARGS__ }                                                          \
    } else {                                                                   \
      constexpr bool UseIdx = false;                                           \
      { __VA_ARGS__ }                                                          \
    }                                                                          \
  } else {                                                                     \
    constexpr bool UseBcast = true;                                            \
    DLContext ctx = (CTX);                                                     \
    auto device = runtime::DeviceAPI::Get(ctx);                                \
    (LHS_OFF) = static_cast<int64_t*>(                                         \
      device->AllocWorkspace(ctx, sizeof(int64_t) * info.lhs_offset.size()));  \
    CUDA_CALL(cudaMemcpy((LHS_OFF), &info.lhs_offset[0],                       \
      sizeof(int64_t) * info.lhs_offset.size(), cudaMemcpyHostToDevice));      \
    (RHS_OFF) = static_cast<int64_t*>(                                         \
      device->AllocWorkspace(ctx, sizeof(int64_t) * info.rhs_offset.size()));  \
    CUDA_CALL(cudaMemcpy((RHS_OFF), &info.rhs_offset[0],                       \
      sizeof(int64_t) * info.rhs_offset.size(), cudaMemcpyHostToDevice));      \
    if ((EDGE_MAP)) {                                                          \
      constexpr bool UseIdx = true;                                            \
      { __VA_ARGS__ }                                                          \
    } else {                                                                   \
      constexpr bool UseIdx = false;                                           \
      { __VA_ARGS__ }                                                          \
    }                                                                          \
    device->FreeWorkspace(ctx, (LHS_OFF));                                     \
    device->FreeWorkspace(ctx, (RHS_OFF));                                     \
  }                                                                            \
} while (0)                                                 

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void SpMMCoo(
    const BcastOff& bcast,
    const dgl::aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  Idx *row = utils::GetPtr<Idx>(coo.row),
      *col = utils::GetPtr<Idx>(coo.col),
      *edge_map = utils::GetPtr<Idx>(coo.data);
  DType *ufeat_data = utils::GetPtr<DType>(ufeat),
        *efeat_data = utils::GetPtr<DType>(efeat),
        *out_data = utils::GetPtr<DType>(out);
  Idx *argu_data = utils::GetPtr<Idx>(argu),
      *arge_data = utils::GetPtr<Idx>(arge);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int64_t N = coo.num_rows, M = coo.num_cols, E = coo.row->shape[0];

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len,
          lhs_len = bcast.lhs_len,
          rhs_len = bcast.rhs_len;
  utils::Fill<kDLGPU, DType>(out->ctx, out_data, len * out->shape[0], ReduceOp::zero);

  const int ntx = utils::FindNumThreads(len, 1024);
  const int nty = 1024 / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = utils::FindNumBlocks((E + nty - 1) / nty, 65535);
  //LOG(INFO) << "nblks=(" << nbx << ", " << nby << ") nthrs=(" << ntx << ", " << nty << ")";
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  BCAST_IDX_CTX_SWITCH(bcast, edge_map, ufeat->ctx, ubcast_off, ebcast_off, {
    SpMMCooKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>
      <<<nblks, nthrs, 0, thr_entry->stream>>>(
        ufeat_data, efeat_data, out_data, argu_data, arge_data,
        row, col, edge_map,
        N, M, E,
        ubcast_off, ebcast_off,
        lhs_len, rhs_len, len
      );
    if (ReduceOp::require_arg) {
      ArgSpMMCooKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>
        <<<nblks, nthrs, 0, thr_entry->stream>>>(
          ufeat_data, efeat_data, out_data, argu_data, arge_data,
          row, col, edge_map,
          N, M, E,
          ubcast_off, ebcast_off,
          lhs_len, rhs_len, len
        );
    }
  });
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void SpMMCsr(
    const BcastOff& bcast,
    const dgl::aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  const Idx *indptr = utils::GetPtr<Idx>(csr.indptr);
  const Idx *indices = utils::GetPtr<Idx>(csr.indices);
  const Idx *edge_map = utils::GetPtr<Idx>(csr.data);
  const DType *ufeat_data = utils::GetPtr<DType>(ufeat);
  const DType *efeat_data = utils::GetPtr<DType>(efeat);
  DType *out_data = utils::GetPtr<DType>(out);
  Idx* argu_data = utils::GetPtr<Idx>(argu);
  Idx* arge_data = utils::GetPtr<Idx>(arge);

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len,
          lhs_len = bcast.lhs_len,
          rhs_len = bcast.rhs_len;
  const int ntx = utils::FindNumThreads(len, 1024);
  const int nty = 1024 / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = utils::FindNumBlocks((csr.num_rows + nty - 1) / nty, 65535);
  //LOG(INFO) << "nblks=(" << nbx << ", " << nby << ") nthrs=(" << ntx << ", " << nty << ")";
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  BCAST_IDX_CTX_SWITCH(bcast, edge_map, ufeat->ctx, ubcast_off, ebcast_off, {
    SpMMCsrKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>
      <<<nblks, nthrs, 0, thr_entry->stream>>>(
        ufeat_data, efeat_data, out_data, argu_data, arge_data,
        indptr, indices, edge_map,
        csr.num_rows, csr.num_cols, efeat->shape[0],
        ubcast_off, ebcast_off,
        lhs_len, rhs_len, len
      );
  });
}


}  // namespace cuda
}  // namespace kernel
}  // namespace dgl

#endif
