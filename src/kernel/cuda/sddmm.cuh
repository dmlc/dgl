/*!
 *  Copyright (c) 2020 by Contributors
 * \file kernel/cuda/sddmm.cuh
 * \brief SDDMM CUDA kernel function header.
 */
#ifndef DGL_KERNEL_CUDA_SDDMM_CUH_
#define DGL_KERNEL_CUDA_SDDMM_CUH_

#include "../utils.h"
#include "../binary_reduce_impl_decl.h"
#include "../bcast.h"
#include "atomic.cuh"
#include "functor2.cuh"
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

template <typename Idx, typename DType, typename BinaryOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void SDDMMCooKernel(
  const DType *ufeat, const DType *vfeat, DType *out,
  const Idx *row, const Idx *col, const Idx* edge_map,
  int64_t N, int64_t M, int64_t E, int64_t reduce_size,
  const int64_t *ubcast_off, const int64_t *vbcast_off,
  int64_t ufeat_len, int64_t vfeat_len, int64_t out_len) {
  // SDDMM with COO.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    const DType* lhsoff = BinaryOp::use_lhs ?
      (ufeat + src * ufeat_len): nullptr;
    const DType* rhsoff = BinaryOp::use_rhs ?
      (vfeat + dst * vfeat_len): nullptr;
    DType* outoff = out + eid * out_len;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride_x = blockDim.x * gridDim.x;
    while (tx < out_len) {
      const Idx lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const Idx rhs_add = UseBcast ? vbcast_off[tx] : tx;
      DType val = BinaryOp::Call(
          lhsoff + lhs_add * reduce_size,
          rhsoff + rhs_add * reduce_size,
          reduce_size);
      outoff[tx] = val;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

// Binary search the row_offsets to find the source node of the edge id.
template <typename Idx>
__device__ __forceinline__ Idx BinarySearchSrc(const Idx *array, Idx length, Idx eid) {
  Idx lo = 0, hi = length - 1;
  while (lo < hi) {
    Idx mid = (lo + hi) >> 1;
    if (_ldg(array + mid) <= eid) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  // INVARIANT: lo == hi
  if (_ldg(array + hi) == eid) {
    return hi;
  } else {
    return hi - 1;
  }
}

template <typename Idx, typename DType, typename BinaryOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void SDDMMCsrKernel(
  const DType *ufeat, const DType *vfeat, DType *out,
  const Idx *indptr, const Idx *indices, const Idx* edge_map,
  int64_t N, int64_t M, int64_t E, int64_t reduce_size,
  int64_t *ubcast_off, int64_t *vbcast_off,
  int64_t ufeat_len, int64_t vfeat_len, int64_t out_len) {
  // SDDMM with Csr.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = BinarySearchSrc<Idx>(indptr, N + 1, ty);
    const Idx dst = _ldg(indices + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* lhsoff = BinaryOp::use_lhs ? (ufeat + src * ufeat_len): nullptr;
    const DType* rhsoff = BinaryOp::use_rhs ? (vfeat + dst * vfeat_len): nullptr;
    DType* outoff = out + eid * out_len;
    while (tx < out_len) {
      const Idx lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const Idx rhs_add = UseBcast ? vbcast_off[tx] : tx;
      DType val = BinaryOp::Call(
          lhsoff + lhs_add * reduce_size,
          rhsoff + rhs_add * reduce_size,
          reduce_size);
      outoff[tx] = val;
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

template <typename Idx, typename DType, typename Op>
void SDDMMCoo(
    const BcastOff& bcast,
    const dgl::aten::COOMatrix& coo,
    NDArray ufeat,
    NDArray vfeat,
    NDArray out) {
  const Idx *row = utils::GetPtr<Idx>(coo.row);
  const Idx *col = utils::GetPtr<Idx>(coo.col);
  const Idx *edge_map = utils::GetPtr<Idx>(coo.data);
  const DType *ufeat_data = utils::GetPtr<DType>(ufeat);
  const DType *vfeat_data = utils::GetPtr<DType>(vfeat);
  DType *out_data = utils::GetPtr<DType>(out);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  int64_t *ubcast_off = nullptr, *vbcast_off = nullptr;
  int64_t len = bcast.out_len,
          lhs_len = bcast.lhs_len,
          rhs_len = bcast.rhs_len;
  int64_t reduce_dim = bcast.reduce_size;

  const int64_t nnz = coo.row->shape[0];
  const int ntx = utils::FindNumThreads(len, 1024);
  const int nty = 1024 / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = utils::FindNumBlocks((nnz + nty - 1) / nty, 65535);
  //LOG(INFO) << "nblks=(" << nbx << ", " << nby << ") nthrs=(" << ntx << ", " << nty << ")";
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  BCAST_IDX_CTX_SWITCH(bcast, edge_map, ufeat->ctx, ubcast_off, vbcast_off, {
    SDDMMCooKernel<Idx, DType, Op, UseBcast, UseIdx>
      <<<nblks, nthrs, 0, thr_entry->stream>>>(
        ufeat_data, vfeat_data, out_data,
        row, col, edge_map,
        coo.num_rows, coo.num_cols, nnz, reduce_dim,
        ubcast_off, vbcast_off,
        lhs_len, rhs_len, len
      );
  });
}

template <typename Idx, typename DType, typename Op>
void SDDMMCsr(
    const BcastOff& bcast,
    const dgl::aten::CSRMatrix& csr,
    NDArray ufeat,
    NDArray vfeat,
    NDArray out) {
  const Idx *indptr = utils::GetPtr<Idx>(csr.indptr);
  const Idx *indices = utils::GetPtr<Idx>(csr.indices);
  const Idx *edge_map = utils::GetPtr<Idx>(csr.data);
  const DType *ufeat_data = utils::GetPtr<DType>(ufeat);
  const DType *vfeat_data = utils::GetPtr<DType>(vfeat);
  DType *out_data = utils::GetPtr<DType>(out);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int64_t N = csr.num_rows, M = csr.num_cols, E = csr.indices->shape[0];

  int64_t *ubcast_off = nullptr, *vbcast_off = nullptr;
  int64_t len = bcast.out_len,
          lhs_len = bcast.lhs_len,
          rhs_len = bcast.rhs_len;
  int64_t reduce_dim = bcast.reduce_size;

  const int ntx = utils::FindNumThreads(len, 1024);
  const int nty = 1024 / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = utils::FindNumBlocks((E + nty - 1) / nty, 65535);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  BCAST_IDX_CTX_SWITCH(bcast, edge_map, ufeat->ctx, ubcast_off, vbcast_off, {
    SDDMMCsrKernel<Idx, DType, Op, UseBcast, UseIdx>
      <<<nblks, nthrs, 0, thr_entry->stream>>>(
        ufeat_data, vfeat_data, out_data,
        indptr, indices, edge_map,
        N, M, E, reduce_dim,
        ubcast_off, vbcast_off,
        lhs_len, rhs_len, len
      );
  });
}


}
}
}

#endif
