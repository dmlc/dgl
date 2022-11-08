/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/sddmm.cuh
 * @brief SDDMM CUDA kernel function header.
 */
#ifndef DGL_ARRAY_CUDA_SDDMM_CUH_
#define DGL_ARRAY_CUDA_SDDMM_CUH_

#include <dgl/bcast.h>

#include "../../runtime/cuda/cuda_common.h"
#include "../selector.h"
#include "./functor.cuh"
#include "./utils.h"
#include "atomic.cuh"
#include "bf16.cuh"
#include "fp16.cuh"
#include "functor.cuh"
#include "macro.cuh"

namespace dgl {

using namespace cuda;

namespace aten {
namespace cuda {

#define SWITCH_OP(op, Op, ...)                                        \
  do {                                                                \
    if ((op) == "add") {                                              \
      typedef cuda::binary::Add<DType> Op;                            \
      { __VA_ARGS__ }                                                 \
    } else if ((op) == "sub") {                                       \
      typedef cuda::binary::Sub<DType> Op;                            \
      { __VA_ARGS__ }                                                 \
    } else if ((op) == "mul") {                                       \
      typedef cuda::binary::Mul<DType> Op;                            \
      { __VA_ARGS__ }                                                 \
    } else if ((op) == "div") {                                       \
      typedef cuda::binary::Div<DType> Op;                            \
      { __VA_ARGS__ }                                                 \
    } else if ((op) == "copy_lhs") {                                  \
      typedef cuda::binary::CopyLhs<DType> Op;                        \
      { __VA_ARGS__ }                                                 \
    } else if ((op) == "copy_rhs") {                                  \
      typedef cuda::binary::CopyRhs<DType> Op;                        \
      { __VA_ARGS__ }                                                 \
    } else if ((op) == "dot") {                                       \
      typedef cuda::binary::Dot<DType> Op;                            \
      { __VA_ARGS__ }                                                 \
    } else {                                                          \
      LOG(FATAL) << "Unsupported SpMM/SDDMM binary operator: " << op; \
    }                                                                 \
  } while (0)

#define SWITCH_RHS(rhs_target, RhsTarget, ...)             \
  do {                                                     \
    if ((rhs_target) == 0) {                               \
      constexpr int RhsTarget = 0;                         \
      { __VA_ARGS__ }                                      \
    } else if ((rhs_target) == 1) {                        \
      constexpr int RhsTarget = 1;                         \
      { __VA_ARGS__ }                                      \
    } else if ((rhs_target) == 2) {                        \
      constexpr int RhsTarget = 2;                         \
      { __VA_ARGS__ }                                      \
    } else {                                               \
      LOG(INFO) << "Invalid rhs target: " << (rhs_target); \
    }                                                      \
  } while (0)

#define SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, ...) \
  do {                                                                   \
    if ((lhs_target) == 0) {                                             \
      constexpr int LhsTarget = 0;                                       \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                    \
    } else if ((lhs_target) == 1) {                                      \
      constexpr int LhsTarget = 1;                                       \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                    \
    } else if ((lhs_target) == 2) {                                      \
      constexpr int LhsTarget = 2;                                       \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                    \
    } else {                                                             \
      LOG(INFO) << "Invalid lhs target: " << (lhs_target);               \
    }                                                                    \
  } while (0)

constexpr unsigned int full_mask = 0xffffffff;

/**
 * @brief CUDA kernel of g-SDDMM on Coo format.
 * @note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 * positions in feature dimension.
 */
template <
    typename Idx, typename DType, typename BinaryOp, bool UseBcast = false,
    bool UseIdx = false, int LhsTarget = 0, int RhsTarget = 2>
__global__ void SDDMMCooKernel(
    const DType* __restrict__ lhs, const DType* __restrict__ rhs,
    DType* __restrict__ out, const Idx* __restrict__ row,
    const Idx* __restrict__ col, const Idx* __restrict__ edge_map, int64_t N,
    int64_t M, int64_t E, int64_t reduce_size,
    const int64_t* __restrict__ lhs_off, const int64_t* __restrict__ rhs_off,
    int64_t lhs_len, int64_t rhs_len, int64_t out_len) {
  // SDDMM with COO.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    const DType* lhsoff =
        BinaryOp::use_lhs
            ? (lhs + Selector<LhsTarget>::Call(src, eid, dst) * lhs_len)
            : nullptr;
    const DType* rhsoff =
        BinaryOp::use_rhs
            ? (rhs + Selector<RhsTarget>::Call(src, eid, dst) * rhs_len)
            : nullptr;
    DType* outoff = out + eid * out_len;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride_x = blockDim.x * gridDim.x;
    while (tx < out_len) {
      const Idx lhs_add = UseBcast ? lhs_off[tx] : tx;
      const Idx rhs_add = UseBcast ? rhs_off[tx] : tx;
      DType val = BinaryOp::Call(
          lhsoff + lhs_add * reduce_size, rhsoff + rhs_add * reduce_size,
          reduce_size);
      outoff[tx] = val;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA kernel of SDDMM-dot on Coo format, accelerated with tree
 * reduction.
 * @note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 * positions in feature dimension.
 */
template <
    typename Idx, typename DType, bool UseBcast = false, bool UseIdx = false,
    int LhsTarget = 0, int RhsTarget = 2>
__global__ void SDDMMCooTreeReduceKernel(
    const DType* __restrict__ lhs, const DType* __restrict__ rhs,
    DType* __restrict__ out, const Idx* __restrict__ row,
    const Idx* __restrict__ col, const Idx* __restrict__ edge_map, int64_t N,
    int64_t M, int64_t E, int64_t reduce_size,
    const int64_t* __restrict__ lhs_off, const int64_t* __restrict__ rhs_off,
    int64_t lhs_len, int64_t rhs_len, int64_t out_len) {
  Idx ty = blockIdx.x * blockDim.y + threadIdx.y;
  if (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    const DType* lhsoff =
        lhs + Selector<LhsTarget>::Call(src, eid, dst) * lhs_len;
    const DType* rhsoff =
        rhs + Selector<RhsTarget>::Call(src, eid, dst) * rhs_len;
    DType* outoff = out + eid * out_len;
    int tx = threadIdx.x;  // tx < 32
    for (int i = blockIdx.y; i < out_len;
         i += gridDim.y) {  // over output feature dimension
      const Idx lhs_add = UseBcast ? __ldg(lhs_off + i) : i;
      const Idx rhs_add = UseBcast ? __ldg(rhs_off + i) : i;
      DType val = reduce::Sum<Idx, DType>::zero();
      for (int j = tx; j < reduce_size; j += 64) {
        val += lhsoff[lhs_add * reduce_size + j] *
               rhsoff[rhs_add * reduce_size + j];
        if (j + 32 < reduce_size)
          val += lhsoff[lhs_add * reduce_size + j + 32] *
                 rhsoff[rhs_add * reduce_size + j + 32];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(full_mask, val, offset);
      if (tx == 0) outoff[i] = val;
    }
  }
}

// Binary search the row_offsets to find the source node of the edge id.
template <typename Idx>
__device__ __forceinline__ Idx
BinarySearchSrc(const Idx* array, Idx length, Idx eid) {
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

/**
 * @brief CUDA kernel of g-SDDMM on Csr format.
 * @note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 * positions in feature dimension. To efficiently find the source node idx and
 * destination node index of an given edge on Csr format, it uses binary search
 * (time complexity O(log N)).
 */
template <
    typename Idx, typename DType, typename BinaryOp, bool UseBcast = false,
    bool UseIdx = false, int LhsTarget = 0, int RhsTarget = 2>
__global__ void SDDMMCsrKernel(
    const DType* __restrict__ lhs, const DType* __restrict__ rhs,
    DType* __restrict__ out, const Idx* __restrict__ indptr,
    const Idx* __restrict__ indices, const Idx* __restrict__ edge_map,
    int64_t N, int64_t M, int64_t E, int64_t reduce_size,
    const int64_t* __restrict__ lhs_off, const int64_t* __restrict__ rhs_off,
    int64_t lhs_len, int64_t rhs_len, int64_t out_len) {
  // SDDMM with Csr.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = BinarySearchSrc<Idx>(indptr, N + 1, ty);
    const Idx dst = _ldg(indices + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* lhsoff =
        BinaryOp::use_lhs
            ? (lhs + Selector<LhsTarget>::Call(src, eid, dst) * lhs_len)
            : nullptr;
    const DType* rhsoff =
        BinaryOp::use_rhs
            ? (rhs + Selector<RhsTarget>::Call(src, eid, dst) * rhs_len)
            : nullptr;
    DType* outoff = out + eid * out_len;
    while (tx < out_len) {
      const Idx lhs_add = UseBcast ? lhs_off[tx] : tx;
      const Idx rhs_add = UseBcast ? rhs_off[tx] : tx;
      DType val = BinaryOp::Call(
          lhsoff + lhs_add * reduce_size, rhsoff + rhs_add * reduce_size,
          reduce_size);
      outoff[tx] = val;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA implementation of g-SDDMM on Coo format.
 * @param bcast Broadcast information.
 * @param coo The Coo matrix.
 * @param lhs The left hand side operand feature.
 * @param rhs The right hand size operand feature.
 * @param out The result feature on edges.
 */
template <
    typename Idx, typename DType, typename Op, int LhsTarget = 0,
    int RhsTarget = 2>
void SDDMMCoo(
    const BcastOff& bcast, const COOMatrix& coo, NDArray lhs, NDArray rhs,
    NDArray out) {
  const Idx* row = coo.row.Ptr<Idx>();
  const Idx* col = coo.col.Ptr<Idx>();
  const Idx* edge_map = coo.data.Ptr<Idx>();
  const DType* lhs_data = lhs.Ptr<DType>();
  const DType* rhs_data = rhs.Ptr<DType>();
  DType* out_data = out.Ptr<DType>();
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  int64_t *lhs_off = nullptr, *rhs_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;
  int64_t reduce_dim = bcast.reduce_size;

  const int64_t nnz = coo.row->shape[0];
  const bool use_idx = !IsNullArray(coo.data);

  if (std::is_same<Op, binary::Dot<DType> >::value && reduce_dim >= 32) {
    const int ntx = 32;  // on feature dimension
    const int nty = 8;   // on out dimension
    const int nbx = (nnz + nty - 1) / nty;
    const int nby = FindNumBlocks<'y'>(len);
    const dim3 nblks(nbx, nby);
    const dim3 nthrs(ntx, nty);
    BCAST_IDX_CTX_SWITCH(bcast, use_idx, out->ctx, lhs_off, rhs_off, {
      CUDA_KERNEL_CALL(
          (SDDMMCooTreeReduceKernel<
              Idx, DType, UseBcast, UseIdx, LhsTarget, RhsTarget>),
          nblks, nthrs, 0, stream, lhs_data, rhs_data, out_data, row, col,
          edge_map, coo.num_rows, coo.num_cols, nnz, reduce_dim, lhs_off,
          rhs_off, lhs_len, rhs_len, len);
    });
  } else {
    const int ntx = FindNumThreads(len);
    const int nty = CUDA_MAX_NUM_THREADS / ntx;
    const int nbx = (len + ntx - 1) / ntx;
    const int nby = FindNumBlocks<'y'>((nnz + nty - 1) / nty);
    const dim3 nblks(nbx, nby);
    const dim3 nthrs(ntx, nty);
    BCAST_IDX_CTX_SWITCH(bcast, use_idx, out->ctx, lhs_off, rhs_off, {
      CUDA_KERNEL_CALL(
          (SDDMMCooKernel<
              Idx, DType, Op, UseBcast, UseIdx, LhsTarget, RhsTarget>),
          nblks, nthrs, 0, stream, lhs_data, rhs_data, out_data, row, col,
          edge_map, coo.num_rows, coo.num_cols, nnz, reduce_dim, lhs_off,
          rhs_off, lhs_len, rhs_len, len);
    });
  }
}

/**
 * @brief CUDA implementation of g-SDDMM on Csr format.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param lhs The left hand side operand feature.
 * @param rhs The right hand size operand feature.
 * @param out The result feature on edges.
 */
template <
    typename Idx, typename DType, typename Op, int LhsTarget = 0,
    int RhsTarget = 2>
void SDDMMCsr(
    const BcastOff& bcast, const CSRMatrix& csr, NDArray lhs, NDArray rhs,
    NDArray out) {
  const Idx* indptr = csr.indptr.Ptr<Idx>();
  const Idx* indices = csr.indices.Ptr<Idx>();
  const Idx* edge_map = csr.data.Ptr<Idx>();
  const DType* lhs_data = lhs.Ptr<DType>();
  const DType* rhs_data = rhs.Ptr<DType>();
  DType* out_data = out.Ptr<DType>();
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  int64_t N = csr.num_rows, M = csr.num_cols, E = csr.indices->shape[0];

  int64_t *lhs_off = nullptr, *rhs_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;
  int64_t reduce_dim = bcast.reduce_size;

  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((E + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(csr.data);

  BCAST_IDX_CTX_SWITCH(bcast, use_idx, out->ctx, lhs_off, rhs_off, {
    CUDA_KERNEL_CALL(
        (SDDMMCsrKernel<
            Idx, DType, Op, UseBcast, UseIdx, LhsTarget, RhsTarget>),
        nblks, nthrs, 0, stream, lhs_data, rhs_data, out_data, indptr, indices,
        edge_map, N, M, E, reduce_dim, lhs_off, rhs_off, lhs_len, rhs_len, len);
  });
}

}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CUDA_SDDMM_CUH_
