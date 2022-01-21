/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/spmm.cuh
 * \brief SPMM CUDA kernel function header.
 */
#ifndef DGL_ARRAY_CUDA_SPMM_CUH_
#define DGL_ARRAY_CUDA_SPMM_CUH_

#include <dgl/bcast.h>
#include "macro.cuh"
#include "fp16.cuh"
#include "atomic.cuh"
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {

using namespace cuda;

namespace aten {
namespace cuda {


/*!
 * \brief CUDA kernel of g-SpMM on Coo format.
 * \note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different positions
 *       in feature dimension.
 *       To avoid possible data hazards, it uses atomic operators for reduction.
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCooKernel(
  const DType* __restrict__ ufeat,
  const DType* __restrict__ efeat,
  DType* __restrict__ out,
  Idx* __restrict__ arg_u,
  Idx* __restrict__ arg_e,
  const Idx* __restrict__ row,
  const Idx* __restrict__ col,
  const Idx* __restrict__ edge_map,
  int64_t N, int64_t M, int64_t E,
  const int64_t* __restrict__ ubcast_off,
  const int64_t* __restrict__ ebcast_off,
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
      Idx* arguoff = nullptr;  // arguoff is not used in SpMMCoo.
      Idx* argeoff = nullptr;  // argeoff is not used in SpMMCoo.
      ReduceOp::Call(outoff + tx, arguoff, argeoff, val, src, eid);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/*!
 * \brief CUDA kernel to compute argu and arge in g-SpMM on Coo format.
 * \note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different positions
 *       in feature dimension.
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void ArgSpMMCooKernel(
  const DType* __restrict__ ufeat,
  const DType* __restrict__ efeat,
  DType* __restrict__ out,
  Idx* __restrict__ arg_u,
  Idx* __restrict__ arg_e,
  const Idx* __restrict__ row,
  const Idx* __restrict__ col,
  const Idx* __restrict__ edge_map,
  int64_t N, int64_t M, int64_t E,
  const int64_t* __restrict__ ubcast_off,
  const int64_t* __restrict__ ebcast_off,
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

/*!
 * \brief CUDA kernel of g-SpMM on Csr format.
 * \note it uses node parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different destination nodes.
 *       Threadblocks on the x-axis are responsible for the computation on
 *       different positions in feature dimension.
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCsrKernel(
  const DType* __restrict__ ufeat,
  const DType* __restrict__ efeat,
  DType* __restrict__ out,
  Idx* __restrict__ arg_u,
  Idx* __restrict__ arg_e,
  const Idx* __restrict__ indptr,
  const Idx* __restrict__ indices,
  const Idx* __restrict__ edge_map,
  int64_t num_rows, int64_t num_cols,
  const int64_t* __restrict__ ubcast_off,
  const int64_t* __restrict__ ebcast_off,
  int64_t ufeat_len, int64_t efeat_len, int64_t out_len) {
  // SPMM with CSR.
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  const int stride_x = blockDim.x * gridDim.x;
  while (ty < num_rows) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    while (tx < out_len) {
      DType local_accum = ReduceOp::zero();
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
      // The use of += is to compute cross-type reducing on heterogeneous graph
      // when reduce op is `sum`.
      //     C = SpMM(SpA, B) + C
      // Separate kernel `SpMMCmpCsrHeteroKernel` is used for max- and min-reducer. It
      // does not affect the output on homogeneous graph as `out` is initialized to zero.
      out[ty * out_len + tx] += local_accum;
      if (ReduceOp::require_arg && BinaryOp::use_lhs)
        arg_u[ty * out_len + tx] = local_argu;
      if (ReduceOp::require_arg && BinaryOp::use_rhs)
        arg_e[ty * out_len + tx] = local_arge;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/*!
 * \brief CUDA kernel of edge_softmax backward Kernel on Csr format.
 * \note it uses node parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different destination nodes.
 *       Threadblocks on the x-axis are responsible for the computation on
 *       different positions in feature dimension.
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void Edge_backwardKernel(
  const DType* __restrict__ f_out_data,
  const DType* __restrict__ sds_data,
  DType* __restrict__ back_out_data,
  const Idx* __restrict__ indptr,
  const Idx* __restrict__ indices,
  const Idx* __restrict__ edge_map,
  int64_t num_rows, int64_t num_cols,
  const int64_t* __restrict__ ubcast_off,
  const int64_t* __restrict__ ebcast_off,
  int64_t f_out_data_len, int64_t sds_data_len, int64_t back_out_data_len) {
  // Edge softmax backward with CSR.
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  const int stride_x = blockDim.x * gridDim.x;
  while (ty < num_rows) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    while (tx < back_out_data_len) {
      DType local_accum = ReduceOp::zero();
      Idx local_argu = 0, local_arge = 0;
      const int lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int rhs_add = UseBcast ? ebcast_off[tx] : tx;
      for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const Idx eid = UseIdx ? _ldg(edge_map + i) : i;
        const Idx cid = _ldg(indices + i);
        const DType* sds_off = BinaryOp::use_rhs ? (sds_data + eid * sds_data_len): nullptr;
        DType out = BinaryOp::Call(sds_off + rhs_add, sds_off + rhs_add);
        ReduceOp::Call(&local_accum, &local_argu, &local_arge, out, cid, eid);
      }
      for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const Idx eid = UseIdx ? _ldg(edge_map + i) : i;
        Idx tmp = eid * sds_data_len;
        const DType* f_out_off = BinaryOp::use_rhs ? (f_out_data + tmp + rhs_add): nullptr;
        const DType* sds_off = BinaryOp::use_rhs ? (sds_data + tmp): nullptr;
        back_out_data[tmp + rhs_add] = (*(sds_off+ rhs_add)) - local_accum * (*(f_out_off));
      }
      tx += stride_x;
    }
    ty += stride_y;
  }
}
/*!
 * \brief CUDA kernel of SpMM-Min/Max on Csr format.
 * \note it uses node parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different destination nodes.
 *       Threadblocks on the x-axis are responsible for the computation on
 *       different positions in feature dimension.
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCmpCsrHeteroKernel(
  const DType* __restrict__ ufeat,
  const DType* __restrict__ efeat,
  DType* __restrict__ out,
  Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
  Idx* __restrict__ arg_u_ntype, Idx* __restrict__ arg_e_etype,
  const Idx* __restrict__ indptr,
  const Idx* __restrict__ indices,
  const Idx* __restrict__ edge_map,
  int64_t num_rows, int64_t num_cols,
  const int64_t* __restrict__ ubcast_off,
  const int64_t* __restrict__ ebcast_off,
  int64_t ufeat_len, int64_t efeat_len, int64_t out_len,
  const int src_type, const int etype) {
  // SPMM with CSR.
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  const int stride_x = blockDim.x * gridDim.x;
  while (ty < num_rows) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    while (tx < out_len) {
      DType new_out = out[ty * out_len + tx];//ReduceOp::zero();
      Idx local_argu = 0, local_arge = 0;
      const int lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int rhs_add = UseBcast ? ebcast_off[tx] : tx;
      for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const Idx eid = UseIdx ? _ldg(edge_map + i) : i;
        const Idx cid = _ldg(indices + i);
        const DType* uoff = BinaryOp::use_lhs ? (ufeat + cid * ufeat_len): nullptr;
        const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len): nullptr;
        DType tmp_out = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
        ReduceOp::Call(&new_out, &local_argu, &local_arge, tmp_out, cid, eid);
      }
      // Update output only when max/min values are different that original output
      if (out[ty * out_len + tx] != new_out) {
        out[ty * out_len + tx] = new_out;
        if (ReduceOp::require_arg && BinaryOp::use_lhs) {
          arg_u[ty * out_len + tx] = local_argu;
          arg_u_ntype[ty * out_len + tx] = src_type;
        }
        if (ReduceOp::require_arg && BinaryOp::use_rhs) {
          arg_e[ty * out_len + tx] = local_arge;
          arg_e_etype[ty * out_len + tx] = etype;
        }
      }
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/*!
 * \brief CUDA implementation of g-SpMM on Coo format.
 * \param bcast Broadcast information.
 * \param coo The Coo matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 * \param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void SpMMCoo(
    const BcastOff& bcast,
    const COOMatrix& coo,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
#if defined(CUDART_VERSION) && CUDART_VERSION <= 10000
  if (std::is_same<DType, half>::value)
    LOG(FATAL) << "SpMMCoo requires atomicCAS, which is not supported "
               << "for float16 in CUDA 10.0. Please upgrade your CUDA "
               << "to later versions.";
#endif
  const Idx *row = coo.row.Ptr<Idx>(),
            *col = coo.col.Ptr<Idx>(),
            *edge_map = coo.data.Ptr<Idx>();
  const DType *ufeat_data = ufeat.Ptr<DType>(),
              *efeat_data = efeat.Ptr<DType>();
  DType *out_data = out.Ptr<DType>();
  Idx *argu_data = argu.Ptr<Idx>(),
      *arge_data = arge.Ptr<Idx>();
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const int64_t N = coo.num_rows, M = coo.num_cols, E = coo.row->shape[0];

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len,
          lhs_len = bcast.lhs_len,
          rhs_len = bcast.rhs_len;

  int64_t out_size = out.NumElements();
  const int nt = FindNumThreads(out_size);
  const int nb = (out_size + nt - 1) / nt;
  CUDA_KERNEL_CALL(_FillKernel, nb, nt, 0, thr_entry->stream,
      out_data, out_size, ReduceOp::zero());

  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((E + nty - 1) / nty);
  // LOG(INFO) << "nblks=(" << nbx << ", " << nby << ") nthrs=(" << ntx << ", " << nty << ")";
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(coo.data);

  BCAST_IDX_CTX_SWITCH(bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off, {
    CUDA_KERNEL_CALL((SpMMCooKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
        nblks, nthrs, 0, thr_entry->stream,
        ufeat_data, efeat_data, out_data, argu_data, arge_data,
        row, col, edge_map,
        N, M, E,
        ubcast_off, ebcast_off,
        lhs_len, rhs_len, len);
    if (ReduceOp::require_arg) {
      CUDA_KERNEL_CALL((ArgSpMMCooKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
          nblks, nthrs, 0, thr_entry->stream,
          ufeat_data, efeat_data, out_data, argu_data, arge_data,
          row, col, edge_map,
          N, M, E,
          ubcast_off, ebcast_off,
          lhs_len, rhs_len, len);
    }
  });
}

/*!
 * \brief CUDA implementation of g-SpMM on Csr format.
 * \param bcast Broadcast information.
 * \param csr The Csr matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 * \param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void SpMMCsr(
    const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  const Idx *indptr = csr.indptr.Ptr<Idx>();
  const Idx *indices = csr.indices.Ptr<Idx>();
  const Idx *edge_map = csr.data.Ptr<Idx>();
  const DType *ufeat_data = ufeat.Ptr<DType>();
  const DType *efeat_data = efeat.Ptr<DType>();
  DType *out_data = out.Ptr<DType>();
  Idx* argu_data = argu.Ptr<Idx>();
  Idx* arge_data = arge.Ptr<Idx>();

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len,
          lhs_len = bcast.lhs_len,
          rhs_len = bcast.rhs_len;
  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((csr.num_rows + nty - 1) / nty);
  // LOG(INFO) << "nblks=(" << nbx << ", " << nby << ") nthrs=(" << ntx << ", " << nty << ")";
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(csr.data);

  BCAST_IDX_CTX_SWITCH(bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off, {
    CUDA_KERNEL_CALL((SpMMCsrKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
        nblks, nthrs, 0, thr_entry->stream,
        ufeat_data, efeat_data, out_data, argu_data, arge_data,
        indptr, indices, edge_map,
        csr.num_rows, csr.num_cols,
        ubcast_off, ebcast_off,
        lhs_len, rhs_len, len)
  });
}

/*!
 * \brief CUDA implementation of edge_softmax backward on Csr format.
 * \param bcast Broadcast information.
 * \param csr The Csr matrix.
 * \param f_out The result of edge_softmax forward.
 * \param sds The result of  f_out * gradiet.
 * \param back_out The result of edge_softmax backward, named grad_score in python .
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void Edge_softmax_csr_backward(
    const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray f_out, NDArray sds,
    NDArray back_out) {
  const Idx *indptr = csr.indptr.Ptr<Idx>();
  const Idx *indices = csr.indices.Ptr<Idx>();
  const Idx *edge_map = csr.data.Ptr<Idx>();
  const DType *f_out_data = f_out.Ptr<DType>();
  const DType *sds_data = sds.Ptr<DType>();
  DType *back_out_data = back_out.Ptr<DType>();

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len,
          lhs_len = bcast.lhs_len,
          rhs_len = bcast.rhs_len;
  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((csr.num_rows + nty - 1) / nty);
  // LOG(INFO) << "rhs_len=(" <<rhs_len;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(csr.data);

  BCAST_IDX_CTX_SWITCH(bcast, use_idx, f_out->ctx, ubcast_off, ebcast_off, {
    CUDA_KERNEL_CALL((Edge_backwardKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
        nblks, nthrs, 0, thr_entry->stream,
        f_out_data, sds_data, back_out_data,
        indptr, indices, edge_map,
        csr.num_rows, csr.num_cols,
        ubcast_off, ebcast_off,
        rhs_len, rhs_len, len)
  });
}
/*!
 * \brief CUDA kernel of SpMM-Min/Max on Csr format on heterogeneous graph.
 * \param bcast Broadcast information.
 * \param csr The Csr matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 * \param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 * \param argu_ntype Node type of the arg-Min/Max on source nodes, which refers the
 *        source node types correspond to the minimum/maximum values of reduction result
 *        on destination nodes. It's useful in computing gradients of Min/Max reducer.
 * \param arge_etype Edge-type of the arg-Min/Max on edges. which refers the source
 *        node indices correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 * \param src_type Node type of the source nodes of an etype
 * \param etype Edge type
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void SpMMCmpCsrHetero(
    const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge,
    NDArray argu_ntype, NDArray arge_etype,
    const int src_type, const int etype) {
  const Idx *indptr = csr.indptr.Ptr<Idx>();
  const Idx *indices = csr.indices.Ptr<Idx>();
  const Idx *edge_map = csr.data.Ptr<Idx>();
  const DType *ufeat_data = ufeat.Ptr<DType>();
  const DType *efeat_data = efeat.Ptr<DType>();
  DType *out_data = out.Ptr<DType>();
  Idx* argu_data = argu.Ptr<Idx>();
  Idx* arge_data = arge.Ptr<Idx>();

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len,
          lhs_len = bcast.lhs_len,
          rhs_len = bcast.rhs_len;
  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((csr.num_rows + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(csr.data);

  BCAST_IDX_CTX_SWITCH(bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off, {
    CUDA_KERNEL_CALL((SpMMCmpCsrHeteroKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
        nblks, nthrs, 0, thr_entry->stream,
        ufeat_data, efeat_data, out_data, argu_data, arge_data,
        static_cast<Idx*>(argu_ntype->data),
        static_cast<Idx*>(arge_etype->data),
        indptr, indices, edge_map,
        csr.num_rows, csr.num_cols,
        ubcast_off, ebcast_off,
        lhs_len, rhs_len, len, src_type, etype)
  });
}


}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif
