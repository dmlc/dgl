/*!
 *  Copyright (c) 2020 by Contributors
 * \file kernel/cuda/sddmm.cuh
 * \brief SpMM CUDA kernel header file
 */

#ifndef DGL_KERNEL_CUDA_SPMM_CUH_
#define DGL_KERNEL_CUDA_SPMM_CUH_

#include "../utils.h"
#include "../binary_reduce_impl_decl.h"
#include "../binary_reduce.h"
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

template <typename DType>
inline DType* get_ndarray_ptr(const NDArray& array) {
  if (aten::IsNullArray(array))
    return nullptr;
  return static_cast<DType*>(array->data);
}

template <typename DType>
__global__ void _FillKernel(DType* ptr, size_t length, DType val) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    ptr[tx] = val;
    tx += stride_x;
  }
}

template <typename DType>
void Fill(const DLContext& ctx, DType* ptr, size_t length, DType val) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int nt = utils::FindNumThreads(length, 1024);
  int nb = (length + nt - 1) / nt;
  _FillKernel<<<nb, nt, 0, thr_entry->stream>>>(ptr, length, val);
}

template void Fill<float>(const DLContext& ctx, float* ptr, size_t length, float val);
template void Fill<double>(const DLContext& ctx, double* ptr, size_t length, double val);

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
__global__ void SpMMCooKernel(
  const DType *ufeat, const DType *efeat, DType *out, Idx *arg_u, Idx *arg_e,
  const Idx *row, const Idx *col, const Idx* edge_map,
  int64_t N, int64_t M, int64_t E,
  int64_t *ubcast_off, int64_t *ebcast_off,
  int64_t ufeat_len, int64_t efeat_len, int64_t out_len) {
  // SPMM with COO.
  const bool has_idx = edge_map;
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = has_idx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* uoff = BinaryOp::use_lhs ? (ufeat + src * ufeat_len): nullptr;
    const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len): nullptr;
    DType* outoff = out + dst * out_len;
    while (tx < out_len) {
      int64_t lhs_add = ubcast_off ? ubcast_off[tx] : tx;
      int64_t rhs_add = ebcast_off ? ebcast_off[tx] : tx;
      DType val = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
      Idx* arguoff = (ReduceOp::require_arg && BinaryOp::use_lhs) ? (arg_u + dst * out_len + tx): nullptr;
      Idx* argeoff = (ReduceOp::require_arg && BinaryOp::use_rhs) ? (arg_e + dst * out_len + tx): nullptr;
      ReduceOp::Call(outoff + tx, arguoff, argeoff, val, src, eid);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
__global__ void ArgSpMMCooKernel(
  const DType *ufeat, const DType *efeat, DType *out, Idx *arg_u, Idx *arg_e,
  const Idx *row, const Idx *col, const Idx* edge_map,
  int64_t N, int64_t M, int64_t E,
  int64_t *ubcast_off, int64_t *ebcast_off,
  int64_t ufeat_len, int64_t efeat_len, int64_t out_len) {
  // SPMM with COO arg max/min.
  const bool has_idx = edge_map;
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = has_idx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* uoff = BinaryOp::use_lhs ? (ufeat + src * ufeat_len): nullptr;
    const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len): nullptr;
    DType* outoff = out + dst * out_len;
    Idx* arguoff = BinaryOp::use_lhs ? (arg_u + dst * out_len): nullptr;
    Idx* argeoff = BinaryOp::use_rhs ? (arg_e + dst * out_len): nullptr;
    while (tx < out_len) {
      int64_t lhs_add = ubcast_off ? ubcast_off[tx] : tx;
      int64_t rhs_add = ebcast_off ? ebcast_off[tx] : tx;
      DType val = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
      ReduceOp::CallArg(tx, arguoff, argeoff, val, outoff[tx], src, eid);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
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
      const int lhs_add = ubcast_off ? ubcast_off[tx] : tx;
      const int rhs_add = ebcast_off ? ebcast_off[tx] : tx;
      for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const Idx eid = edge_map ? _ldg(edge_map + i) : i;
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

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void SpMMCoo(
    const dgl::aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  Idx *row = get_ndarray_ptr<Idx>(coo.row),
      *col = get_ndarray_ptr<Idx>(coo.col),
      *edge_map = get_ndarray_ptr<Idx>(coo.data);
  DType *ufeat_data = get_ndarray_ptr<DType>(ufeat),
        *efeat_data = get_ndarray_ptr<DType>(efeat),
        *out_data = get_ndarray_ptr<DType>(out);
  Idx *argu_data = get_ndarray_ptr<Idx>(argu),
      *arge_data = get_ndarray_ptr<Idx>(arge);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int64_t N = coo.num_rows, M = coo.num_cols, E = coo.row->shape[0];

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = 1;
  for (int64_t i = 1; i < out->ndim; ++i)
    len *= out->shape[i];
  Fill<DType>(out->ctx, out_data, len * out->shape[0], ReduceOp::zero);

  const int ntx = utils::FindNumThreads(len, 1024);
  const int nty = 1024 / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = utils::FindNumBlocks((E + nty - 1) / nty, 65535);
  //LOG(INFO) << "nblks=(" << nbx << ", " << nby << ") nthrs=(" << ntx << ", " << nty << ")";
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  SpMMCooKernel<Idx, DType, BinaryOp, ReduceOp>
    <<<nblks, nthrs, 0, thr_entry->stream>>>(
      ufeat_data, efeat_data, out_data, argu_data, arge_data,
      row, col, edge_map,
      N, M, E,
      ubcast_off, ebcast_off,
      len, len, len
    );
  if (ReduceOp::require_arg) {
    ArgSpMMCooKernel<Idx, DType, BinaryOp, ReduceOp>
      <<<nblks, nthrs, 0, thr_entry->stream>>>(
        ufeat_data, efeat_data, out_data, argu_data, arge_data,
        row, col, edge_map,
        N, M, E,
        ubcast_off, ebcast_off,
        len, len, len
      );
  }
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void SpMMBcastCoo(
    const BcastInfo& info,
    const dgl::aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  Idx *row = get_ndarray_ptr<Idx>(coo.row),
      *col = get_ndarray_ptr<Idx>(coo.col),
      *edge_map = get_ndarray_ptr<Idx>(coo.data);
  DType *ufeat_data = get_ndarray_ptr<DType>(ufeat),
        *efeat_data = get_ndarray_ptr<DType>(efeat),
        *out_data = get_ndarray_ptr<DType>(out);
  Idx *argu_data = get_ndarray_ptr<Idx>(argu),
      *arge_data = get_ndarray_ptr<Idx>(arge);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int64_t N = coo.num_rows, M = coo.num_cols, E = coo.row->shape[0]; 

  DLContext ctx = ufeat->ctx;
  auto device = runtime::DeviceAPI::Get(ctx); 
  int64_t *ubcast_off = static_cast<int64_t*>(
    device->AllocWorkspace(ctx, sizeof(int64_t) * info.lhs_offset.size()));
  CUDA_CALL(cudaMemcpy(ubcast_off, &info.lhs_offset[0],
    sizeof(int64_t) * info.lhs_offset.size(), cudaMemcpyHostToDevice));
  int64_t *ebcast_off = static_cast<int64_t*>(
    device->AllocWorkspace(ctx, sizeof(int64_t) * info.rhs_offset.size()));
  CUDA_CALL(cudaMemcpy(ebcast_off, &info.rhs_offset[0],
    sizeof(int64_t) * info.rhs_offset.size(), cudaMemcpyHostToDevice));

  int64_t ufeat_len = utils::Prod(info.lhs_shape);
  int64_t efeat_len = utils::Prod(info.rhs_shape);
  int64_t out_len = utils::Prod(info.out_shape);
  Fill<DType>(ctx, out_data, out_len * out->shape[0], ReduceOp::zero);

  const int ntx = utils::FindNumThreads(out_len, 1024);
  const int nty = 1024 / ntx;
  const int nbx = (out_len + ntx - 1) / ntx;
  const int nby = utils::FindNumBlocks((E + nty - 1) / nty, 65535);
  //LOG(INFO) << "nblks=(" << nbx << ", " << nby << ") nthrs=(" << ntx << ", " << nty << ")";
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  SpMMCooKernel<Idx, DType, BinaryOp, ReduceOp>
    <<<nblks, nthrs, 0, thr_entry->stream>>>(
      ufeat_data, efeat_data, out_data, argu_data, arge_data,
      row, col, edge_map,
      N, M, E,
      ubcast_off, ebcast_off,
      ufeat_len, efeat_len, out_len
    );
  if (ReduceOp::require_arg) {
    ArgSpMMCooKernel<Idx, DType, BinaryOp, ReduceOp>
      <<<nblks, nthrs, 0, thr_entry->stream>>>(
        ufeat_data, efeat_data, out_data, argu_data, arge_data,
        row, col, edge_map,
        N, M, E,
        ubcast_off, ebcast_off,
        ufeat_len, efeat_len, out_len
      );
  }
  device->FreeWorkspace(ctx, ubcast_off);
  device->FreeWorkspace(ctx, ebcast_off);
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void SpMMCsr(
    const dgl::aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  const Idx *indptr = get_ndarray_ptr<Idx>(csr.indptr);
  const Idx *indices = get_ndarray_ptr<Idx>(csr.indices);
  const Idx *edge_map = get_ndarray_ptr<Idx>(csr.data);
  const DType *ufeat_data = get_ndarray_ptr<DType>(ufeat);
  const DType *efeat_data = get_ndarray_ptr<DType>(efeat);
  DType *out_data = get_ndarray_ptr<DType>(out);
  Idx* argu_data = get_ndarray_ptr<Idx>(argu);
  Idx* arge_data = get_ndarray_ptr<Idx>(arge);

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = 1;
  for (int64_t i = 1; i < out->ndim; ++i)
    len *= out->shape[i];

  const int ntx = utils::FindNumThreads(len, 1024);
  const int nty = 1024 / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = utils::FindNumBlocks((csr.num_rows + nty - 1) / nty, 65535);
  //LOG(INFO) << "nblks=(" << nbx << ", " << nby << ") nthrs=(" << ntx << ", " << nty << ")";
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  SpMMCsrKernel<Idx, DType, BinaryOp, ReduceOp>
    <<<nblks, nthrs, 0, thr_entry->stream>>>(
      ufeat_data, efeat_data, out_data, argu_data, arge_data,
      indptr, indices, edge_map,
      csr.num_rows, csr.num_cols, efeat->shape[0],
      ubcast_off, ebcast_off,
      len, len, len
    );
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void SpMMBcastCsr(
    const BcastInfo& info,
    const dgl::aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  const Idx *indptr = get_ndarray_ptr<Idx>(csr.indptr);
  const Idx *indices = get_ndarray_ptr<Idx>(csr.indices);
  const Idx *edge_map = get_ndarray_ptr<Idx>(csr.data);
  const DType *ufeat_data = get_ndarray_ptr<DType>(ufeat);
  const DType *efeat_data = get_ndarray_ptr<DType>(efeat);
  DType *out_data = get_ndarray_ptr<DType>(out);
  Idx* argu_data = get_ndarray_ptr<Idx>(argu);
  Idx* arge_data = get_ndarray_ptr<Idx>(arge);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  DLContext ctx = ufeat->ctx;
  auto device = runtime::DeviceAPI::Get(ctx); 
  int64_t *ubcast_off = static_cast<int64_t*>(
    device->AllocWorkspace(ctx, sizeof(int64_t) * info.lhs_offset.size()));
  CUDA_CALL(cudaMemcpy(ubcast_off, &info.lhs_offset[0],
    sizeof(int64_t) * info.lhs_offset.size(), cudaMemcpyHostToDevice));
  int64_t *ebcast_off = static_cast<int64_t*>(
    device->AllocWorkspace(ctx, sizeof(int64_t) * info.rhs_offset.size()));
  CUDA_CALL(cudaMemcpy(ebcast_off, &info.rhs_offset[0],
    sizeof(int64_t) * info.rhs_offset.size(), cudaMemcpyHostToDevice));

  int64_t len = utils::Prod(info.out_shape),
          lhs_len = utils::Prod(info.lhs_shape),
          rhs_len = utils::Prod(info.rhs_shape);

  const int ntx = utils::FindNumThreads(len, 1024);
  const int nty = 1024 / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = utils::FindNumBlocks((csr.num_rows + nty - 1) / nty, 65535);
  //LOG(INFO) << "nblks=(" << nbx << ", " << nby << ") nthrs=(" << ntx << ", " << nty << ")";
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  SpMMCsrKernel<Idx, DType, BinaryOp, ReduceOp>
    <<<nblks, nthrs, 0, thr_entry->stream>>>(
      ufeat_data, efeat_data, out_data, argu_data, arge_data,
      indptr, indices, edge_map,
      csr.num_rows, csr.num_cols, efeat->shape[0],
      ubcast_off, ebcast_off,
      lhs_len, rhs_len, len
    );
  device->FreeWorkspace(ctx, ubcast_off);
  device->FreeWorkspace(ctx, ebcast_off);
}

}  // namespace cuda
}  // namespace kernel
}  // namespace dgl

#endif
