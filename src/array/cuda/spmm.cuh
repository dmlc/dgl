/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/spmm.cuh
 * @brief SPMM CUDA kernel function header.
 */
#ifndef DGL_ARRAY_CUDA_SPMM_CUH_
#define DGL_ARRAY_CUDA_SPMM_CUH_

#include <dgl/bcast.h>

#include <limits>

#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"
#include "atomic.cuh"
#include "bf16.cuh"
#include "fp16.cuh"
#include "macro.cuh"

namespace dgl {

using namespace cuda;

namespace aten {

/**
 * @brief Determine whether cusparse SpMM function is applicable.
 */
template <typename DType, typename IdType>
inline bool cusparse_available(bool more_nnz_than_matrix_size) {
  if (std::is_same<DType, __half>::value ||
      std::is_same<DType, __nv_bfloat16>::value)
    return false;  // cusparse's SpMM on fp16 is slow, temporally disabled.
  // If the CSR matrix has more NNZ than matrix size, we should not use
  // cuSPARSE 11.1.
  return !more_nnz_than_matrix_size;
}

namespace {

/** Cusparse implementation of SpMM on Csr format. */
template <typename DType, typename IdType>
void CusparseCsrmm2(
    const DGLContext& ctx, const CSRMatrix& csr, const DType* B_data,
    const DType* A_data, DType* C_data, int x_length,
    bool use_deterministic_alg_only = false) {
  // We use csrmm2 to perform following operation:
  // C = A x B, where A is a sparse matrix in csr format, B is the dense matrix
  // for node feature tensor. However, since cusparse only supports
  // column-major, while our tensor is stored in row-major, the actual
  // computation is: C = trans(A x trans(B)). Currently, we use cublasXgeam to
  // implement transposition and allocate intermediate workspace memory for
  // this.
  const int m = csr.num_rows;
  const int n = x_length;
  const int k = csr.num_cols;
  const int nnz = csr.indices->shape[0];
  const DType alpha = 1.0;
  const DType beta = 0.0;
  // device
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, stream));
  // all one data array
  DType* valptr = nullptr;
  if (!A_data) {
    valptr =
        static_cast<DType*>(device->AllocWorkspace(ctx, nnz * sizeof(DType)));
    _Fill(valptr, nnz, static_cast<DType>(1.));
  }
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  constexpr auto dtype = cuda_dtype<DType>::value;
  constexpr auto idtype = cusparse_idtype<IdType>::value;
  CUSPARSE_CALL(cusparseCreateCsr(
      &matA, m, k, nnz, static_cast<IdType*>(csr.indptr->data),
      static_cast<IdType*>(csr.indices->data),
      const_cast<DType*>(valptr ? valptr : A_data), idtype, idtype,
      CUSPARSE_INDEX_BASE_ZERO, dtype));
  CUSPARSE_CALL(cusparseCreateDnMat(
      &matB, k, n, n, const_cast<DType*>(B_data), dtype, CUSPARSE_ORDER_ROW));
  CUSPARSE_CALL(
      cusparseCreateDnMat(&matC, m, n, n, C_data, dtype, CUSPARSE_ORDER_ROW));

  auto transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  size_t workspace_size;
  cusparseSpMMAlg_t spmm_alg = use_deterministic_alg_only
                                   ? CUSPARSE_SPMM_CSR_ALG3
                                   : CUSPARSE_SPMM_CSR_ALG2;
  CUSPARSE_CALL(cusparseSpMM_bufferSize(
      thr_entry->cusparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, spmm_alg, &workspace_size));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);
  CUSPARSE_CALL(cusparseSpMM(
      thr_entry->cusparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, spmm_alg, workspace));
  device->FreeWorkspace(ctx, workspace);

  CUSPARSE_CALL(cusparseDestroySpMat(matA));
  CUSPARSE_CALL(cusparseDestroyDnMat(matB));
  CUSPARSE_CALL(cusparseDestroyDnMat(matC));
  if (valptr) device->FreeWorkspace(ctx, valptr);
}

/** Cusparse implementation of SpMM on Csr format. */
template <typename DType, typename IdType>
void CusparseCsrmm2Hetero(
    const DGLContext& ctx, const CSRMatrix& csr, const DType* B_data,
    const DType* A_data, DType* C_data, int64_t x_length, cudaStream_t strm_id,
    bool use_deterministic_alg_only = false) {
  // We use csrmm2 to perform following operation:
  // C = A x B, where A is a sparse matrix in csr format, B is the dense matrix
  // for node feature tensor. However, since cusparse only supports
  // column-major, while our tensor is stored in row-major, the actual
  // computation is: C = trans(A x trans(B)). Currently, we use cublasXgeam to
  // implement transposition and allocate intermediate workspace memory for
  // this.
  int int_maxlimit = std::numeric_limits<int>::max();
  CHECK_GE(int_maxlimit, (csr.num_rows));
  CHECK_GE(int_maxlimit, csr.num_cols);
  CHECK_GE(int_maxlimit, csr.indices->shape[0]);
  const int m = csr.num_rows;
  const int n = x_length;
  const int k = csr.num_cols;
  const int nnz = csr.indices->shape[0];
  const DType alpha = 1.0;
  const DType beta = 1.0;
  // device
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, strm_id));
  // all one data array
  DType* valptr = nullptr;
  if (!A_data) {
    valptr =
        static_cast<DType*>(device->AllocWorkspace(ctx, nnz * sizeof(DType)));
    _Fill(valptr, nnz, static_cast<DType>(1.));
  }
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  constexpr auto dtype = cuda_dtype<DType>::value;
  constexpr auto idtype = cusparse_idtype<IdType>::value;
  CUSPARSE_CALL(cusparseCreateCsr(
      &matA, m, k, nnz, static_cast<IdType*>(csr.indptr->data),
      static_cast<IdType*>(csr.indices->data),
      const_cast<DType*>(valptr ? valptr : A_data), idtype, idtype,
      CUSPARSE_INDEX_BASE_ZERO, dtype));
  CUSPARSE_CALL(cusparseCreateDnMat(
      &matB, k, n, n, const_cast<DType*>(B_data), dtype, CUSPARSE_ORDER_ROW));
  CUSPARSE_CALL(
      cusparseCreateDnMat(&matC, m, n, n, C_data, dtype, CUSPARSE_ORDER_ROW));

  auto transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  size_t workspace_size;
  cusparseSpMMAlg_t spmm_alg = use_deterministic_alg_only
                                   ? CUSPARSE_SPMM_CSR_ALG3
                                   : CUSPARSE_SPMM_CSR_ALG2;
  CUSPARSE_CALL(cusparseSpMM_bufferSize(
      thr_entry->cusparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, spmm_alg, &workspace_size));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);
  CUSPARSE_CALL(cusparseSpMM(
      thr_entry->cusparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, spmm_alg, workspace));
  device->FreeWorkspace(ctx, workspace);

  CUSPARSE_CALL(cusparseDestroySpMat(matA));
  CUSPARSE_CALL(cusparseDestroyDnMat(matB));
  CUSPARSE_CALL(cusparseDestroyDnMat(matC));
  if (valptr) device->FreeWorkspace(ctx, valptr);
}

}  // namespace

#define SWITCH_OP(op, Op, ...)                                  \
  do {                                                          \
    if ((op) == "add") {                                        \
      typedef cuda::binary::Add<DType> Op;                      \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "sub") {                                 \
      typedef cuda::binary::Sub<DType> Op;                      \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "mul") {                                 \
      typedef cuda::binary::Mul<DType> Op;                      \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "div") {                                 \
      typedef cuda::binary::Div<DType> Op;                      \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "copy_lhs") {                            \
      typedef cuda::binary::CopyLhs<DType> Op;                  \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "copy_rhs") {                            \
      typedef cuda::binary::CopyRhs<DType> Op;                  \
      { __VA_ARGS__ }                                           \
    } else {                                                    \
      LOG(FATAL) << "Unsupported SpMM binary operator: " << op; \
    }                                                           \
  } while (0)

namespace cuda {

/**
 * @brief CUDA kernel of g-SpMM on Coo format.
 * @note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 * positions in feature dimension. To avoid possible data hazards, it uses
 * atomic operators for reduction.
 */
template <
    typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
    bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCooKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    const Idx* __restrict__ row, const Idx* __restrict__ col,
    const Idx* __restrict__ edge_map, int64_t N, int64_t M, int64_t E,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len) {
  // SPMM with COO.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* uoff = BinaryOp::use_lhs ? (ufeat + src * ufeat_len) : nullptr;
    const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
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

/**
 * @brief CUDA kernel to compute argu and arge in g-SpMM on Coo format.
 * @note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 * positions in feature dimension.
 */
template <
    typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
    bool UseBcast = false, bool UseIdx = false>
__global__ void ArgSpMMCooKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    const Idx* __restrict__ row, const Idx* __restrict__ col,
    const Idx* __restrict__ edge_map, int64_t N, int64_t M, int64_t E,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len) {
  // SPMM with COO arg max/min.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* uoff = BinaryOp::use_lhs ? (ufeat + src * ufeat_len) : nullptr;
    const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
    const DType* outoff = out + dst * out_len;
    Idx* arguoff = BinaryOp::use_lhs ? (arg_u + dst * out_len) : nullptr;
    Idx* argeoff = BinaryOp::use_rhs ? (arg_e + dst * out_len) : nullptr;
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

/**
 * @brief CUDA kernel of g-SpMM on Csr format.
 * @note it uses node parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different destination nodes.
 *       Threadblocks on the x-axis are responsible for the computation on
 *       different positions in feature dimension.
 */
template <
    typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
    bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCsrKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    const Idx* __restrict__ indptr, const Idx* __restrict__ indices,
    const Idx* __restrict__ edge_map, int64_t num_rows, int64_t num_cols,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len) {
  // SPMM with CSR.
  int ty = blockIdx.x * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.x;
  const int stride_x = blockDim.x * gridDim.y;
  while (ty < num_rows) {
    int tx = blockIdx.y * blockDim.x + threadIdx.x;
    while (tx < out_len) {
      typename accum_dtype<DType>::type local_accum = ReduceOp::zero();
      Idx local_argu = 0, local_arge = 0;
      const int lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int rhs_add = UseBcast ? ebcast_off[tx] : tx;
      for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const Idx eid = UseIdx ? _ldg(edge_map + i) : i;
        const Idx cid = _ldg(indices + i);
        const DType* uoff =
            BinaryOp::use_lhs ? (ufeat + cid * ufeat_len) : nullptr;
        const DType* eoff =
            BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
        DType out = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
        ReduceOp::Call(&local_accum, &local_argu, &local_arge, out, cid, eid);
      }
      // The use of += is to compute cross-type reducing on heterogeneous graph
      // when reduce op is `sum`.
      //     C = SpMM(SpA, B) + C
      // Separate kernel `SpMMCmpCsrHeteroKernel` is used for max- and
      // min-reducer. It does not affect the output on homogeneous graph as
      // `out` is initialized to zero.
      out[ty * out_len + tx] += static_cast<DType>(local_accum);
      if (ReduceOp::require_arg && BinaryOp::use_lhs)
        arg_u[ty * out_len + tx] = local_argu;
      if (ReduceOp::require_arg && BinaryOp::use_rhs)
        arg_e[ty * out_len + tx] = local_arge;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA kernel of SpMM-Min/Max on Csr format.
 * @note it uses node parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different destination nodes.
 *       Threadblocks on the x-axis are responsible for the computation on
 *       different positions in feature dimension.
 */
template <
    typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
    bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCmpCsrHeteroKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    Idx* __restrict__ arg_u_ntype, Idx* __restrict__ arg_e_etype,
    const Idx* __restrict__ indptr, const Idx* __restrict__ indices,
    const Idx* __restrict__ edge_map, int64_t num_rows, int64_t num_cols,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len, const int src_type, const int etype) {
  // SPMM with CSR.
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  const int stride_x = blockDim.x * gridDim.x;
  while (ty < num_rows) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    while (tx < out_len) {
      using accum_type = typename accum_dtype<DType>::type;
      accum_type local_accum =
          static_cast<accum_type>(out[ty * out_len + tx]);  // ReduceOp::zero();
      Idx local_argu = 0, local_arge = 0;
      const int lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int rhs_add = UseBcast ? ebcast_off[tx] : tx;
      for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const Idx eid = UseIdx ? _ldg(edge_map + i) : i;
        const Idx cid = _ldg(indices + i);
        const DType* uoff =
            BinaryOp::use_lhs ? (ufeat + cid * ufeat_len) : nullptr;
        const DType* eoff =
            BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
        DType tmp_out = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
        ReduceOp::Call(
            &local_accum, &local_argu, &local_arge, tmp_out, cid, eid);
      }
      // Update output only when max/min values are different that original
      // output
      DType new_out = static_cast<DType>(local_accum);
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

/**
 * @brief CUDA implementation of g-SpMM on Coo format.
 * @param bcast Broadcast information.
 * @param coo The Coo matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 */
template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp>
void SpMMCoo(
    const BcastOff& bcast, const COOMatrix& coo, NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  /**
   * TODO(Xin): Disable half precision for SpMMCoo due to the round-off error.
   * We should use fp32 for the accumulation but it's hard to modify the
   * current implementation.
   */
  if (std::is_same<DType, __half>::value ||
      std::is_same<DType, __nv_bfloat16>::value)
    LOG(FATAL) << "SpMMCoo doesn't support half precision fow now. "
               << "Please use SpMMCsr instead by allowing the graph "
               << "materialize CSR/CSC formats.";
  const Idx *row = coo.row.Ptr<Idx>(), *col = coo.col.Ptr<Idx>(),
            *edge_map = coo.data.Ptr<Idx>();
  const DType *ufeat_data = ufeat.Ptr<DType>(),
              *efeat_data = efeat.Ptr<DType>();
  DType* out_data = out.Ptr<DType>();
  Idx *argu_data = argu.Ptr<Idx>(), *arge_data = arge.Ptr<Idx>();
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const int64_t N = coo.num_rows, M = coo.num_cols, E = coo.row->shape[0];

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;

  int64_t out_size = out.NumElements();
  const int nt = FindNumThreads(out_size);
  const int nb = (out_size + nt - 1) / nt;
  CUDA_KERNEL_CALL(
      _FillKernel, nb, nt, 0, stream, out_data, out_size, ReduceOp::zero());

  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((E + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(coo.data);

  BCAST_IDX_CTX_SWITCH(bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off, {
    CUDA_KERNEL_CALL(
        (SpMMCooKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
        nblks, nthrs, 0, stream, ufeat_data, efeat_data, out_data, argu_data,
        arge_data, row, col, edge_map, N, M, E, ubcast_off, ebcast_off, lhs_len,
        rhs_len, len);
    if (ReduceOp::require_arg) {
      CUDA_KERNEL_CALL(
          (ArgSpMMCooKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
          nblks, nthrs, 0, stream, ufeat_data, efeat_data, out_data, argu_data,
          arge_data, row, col, edge_map, N, M, E, ubcast_off, ebcast_off,
          lhs_len, rhs_len, len);
    }
  });
}

/**
 * @brief CUDA implementation of g-SpMM on Csr format.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 */
template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp>
void SpMMCsr(
    const BcastOff& bcast, const CSRMatrix& csr, NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  const Idx* indptr = csr.indptr.Ptr<Idx>();
  const Idx* indices = csr.indices.Ptr<Idx>();
  const Idx* edge_map = csr.data.Ptr<Idx>();
  const DType* ufeat_data = ufeat.Ptr<DType>();
  const DType* efeat_data = efeat.Ptr<DType>();
  DType* out_data = out.Ptr<DType>();
  Idx* argu_data = argu.Ptr<Idx>();
  Idx* arge_data = arge.Ptr<Idx>();

  cudaStream_t stream = runtime::getCurrentCUDAStream();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;
  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nby = (len + ntx - 1) / ntx;
  const int nbx = FindNumBlocks<'x'>((csr.num_rows + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(csr.data);

  BCAST_IDX_CTX_SWITCH(
      bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off,
      {CUDA_KERNEL_CALL(
          (SpMMCsrKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
          nblks, nthrs, 0, stream, ufeat_data, efeat_data, out_data, argu_data,
          arge_data, indptr, indices, edge_map, csr.num_rows, csr.num_cols,
          ubcast_off, ebcast_off, lhs_len, rhs_len, len)});
}

/**
 * @brief CUDA kernel of SpMM-Min/Max on Csr format on heterogeneous graph.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param argu_ntype Node type of the arg-Min/Max on source nodes, which refers
 * the source node types correspond to the minimum/maximum values of reduction
 * result on destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param arge_etype Edge-type of the arg-Min/Max on edges. which refers the
 * source node indices correspond to the minimum/maximum values of reduction
 * result on destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param src_type Node type of the source nodes of an etype
 * @param etype Edge type
 */
template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp>
void SpMMCmpCsrHetero(
    const BcastOff& bcast, const CSRMatrix& csr, NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge, NDArray argu_ntype,
    NDArray arge_etype, const int src_type, const int etype) {
  const Idx* indptr = csr.indptr.Ptr<Idx>();
  const Idx* indices = csr.indices.Ptr<Idx>();
  const Idx* edge_map = csr.data.Ptr<Idx>();
  const DType* ufeat_data = ufeat.Ptr<DType>();
  const DType* efeat_data = efeat.Ptr<DType>();
  DType* out_data = out.Ptr<DType>();
  Idx* argu_data = argu.Ptr<Idx>();
  Idx* arge_data = arge.Ptr<Idx>();

  cudaStream_t stream = runtime::getCurrentCUDAStream();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;
  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((csr.num_rows + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(csr.data);

  BCAST_IDX_CTX_SWITCH(
      bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off,
      {CUDA_KERNEL_CALL(
          (SpMMCmpCsrHeteroKernel<
              Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
          nblks, nthrs, 0, stream, ufeat_data, efeat_data, out_data, argu_data,
          arge_data, static_cast<Idx*>(argu_ntype->data),
          static_cast<Idx*>(arge_etype->data), indptr, indices, edge_map,
          csr.num_rows, csr.num_cols, ubcast_off, ebcast_off, lhs_len, rhs_len,
          len, src_type, etype)});
}

}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CUDA_SPMM_CUH_
