/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/segment_reduce.cuh
 * \brief Segment reduce kernel function header.
 */
#ifndef DGL_ARRAY_SEGMENT_REDUCE_CUH_
#define DGL_ARRAY_SEGMENT_REDUCE_CUH_

#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"
#include "./atomic.cuh"
#include <algorithm>
#include <vector>

namespace dgl {

using namespace cuda;

namespace aten {
namespace cuda {

/*!
 * \brief CUDA kernel of segment reduce.
 * \note each blockthread is responsible for aggregation on a row
 *       in the result tensor.
 */
template <typename IdType, typename DType,
          typename ReduceOp>
__global__ void SegmentReduceKernel(
    const DType* feat, const IdType* offsets,
    DType* out, IdType* arg,
    int64_t n, int64_t dim){
  for (int row = blockIdx.x; row < n; row += gridDim.x) {
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    while (col < dim) {
      DType local_accum = ReduceOp::zero();
      IdType local_arg = -1;
      for (IdType i = offsets[row]; i < offsets[row + 1]; ++i) {
        ReduceOp::Call(&local_accum, &local_arg, feat[i * dim + col], i);
      }
      out[row * dim + col] = local_accum;
      if (ReduceOp::require_arg)
        arg[row * dim + col] = local_arg;
      col += gridDim.y * blockDim.x;
    }
  }
}

/*!
 * \brief CUDA kernel of scatter add.
 * \note each blockthread is responsible for adding a row in feature tensor
 *       to a target row in output tensor.
 */
template <typename IdType, typename DType>
__global__ void ScatterAddKernel(
    const DType *feat, const IdType *idx, DType *out,
    int64_t n, int64_t dim) {
  for (int row = blockIdx.x; row < n; row += gridDim.x) {
    const int write_row = idx[row];
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    while (col < dim) {
      cuda::AtomicAdd(out + write_row * dim + col, feat[row * dim + col]);
      col += gridDim.y * blockDim.x;
    }
  }
}

/*!
 * \brief CUDA kernel of backward phase in segment min/max.
 * \note each blockthread is responsible for writing a row in the
 *       result gradient tensor by lookup the ArgMin/Max for index information.
 */
template <typename IdType, typename DType>
__global__ void BackwardSegmentCmpKernel(
    const DType *feat, const IdType *arg, DType *out,
    int64_t n, int64_t dim) {
  for (int row = blockIdx.x; row < n; row += gridDim.x) {
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    while (col < dim) {
      int write_row = arg[row * dim + col];
      if (write_row >= 0) {
        out[write_row * dim + col] = feat[row * dim + col];
      }
      col += gridDim.y * blockDim.x;
    }
  }
}

/*!
 * \brief CUDA implementation of forward phase of Segment Reduce.
 * \param feat The input tensor.
 * \param offsets The offsets tensor.
 * \param out The output tensor.
 * \param arg An auxiliary tensor storing ArgMax/Min information,
 */
template <typename IdType, typename DType, typename ReduceOp>
void SegmentReduce(
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg) {
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* offsets_data = offsets.Ptr<IdType>();
  DType* out_data = out.Ptr<DType>();
  IdType* arg_data = arg.Ptr<IdType>();

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int64_t n = out->shape[0];
  int64_t dim = 1;
  for (int i = 1; i < out->ndim; ++i)
    dim *= out->shape[i];

  const int nbx = FindNumBlocks<'x'>(n);
  const int ntx = FindNumThreads(dim);
  const int nby = FindNumBlocks<'y'>((dim + ntx - 1) / ntx);
  const int nty = 1;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  // TODO(zihao): try cub's DeviceSegmentedReduce and compare the performance.
  CUDA_KERNEL_CALL((SegmentReduceKernel<IdType, DType, ReduceOp>),
      nblks, nthrs, 0, thr_entry->stream,
      feat_data, offsets_data, out_data, arg_data,
      n, dim);
}

namespace {

template <typename DType>
cublasStatus_t
Xgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
      int m, int n, int k, const DType *alpha, const DType *A, int lda,
      const DType *B, int ldb, const DType *beta, DType *C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <>
cublasStatus_t
Xgemm<float> (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
              int m, int n, int k, const float *alpha, const float *A, int lda,
              const float *B, int ldb, const float *beta, float *C, int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
cublasStatus_t
Xgemm<double> (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
               int m, int n, int k, const double *alpha, const double *A, int lda,
               const double *B, int ldb, const double *beta, double *C, int ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
cublasStatus_t
Xgemm<half> (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
             int m, int n, int k, const half *alpha, const half *A, int lda,
             const half *B, int ldb, const half *beta, half *C, int ldc) {
  return cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <typename DType>
cublasStatus_t
XgemmBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const DType *alpha,
    const DType *A, int lda, int64_t stride_a,
    const DType *B, int ldb, int64_t stride_b,
    const DType *beta,
    DType *C, int ldc, int64_t stride_c,
    int batchCount) {
  LOG(INFO) << "Not supported dtype";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <>
cublasStatus_t
XgemmBatched<float>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda, int64_t stride_a,
    const float *B, int ldb, int64_t stride_b,
    const float *beta,
    float *C, int ldc, int64_t stride_c,
    int batch_count) {
  return cublasSgemmStridedBatched(
      handle,
      transa, transb,
      m, n, k,
      alpha,
      A, lda,
      stride_a,
      B, ldb,
      stride_b,
      beta,
      C, ldc,
      stride_c,
      batch_count
  );
}

template <>
cublasStatus_t
XgemmBatched<double>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double *alpha,
    const double *A, int lda, int64_t stride_a,
    const double *B, int ldb, int64_t stride_b,
    const double *beta,
    double *C, int ldc, int64_t stride_c,
    int batch_count) {
  return cublasDgemmStridedBatched(
      handle,
      transa, transb,
      m, n, k,
      alpha,
      A, lda,
      stride_a,
      B, ldb,
      stride_b,
      beta,
      C, ldc,
      stride_c,
      batch_count
  );
}

template <>
cublasStatus_t
XgemmBatched<half>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const half *alpha,
    const half *A, int lda, int64_t stride_a,
    const half *B, int ldb, int64_t stride_b,
    const half *beta,
    half *C, int ldc, int64_t stride_c,
    int batch_count) {
  return cublasHgemmStridedBatched(
      handle,
      transa, transb,
      m, n, k,
      alpha,
      A, lda,
      stride_a,
      B, ldb,
      stride_b,
      beta,
      C, ldc,
      stride_c,
      batch_count
  );
}

}  // namespace

template <typename DType>
__global__ void SegmentGemmKernel(
    const DType* __restrict__ A,
    const DType* __restrict__ B,
    DType* __restrict__ C,
    const int64_t* __restrict__ m,
    const int64_t* __restrict__ n,
    const int64_t* __restrict__ k,
    const int64_t* __restrict__ blk_seg_map,
    bool transA,
    bool transB
) {
  
  // TODO(zihao)
}

/*
 * \brief CUDA implementation of forward phase of SegmentGemm.
 * \param A the concatenation of {A_i}
 * \param B the concatenation of {B_i}
 * \param C the concatenation of {C_i}
 * \param m the array of number of rows in A.
 * \param n the array of number of columns in C.
 * \param k the array of number of rows in B / number of columns in A.
 */
template <typename DType>
void SegmentGemm(NDArray A, NDArray B, NDArray C,
                 NDArray m, NDArray n, NDArray k,
                 bool transA, bool transB) {
  auto ctx = A->ctx;
  const DType *A_data = A.Ptr<DType>();
  const DType *B_data = B.Ptr<DType>();
  DType* C_data = C.Ptr<DType>();
  const int64_t *m_data = m.Ptr<int64_t>();
  const int64_t *n_data = n.Ptr<int64_t>();
  const int64_t *k_data = k.Ptr<int64_t>();
  int64_t batch_size = n->shape[0];
  auto device = runtime::DeviceAPI::Get(ctx);
  auto *thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  if (!thr_entry->cublas_handle)
    CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
  CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle, thr_entry->stream));

  std::vector<int64_t> A_off(batch_size + 1, 0),
                       B_off(batch_size + 1, 0),
                       C_off(batch_size + 1, 0);
  std::vector<int64_t> blk_seg_map_cpu;
  // collect information
  for (int i = 0; i < batch_size; ++i) {
    int64_t mi = m_data[i],
            ni = n_data[i],
            ki = k_data[i];
    A_off[i + 1] = A_off[i] + mi * ki;
    B_off[i + 1] = B_off[i] + ki * ni;
    C_off[i + 1] = C_off[i] + mi * ni;
    int seg_mi = (mi + 31) / 32,
        seg_ni = (ni + 31) / 32;
    for (int j = 0; j < seg_mi * seg_ni; ++j) {
      blk_seg_map_cpu.push_back(i);
    }
  } 
  
  // parallelize with CUDA stream.
  for (int i = 0; i < batch_size; ++i) {
    int64_t mi = m_data[i],
            ni = n_data[i],
            ki = k_data[i];
    if (mi > 0 && ni > 0 && ki > 0) {
      DType alpha = 1., beta = 0.;
      int ldb = transB ? ki: ni,
          lda = transA ? mi: ki,
          ldc = ni;
      CUBLAS_CALL(Xgemm<DType>(
          thr_entry->cublas_handle,
          transB ? CUBLAS_OP_T : CUBLAS_OP_N,
          transA ? CUBLAS_OP_T : CUBLAS_OP_N,
          ni, mi, ki,
          &alpha,
          B_data + B_off[i], ldb,
          A_data + A_off[i], lda,
          &beta,
          C_data + C_off[i], ldc
      ));
    }
  }
}

/*!
 * \brief CUDA implementation of Scatter Add (on first dimension).
 * \note math equation: out[idx[i], *] += feat[i, *]
 * \param feat The input tensor.
 * \param idx The indices tensor.
 * \param out The output tensor.
 */
template <typename IdType, typename DType>
void ScatterAdd(
    NDArray feat,
    NDArray idx,
    NDArray out) {
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* idx_data = idx.Ptr<IdType>();
  DType *out_data = out.Ptr<DType>();
  
  auto *thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int64_t n = feat->shape[0];
  int64_t dim = 1;
  for (int i = 1; i < out->ndim; ++i)
    dim *= out->shape[i];

  const int nbx = FindNumBlocks<'x'>(n);
  const int ntx = FindNumThreads(dim);
  const int nby = FindNumBlocks<'y'>((dim + ntx - 1) / ntx);
  const int nty = 1;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  CUDA_KERNEL_CALL((ScatterAddKernel<IdType, DType>),
                   nblks, nthrs, 0, thr_entry->stream,
                   feat_data, idx_data, out_data,
                   n, dim);
}

/*!
 * \brief CUDA implementation of backward phase of Segment Reduce with Min/Max reducer.
 * \note math equation: out[arg[i, k], k] = feat[i, k]
 * \param feat The input tensor.
 * \param arg The ArgMin/Max information, used for indexing.
 * \param out The output tensor.
 */
template <typename IdType, typename DType>
void BackwardSegmentCmp(
    NDArray feat,
    NDArray arg,
    NDArray out) {
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* arg_data = arg.Ptr<IdType>();
  DType *out_data = out.Ptr<DType>();

  auto *thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int64_t n = feat->shape[0];
  int64_t dim = 1;
  for (int i = 1; i < out->ndim; ++i)
    dim *= out->shape[i];

  const int nbx = FindNumBlocks<'x'>(n);
  const int ntx = FindNumThreads(dim);
  const int nby = FindNumBlocks<'y'>((dim + ntx - 1) / ntx);
  const int nty = 1;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  CUDA_KERNEL_CALL((BackwardSegmentCmpKernel<IdType, DType>),
                   nblks, nthrs, 0, thr_entry->stream,
                   feat_data, arg_data, out_data,
                   n, dim);
}

}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif
