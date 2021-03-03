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

}  // namespace

/*
 * \brief CUDA implementation of forward phase of SegmentGemm.
 * \param A the concatenation of {A_i}
 * \param B the concatenation of {B_i}
 * \param C the concatenation of {C_i}
 * \param n the array of number of rows in A.
 * \param m the array of number of rows in B / number of columns in A.
 * \param p the array of number of columns in C.
 */
template <typename DType>
void SegmentGemm(NDArray A, NDArray B, NDArray C,
                 NDArray n, NDArray m, NDArray p,
                 bool transA, bool transB) {
  const DType* A_data = A.Ptr<DType>();
  const DType* B_data = B.Ptr<DType>();
  DType* C_data = C.Ptr<DType>();
  const int64_t* n_data = n.Ptr<int64_t>();
  const int64_t* m_data = m.Ptr<int64_t>();
  const int64_t* p_data = p.Ptr<int64_t>();
  int batch_size = n->shape[0];
  // TODO(zihao): use multiple stream.
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  if (!thr_entry->cublas_handle)
    CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
  CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle, thr_entry->stream));
  int64_t A_off = 0, B_off = 0, C_off = 0;
  int stream_cnt = 0;
  for (int i = 0; i < batch_size; ++i) {
    int64_t ni = n_data[i],
            mi = m_data[i],
            pi = p_data[i];
    if (ni > 0 && mi > 0 && pi > 0) {
      if (stream_cnt == 4)
        stream_cnt = 0;
      DType alpha = 1., beta = 0.;
      int ldb = transB ? mi: pi,
          lda = transA ? ni: mi,
          ldc = pi;
      CUBLAS_CALL(Xgemm<DType>(
          thr_entry->cublas_handle,
          transB ? CUBLAS_OP_T : CUBLAS_OP_N,
          transA ? CUBLAS_OP_T : CUBLAS_OP_N,
          pi, ni, mi,
          &alpha,
          B_data + B_off, ldb,
          A_data + A_off, lda,
          &beta,
          C_data + C_off, ldc
      ));
    }
    A_off += ni * mi;
    B_off += mi * pi;
    C_off += ni * pi;
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
