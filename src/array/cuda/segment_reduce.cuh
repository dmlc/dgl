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
__global__ void _PadKernel(
    DType* __restrict__ dst, const DType* __restrict__ src,
    int64_t* __restrict__ parallel_indices,
    int64_t* row, int64_t* col,
    int64_t* offset,
    int64_t stride) {
  // TODO(zihao)
}

template <typename DType>
__global__ void _UnpadKernel(
    DType* __restrict__ C_dst, const DType* __restrict__ C_src,
    int64_t* __restrict__ parallel_indices,
    int64_t* n_arr, int64_t* p_arr,
    int64_t* C_off,
    int64_t C_stride) {
  // TODO(zihao)
}

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
  auto ctx = A->ctx;
  const DType* A_data = A.Ptr<DType>();
  const DType* B_data = B.Ptr<DType>();
  DType* C_data = C.Ptr<DType>();
  const int64_t* n_data = n.Ptr<int64_t>();
  const int64_t* m_data = m.Ptr<int64_t>();
  const int64_t* p_data = p.Ptr<int64_t>();
  int batch_size = n->shape[0];
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  if (!thr_entry->cublas_handle)
    CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
  CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle, thr_entry->stream));
  int max_n = 0,
      max_m = 0,
      max_p = 0;
  std::vector<int> parallel_indices;
  std::vector<bool> mask{false};
  std::vector<int> A_off, B_off, C_off;
  // collect information
  for (int i = 0; i < batch_size; ++i) {
    int64_t ni = n_data[i],
            mi = m_data[i],
            pi = p_data[i];
    A_off.push_back(ni * mi);
    B_off.push_back(mi * pi);
    C_off.push_back(ni * pi);
    if (ni > 128 || mi > 128 || pi > 128) {
      mask[i] = true;
    } else if (ni > 0 && mi > 0 && pi > 0) {
      parallel_indices.push_back(i);
      max_n = std::max(int(ni), max_n);
      max_m = std::max(int(mi), max_m);
      max_p = std::max(int(pi), max_p);
    }
  } 
  // alignment to use TensorCore.
  const int multiple_shp = 8;
  max_n = multiple_shp * ((max_n + multiple_shp - 1) / multiple_shp);
  max_m = multiple_shp * ((max_m + multiple_shp - 1) / multiple_shp);
  max_p = multiple_shp * ((max_p + multiple_shp - 1) / multiple_shp);

  // parallel with cublas batched gemm.
  int parallel_batch_size = parallel_indices.size();
  if (parallel_batch_size > 0) {
    int64_t stride_a = max_n * max_m,
            stride_b = max_m * max_p,
            stride_c = max_n * max_p;
    // allocate memory
    DType *ptr_0 = static_cast<DType*>(
        device->AllocWorkspace(ctx, sizeof(DType) * (stride_a + stride_b + stride_c) * parallel_batch_size));
    DType *A_padded = ptr_0,
          *B_padded = ptr_0 + stride_a * parallel_batch_size,
          *C_padded = ptr_0 + (stride_a + stride_b) * parallel_batch_size;
    int64_t *ptr_1 = static_cast<int64_t*>(device->AllocWorkspace(ctx, sizeof(int64_t) * batch_size * 7));
    int64_t *n_data_gpu = ptr_1,
            *m_data_gpu = ptr_1 + batch_size,
            *p_data_gpu = ptr_1 + 2 * batch_size,
            *A_off_gpu = ptr_1 + 3 * batch_size,
            *B_off_gpu = ptr_1 + 4 * batch_size,
            *C_off_gpu = ptr_1 + 5 * batch_size,
            *parallel_indices_gpu = ptr_1 + 6 * batch_size;
    // copy data from CPU to GPU;
    CUDA_CALL(cudaMemcpy(n_data_gpu, n_data, sizeof(int64_t) * batch_size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(m_data_gpu, m_data, sizeof(int64_t) * batch_size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(p_data_gpu, p_data, sizeof(int64_t) * batch_size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(A_off_gpu, A_off.data(), sizeof(int64_t) * batch_size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(B_off_gpu, B_off.data(), sizeof(int64_t) * batch_size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(C_off_gpu, C_off.data(), sizeof(int64_t) * batch_size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(
        parallel_indices_gpu, parallel_indices.data(), sizeof(int64_t) * parallel_batch_size, cudaMemcpyHostToDevice));
    // pad
    dim3 nblks(parallel_batch_size, (max_n + 31) / 32, (max_m + 31) / 32),
         nthrs(32, 32);
    if (transA) {
      std::swap(nblks.y, nblks.z);
    }
    CUDA_KERNEL_CALL((_PadKernel<DType>),
        nblks, nthrs, 0, thr_entry->stream,
        A_padded, A_data,
        parallel_indices_gpu,
        transA ? m_data_gpu: n_data_gpu,
        transA ? n_data_gpu: m_data_gpu,
        A_off_gpu,
        stride_a);
    nblks.y = (max_m + 31) / 32;
    nblks.z = (max_p + 31) / 32;
    if (transB) {
      std::swap(nblks.y, nblks.z);
    }
    CUDA_KERNEL_CALL((_PadKernel<DType>),
        nblks, nthrs, 0, thr_entry->stream,
        B_padded, B_data,
        parallel_indices_gpu,
        transB ? p_data_gpu: m_data_gpu,
        transB ? m_data_gpu: p_data_gpu,
        B_off_gpu,
        stride_b);
    // batched gemm
    DType alpha = 1., beta = 0.;
    int ldb = transB ? max_m: max_p,
        lda = transA ? max_n: max_m,
        ldc = max_p;
    CUBLAS_CALL(XgemmBatched<DType>(
        thr_entry->cublas_handle,
        transB ? CUBLAS_OP_T : CUBLAS_OP_N,
        transA ? CUBLAS_OP_T : CUBLAS_OP_N,
        max_p, max_n, max_m,
        &alpha,
        B_padded, ldb, stride_b,
        A_padded, lda, stride_a,
        &beta,
        C_padded, ldc, stride_c,
        parallel_batch_size
    ));
    // unpad
    nblks.y = (max_n + 31) / 32;
    nblks.z = (max_p + 31) / 32;
    CUDA_KERNEL_CALL((_UnpadKernel<DType>),
        nblks, nthrs, 0, thr_entry->stream,
        C_data, C_padded,
        parallel_indices_gpu,
        n_data_gpu, p_data_gpu,
        C_off_gpu,
        stride_c);
    // free allocated memory
    device->FreeWorkspace(ctx, ptr_0);
    device->FreeWorkspace(ctx, ptr_1);
  }

  // not parallel (or parallel w/ CUDA stream)
  for (int i = 0; i < batch_size; ++i) {
    int64_t ni = n_data[i],
            mi = m_data[i],
            pi = p_data[i];
    if (mask[i]) {
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
