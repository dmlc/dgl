/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/gather_mm.cu
 * @brief GatherMM C APIs and definitions.
 */
#include <dgl/array.h>

#include <algorithm>  // std::swap

#include "./atomic.cuh"
#include "./functor.cuh"
#include "./utils.h"

namespace dgl {
using namespace cuda;
namespace aten {

namespace {

/** @brief Call cuBLAS GEMM API for dense matmul operation for float and double.
 */
template <typename DType>
cublasStatus_t cublasGemm(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const DType* alpha, const DType* A, int lda,
    const DType* B, int ldb, const DType* beta, DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <>
cublasStatus_t cublasGemm<__half>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const __half* alpha, const __half* A, int lda,
    const __half* B, int ldb, const __half* beta, __half* C, int ldc) {
  return cublasHgemm(
      handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

#if BF16_ENABLED
template <>
cublasStatus_t cublasGemm<__nv_bfloat16>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const __nv_bfloat16* alpha, const __nv_bfloat16* A,
    int lda, const __nv_bfloat16* B, int ldb, const __nv_bfloat16* beta,
    __nv_bfloat16* C, int ldc) {
  float alpha_float = __bfloat162float(*alpha);
  float beta_float = __bfloat162float(*beta);
  return cublasGemmEx(
      handle, transa, transb, m, n, k, &alpha_float, A, CUDA_R_16BF, lda, B,
      CUDA_R_16BF, ldb, &beta_float, C, CUDA_R_16BF, ldc, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
#endif  // BF16_ENABLED

template <>
cublasStatus_t cublasGemm<float>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta, float* C, int ldc) {
  return cublasSgemm(
      handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
cublasStatus_t cublasGemm<double>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double* alpha, const double* A, int lda,
    const double* B, int ldb, const double* beta, double* C, int ldc) {
  return cublasDgemm(
      handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

}  // namespace

namespace cuda {

/**
 * @note Each row of A multiplies a segment of matrix of B of dimension in_len *
 * outlen. One warp is assigned to process one row of A. Each WARP sequentially
 * multiplies one element of A and a row of B to compute partial result of the
 * output. A is loaded in shared memory in a coalesced way. Output matrix is
 * loaded in registers. B should get benefit from L2 cache.
 */
template <typename Idx, typename DType>
__global__ void GatherMMScatterKernel(
    const DType* __restrict__ A, const DType* __restrict__ B,
    DType* __restrict__ C, const Idx* __restrict__ idx_a,
    const Idx* __restrict__ idx_b, const Idx* __restrict__ idx_c,
    const int64_t num_rows, const int64_t in_len, const int64_t out_len) {
  unsigned int tId = threadIdx.x;
  unsigned int laneId = tId & 31;
  unsigned int gId = (blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int warpId = gId >> 5;
  unsigned int row = warpId;
  if (row < num_rows) {
    const unsigned int local_row =
        row & 3;  // hardcoded for TB size 128 (4 warps)
    const Idx cur_rowA = (idx_a) ? idx_a[row] : row;
    const Idx cur_rowB = (idx_b) ? idx_b[row] : row;
    const Idx cur_rowC = (idx_c) ? idx_c[row] : row;
    const Idx B_offset = cur_rowB * in_len * out_len;
    const int sh_a_tile = 64;
    __shared__ DType sh_A[4 * sh_a_tile];
    int a_tile = sh_a_tile;
    for (unsigned int k_start = 0; k_start < in_len; k_start += 64) {
      if ((in_len - k_start) < a_tile) a_tile = in_len - k_start;
      // Load A in shared mem in a coalesced way
      for (unsigned int l = laneId; l < a_tile; l += 32)
        sh_A[local_row * sh_a_tile + l] = A[cur_rowA * in_len + (k_start + l)];
      __syncwarp();

      for (unsigned int outloop = 0; outloop < out_len; outloop += 32) {
        DType out_reg = static_cast<DType>(0.0f);  // thread private
        const unsigned int l = laneId;
        if (l < out_len) {
          // iterate over elements of a row of A
          for (unsigned int i = 0; i < a_tile; i++) {
            const DType a_val = sh_A[local_row * sh_a_tile + i];
            // iterate over elements of a row of B in parallel
            out_reg +=
                a_val * B[B_offset + ((i + k_start) * out_len + (outloop + l))];
          }
          if (idx_c) {
            AtomicAdd(C + cur_rowC * out_len + (outloop + l), out_reg);
          } else {
            C[cur_rowC * out_len + (outloop + l)] += out_reg;
          }
        }
      }
    }
  }
}

/**
 * @note Output matrix is accumulated via atomic operations. Rest of the
 * strategies are similar to GatherMMKernel. One warp is assigned to process one
 * row of A. Each WARP sequentially multiplies one element of A and a row of B
 * to compute partial result of the output. A is loaded in shared memory in a
 * coalesced way. B should get benefit from L2 cache.
 */
template <typename Idx, typename DType>
__global__ void GatherMMScatterKernel2(
    const DType* __restrict__ A, const DType* __restrict__ B,
    DType* __restrict__ C, const Idx* __restrict__ idx_a,
    const Idx* __restrict__ idx_b, const Idx* __restrict__ idx_c,
    const int64_t num_rows, const int64_t in_len, const int64_t out_len) {
  unsigned int tId = threadIdx.x;
  unsigned int laneId = tId & 31;
  unsigned int gId = (blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int warpId = gId >> 5;
  unsigned int row = warpId;
  if (row < num_rows) {
    const unsigned int local_row =
        row & 3;  // hardcoded for TB size 128 (4 warps)
    const Idx row_a = (idx_a) ? idx_a[row] : row;
    const Idx row_b = (idx_b) ? idx_b[row] : row;
    const Idx row_c = (idx_c) ? idx_c[row] : row;
    const Idx C_offset = row_c * in_len * out_len;
    const int sh_a_tile = 64;
    __shared__ DType sh_A[4 * sh_a_tile];
    int a_tile = sh_a_tile;
    for (unsigned int k_start = 0; k_start < in_len; k_start += 64) {
      if ((in_len - k_start) < a_tile) a_tile = in_len - k_start;
      /* Load A in shared mem in a coalesced way */
      for (unsigned int l = laneId; l < a_tile; l += 32)
        sh_A[local_row * sh_a_tile + l] = A[row_a * in_len + (k_start + l)];
      __syncwarp();

      for (unsigned int outloop = 0; outloop < out_len; outloop += 32) {
        DType out_reg = static_cast<DType>(0.0f);  // thread private
        const unsigned int l = laneId;
        if (l < out_len) {
          const DType b_val = B[row_b * out_len + (outloop + l)];
          /* iterate over elements of a row of A */
          for (unsigned int i = 0; i < a_tile; i++) {
            const DType a_val = sh_A[local_row * sh_a_tile + i];
            const Idx C_idx =
                C_offset + ((i + k_start) * out_len + (outloop + l));
            AtomicAdd(C + C_idx, a_val * b_val);
          }
        }
      }
    }
  }
}

}  // namespace cuda

/**
 * @brief Implementation of Gather_mm operator. The input matrix A is
 *        expected to be sorted according to relation type.
 * @param A The input dense matrix of dimension m x k
 * @param B The input dense matrix of dimension k x n
 * @param C The output dense matrix of dimension m x n
 * @param seglen_A The input vector of size R. Each element
 *        is the length of segments of input ``A``
 * @param a_trans Matrix A to be transposed
 * @param b_trans Matrix B to be transposed
 */
template <int XPU, typename IdType, typename DType>
void SegmentMM(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans) {
  auto device = runtime::DeviceAPI::Get(A->ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const DType* A_data = A.Ptr<DType>();
  const DType* B_data = B.Ptr<DType>();
  const IdType* seglen_A_data = seglen_A.Ptr<IdType>();
  DType* C_data = C.Ptr<DType>();
  int64_t A_offset = 0, B_offset = 0, C_offset = 0;
  int64_t m, n, k;
  int64_t num_rel = seglen_A.NumElements();
  DType alpha = 1., beta = 0.;

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  if (!thr_entry->cublas_handle)
    CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
  CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle, stream));

  IdType m_offset = 0;
  for (IdType etype = 0; etype < num_rel; ++etype) {
    m = seglen_A_data[etype];  // rows of A
    CHECK_LE(m_offset + m, A->shape[0])
        << "Segment index out of bound of A->shape[0].";
    n = B->shape[2];  // cols of B
    k = B->shape[1];  // cols of A == rows of B
    int ldb = n, lda = k, ldc = n;
    cublasOperation_t transB = CUBLAS_OP_N;
    cublasOperation_t transA = CUBLAS_OP_N;
    if (b_trans) {
      transB = CUBLAS_OP_T;
      ldb = n, lda = n, ldc = k;
      std::swap(n, k);
    }
    CUBLAS_CALL(cublasGemm<DType>(
        thr_entry->cublas_handle, transB, transA, n, m, k, &alpha,
        B_data + B_offset, ldb, A_data + A_offset, lda, &beta,
        C_data + C_offset, ldc));
    A_offset += m * k;
    B_offset += k * n;
    C_offset += m * n;
    m_offset += m;
  }
}

template <int XPU, typename IdType, typename DType>
void SegmentMMBackwardB(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen) {
  auto device = runtime::DeviceAPI::Get(A->ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const DType* A_data = A.Ptr<DType>();
  const DType* dC_data = dC.Ptr<DType>();
  const IdType* seglen_data = seglen.Ptr<IdType>();
  DType* dB_data = dB.Ptr<DType>();
  int64_t A_offset = 0, dC_offset = 0, dB_offset = 0;
  int64_t m, n, k;
  int64_t num_rel = seglen.NumElements();
  DType alpha = 1., beta = 0.;

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  if (!thr_entry->cublas_handle)
    CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
  CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle, stream));

  IdType k_offset = 0;
  for (IdType etype = 0; etype < num_rel; ++etype) {
    m = dC->shape[1];
    n = A->shape[1];
    k = seglen_data[etype];
    CHECK_LE(k_offset + k, A->shape[0])
        << "Segement index out of bound of A->shape[0].";
    int lddC = m, ldA = n, lddB = m;
    cublasOperation_t trans_dC = CUBLAS_OP_N;
    cublasOperation_t trans_A = CUBLAS_OP_T;
    CUBLAS_CALL(cublasGemm<DType>(
        thr_entry->cublas_handle, trans_dC, trans_A, m, n, k, &alpha,
        dC_data + dC_offset, lddC, A_data + A_offset, ldA, &beta,
        dB_data + dB_offset, lddB));
    dC_offset += m * k;
    A_offset += n * k;
    dB_offset += m * n;
    k_offset += k;
  }
}

/**
 * @brief Implementation of Gather_mm operator. The input matrix A is
 *        expected to be sorted according to relation type.
 * @param A The input dense matrix of dimension m x k
 * @param B The input dense matrix of dimension k x n
 * @param C The output dense matrix of dimension m x n
 * @param idx_a The input vector to gather left hand operand on
 * @param idx_b The input vector to gather right hand operand on
 */

template <int XPU, typename IdType, typename DType>
void GatherMM(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b) {
  auto device = runtime::DeviceAPI::Get(A->ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  int64_t out_len = B->shape[2];  // cols of B
  int64_t in_len = A->shape[1];   // cols of A
  const int64_t tot_num_rows = A->shape[0];
  const int ntx = 128;
  const int warp_size = 32;
  const int nbx = ((tot_num_rows * warp_size + ntx - 1) / ntx);
  const dim3 nblks(nbx);
  const dim3 nthrs(ntx);
  CUDA_KERNEL_CALL(
      (cuda::GatherMMScatterKernel<IdType, DType>), nblks, nthrs, 0, stream,
      A.Ptr<DType>(), B.Ptr<DType>(), C.Ptr<DType>(), idx_a.Ptr<IdType>(),
      idx_b.Ptr<IdType>(), nullptr, tot_num_rows, in_len, out_len);
}

/**
 * @brief Implementation of Gather_mm operator. The input matrix A is
 *        expected to be sorted according to relation type.
 * @param A The input dense matrix of dimension m x k
 * @param B The input dense matrix of dimension k x n
 * @param C The output dense matrix of dimension m x n
 * @param idx_a The input vector to gather left hand operand on
 * @param idx_b The input vector to gather right hand operand on
 * @param idx_c The input vector to gather output operand on
 * @param num_rel The number of idx types in idx_b
 * @param a_trans Matrix A to be transposed
 * @param b_trans Matrix B to be transposed
 */
template <int XPU, typename IdType, typename DType>
void GatherMMScatter(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c) {
  auto device = runtime::DeviceAPI::Get(A->ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const IdType* idx_c_data = idx_c.Ptr<IdType>();
  int64_t out_len = (B->ndim == 2) ? B->shape[1] : B->shape[2];  // cols of B
  int64_t in_len = A->shape[1];                                  // cols of A
  int64_t tot_num_rows = A->shape[0];
  const int ntx = 128;
  const int warp_size = 32;
  const int nbx = ((tot_num_rows * warp_size + ntx - 1) / ntx);
  const dim3 nblks(nbx);
  const dim3 nthrs(ntx);
  if (B->ndim == 3) {
    CUDA_KERNEL_CALL(
        (cuda::GatherMMScatterKernel<IdType, DType>), nblks, nthrs, 0, stream,
        A.Ptr<DType>(), B.Ptr<DType>(), C.Ptr<DType>(), idx_a.Ptr<IdType>(),
        idx_b.Ptr<IdType>(), idx_c.Ptr<IdType>(), tot_num_rows, in_len,
        out_len);
  } else {
    // Custom kernel for W_grad[idx_c[i]] = H^T[i] * C.grad[i]
    // This kernel accesses rows of A in a transposed way w/o explicitly
    // converting A
    CUDA_KERNEL_CALL(
        (cuda::GatherMMScatterKernel2<IdType, DType>), nblks, nthrs, 0, stream,
        A.Ptr<DType>(), B.Ptr<DType>(), C.Ptr<DType>(), idx_a.Ptr<IdType>(),
        idx_b.Ptr<IdType>(), idx_c.Ptr<IdType>(), tot_num_rows, in_len,
        out_len);
  }
}

template void GatherMM<kDGLCUDA, int32_t, __half>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b);
template void GatherMM<kDGLCUDA, int64_t, __half>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b);
#if BF16_ENABLED
template void GatherMM<kDGLCUDA, int32_t, __nv_bfloat16>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b);
template void GatherMM<kDGLCUDA, int64_t, __nv_bfloat16>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b);
#endif  // BF16_ENABLED
template void GatherMM<kDGLCUDA, int32_t, float>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b);
template void GatherMM<kDGLCUDA, int64_t, float>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b);
template void GatherMM<kDGLCUDA, int32_t, double>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b);
template void GatherMM<kDGLCUDA, int64_t, double>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b);

template void GatherMMScatter<kDGLCUDA, int32_t, __half>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c);
template void GatherMMScatter<kDGLCUDA, int64_t, __half>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c);
#if BF16_ENABLED
template void GatherMMScatter<kDGLCUDA, int32_t, __nv_bfloat16>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c);
template void GatherMMScatter<kDGLCUDA, int64_t, __nv_bfloat16>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c);
#endif  // BF16_ENABLED
template void GatherMMScatter<kDGLCUDA, int32_t, float>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c);
template void GatherMMScatter<kDGLCUDA, int64_t, float>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c);
template void GatherMMScatter<kDGLCUDA, int32_t, double>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c);
template void GatherMMScatter<kDGLCUDA, int64_t, double>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c);

template void SegmentMM<kDGLCUDA, int32_t, __half>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans);
template void SegmentMM<kDGLCUDA, int64_t, __half>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans);
#if BF16_ENABLED
template void SegmentMM<kDGLCUDA, int32_t, __nv_bfloat16>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans);
template void SegmentMM<kDGLCUDA, int64_t, __nv_bfloat16>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans);
#endif  // BF16_ENABLED
template void SegmentMM<kDGLCUDA, int32_t, float>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans);
template void SegmentMM<kDGLCUDA, int64_t, float>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans);
template void SegmentMM<kDGLCUDA, int32_t, double>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans);
template void SegmentMM<kDGLCUDA, int64_t, double>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans);

template void SegmentMMBackwardB<kDGLCUDA, int32_t, __half>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);
template void SegmentMMBackwardB<kDGLCUDA, int64_t, __half>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);
#if BF16_ENABLED
template void SegmentMMBackwardB<kDGLCUDA, int32_t, __nv_bfloat16>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);
template void SegmentMMBackwardB<kDGLCUDA, int64_t, __nv_bfloat16>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);
#endif  // BF16_ENABLED
template void SegmentMMBackwardB<kDGLCUDA, int32_t, float>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);
template void SegmentMMBackwardB<kDGLCUDA, int64_t, float>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);
template void SegmentMMBackwardB<kDGLCUDA, int32_t, double>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);
template void SegmentMMBackwardB<kDGLCUDA, int64_t, double>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);

}  // namespace aten
}  // namespace dgl
