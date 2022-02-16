/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/gather_mm.cu
 * \brief GatherMM C APIs and definitions.
 */
#include <dgl/array.h>
#include <algorithm>  // std::swap
#include "./utils.h"
#include "./functor.cuh"
#include "./atomic.cuh"

namespace dgl {
using namespace cuda;
namespace aten {

namespace {

/*! \brief Call cuBLAS geam API for transpose operation for float and double. */
template <typename DType>
cublasStatus_t Xgeam(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const DType* alpha, const DType* A, int lda,
    const DType* beta, const DType* B, int ldb,
    DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <>
cublasStatus_t Xgeam<float>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const float* alpha, const float* A, int lda,
    const float* beta, const float* B, int ldb,
    float* C, int ldc) {
  return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda,
      beta, B, ldb, C, ldc);
}

template <>
cublasStatus_t Xgeam<double>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const double* alpha, const double* A, int lda,
    const double* beta, const double* B, int ldb,
    double* C, int ldc) {
  return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda,
      beta, B, ldb, C, ldc);
}

/*! \brief Call cuBLAS GEMM API for dense matmul operation for float and double. */
template <typename DType>
cublasStatus_t cublasGemm(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k,
    const DType* alpha, const DType* A, int lda,
    const DType* B, int ldb, const DType* beta,
    DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <>
cublasStatus_t cublasGemm<float>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k,
    const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta,
    float* C, int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda,
      B, ldb, beta, C, ldc);
}

template <>
cublasStatus_t cublasGemm<double>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k,
    const double* alpha, const double* A, int lda,
    const double* B, int ldb, const double* beta,
    double* C, int ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda,
      B, ldb, beta, C, ldc);
}

/*
 * \brief Tranpose the input matrix.
 * \param row number of rows of input matrix.
 * \param col number of columns of input matrix.
 */
template <typename DType>
void _Transpose(cublasHandle_t handle,
                const DType* in, DType* out,
                int row, int col) {
  DType alpha = 1., beta = 0.;
  CUBLAS_CALL(Xgeam<DType>(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      row, col,
      &alpha, in, col,
      &beta, nullptr, row,
      out, row));
}

}  // namespace

namespace cuda {

/* \Note Each row of A multiplies a segment of matrix of B of dimension in_len * outlen.
  One warp is assigned to process one row of A. Each WARP sequentially multiplies
  one element of A and a row of B to compute partial result of the output. A
  is loaded in shared memory in a coalesced way. Output matrix is loaded in
  registers. B should get benefit from L2 cache.
*/
template <typename Idx, typename DType>
__global__ void gatherMMKernel(
    const DType* __restrict__ A,
    const DType* __restrict__ B,
    DType* __restrict__ C,
    const Idx* __restrict__ idx_a,
    const Idx* __restrict__ idx_b,
    int64_t num_rows,
    int64_t in_len, int64_t out_len) {
    unsigned int tId = threadIdx.x;
    unsigned int laneId = tId & 31;
    unsigned int gId = (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int warpId = gId >> 5;
    unsigned int row = warpId;
    if (row < num_rows) {
        unsigned int local_row = row & 3;  // hardcoded for TB size 128 (4 warps)
        Idx cur_rowA = (idx_a) ? idx_a[row] : row;
        Idx cur_rowB = (idx_b) ? idx_b[row] : row / in_len;
        Idx B_offset = cur_rowB * in_len * out_len;
        const int sh_a_tile = 64;
        __shared__ DType sh_A[4 * sh_a_tile];
        int a_tile = sh_a_tile;
        for (unsigned int k_start = 0; k_start < in_len; k_start += 64) {
            if ((in_len - k_start) < a_tile) a_tile = in_len - k_start;
            /* Load A in shared mem in a coalesced way */
            for (unsigned int l = laneId; l < a_tile; l += 32)
                sh_A[local_row * sh_a_tile + l] = A[cur_rowA * in_len + (k_start + l)];
            __syncwarp();

            for (unsigned int outloop = 0; outloop < out_len; outloop +=32) {
                DType out_reg = 0;  // thread private
                const unsigned int l = laneId;
                if (l < out_len) {
                    /* iterate over elements of a row of A */
                    for (unsigned int i = 0; i < a_tile; i++) {
                        const DType a_val =  sh_A[local_row * sh_a_tile + i];
                        /* iterate over elements of a row of B in parallel */
                        out_reg += a_val * B[B_offset + ((i + k_start) * out_len + (outloop + l))];
                    }
                    C[row * out_len + (outloop + l)] += out_reg;
                }
            }
        }
    }
}

/* \Note Output matrix is accumulated via atomic operations. Rest of the strategies
  are similar to gatherMMKernel. One warp is assigned to process one row of A. Each
  WARP sequentially multiplies one element of A and a row of B to compute partial
  result of the output. A is loaded in shared memory in a coalesced way. B should
  get benefit from L2 cache.
*/
template <typename Idx, typename DType>
__global__ void gatherMMScatterKernel(
    const DType* __restrict__ A,
    const DType* __restrict__ B,
    DType* __restrict__ C,
    const Idx* __restrict__ idx_a,
    const Idx* __restrict__ idx_b,
    const Idx* __restrict__ idx_c,
    int64_t num_rows,
    int64_t in_len, int64_t out_len) {
    unsigned int tId = threadIdx.x;
    unsigned int laneId = tId & 31;
    unsigned int gId = (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int warpId = gId >> 5;
    unsigned int row = warpId;
    if (row < num_rows) {
        unsigned int local_row = row & 3;  // hardcoded for TB size 128 (4 warps)
        unsigned int row_a = (idx_a) ? idx_a[row] : row;
        unsigned int row_b = (idx_b) ? idx_b[row] : row;
        Idx C_offset = (idx_c) ? idx_c[row] * in_len * out_len : 0;
        const int sh_a_tile = 64;
        __shared__ DType sh_A[4 * sh_a_tile];
        int a_tile = sh_a_tile;
        for (unsigned int k_start = 0; k_start < in_len; k_start += 64) {
            if ((in_len - k_start) < a_tile) a_tile = in_len - k_start;
            /* Load A in shared mem in a coalesced way */
            for (unsigned int l = laneId; l < a_tile; l += 32)
                sh_A[local_row * sh_a_tile + l] = A[row_a * in_len + (k_start + l)];
            __syncwarp();

            for (unsigned int outloop = 0; outloop < out_len; outloop +=32) {
                DType out_reg = 0;  // thread private
                const unsigned int l = laneId;
                if (l < out_len) {
                    const DType b_val = B[row_b * out_len + (outloop + l)];
                    /* iterate over elements of a row of A */
                    for (unsigned int i = 0; i < a_tile; i++) {
                        const DType a_val = sh_A[local_row * sh_a_tile + i];
                        const Idx C_idx = C_offset + ((i + k_start) * out_len + (outloop + l));
                        atomicAdd(reinterpret_cast<float*>(&C[C_idx]),
                            static_cast<float>(a_val * b_val));
                    }
                }
            }
        }
    }
}


/* \brief Implementation of GatherMM operator. The indices of A (or B)
 * are looked up from idx_a (or idx_b) when defined.
 */
template <int XPU, typename IdType, int bits>
void gatherMM(const NDArray A,
              const NDArray B,
              NDArray C,
              const NDArray idx_a,
              const NDArray idx_b,
              int64_t num_rel) {
    SWITCH_BITS(bits, DType, {
        auto device = runtime::DeviceAPI::Get(A->ctx);
        auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        const DType *A_data = A.Ptr<DType>();
        const DType *B_data = B.Ptr<DType>();
        int64_t out_len = B->shape[1];  // cols of B
        int64_t in_len = A->shape[1];  // cols of A
        if (!thr_entry->cublas_handle)
            CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
        CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle,
            thr_entry->stream));
        int64_t tot_num_rows = A->shape[0];
        const int ntx = 128;
        const int warp_size = 32;
        const int nbx =  ((tot_num_rows * warp_size + ntx - 1) / ntx);
        const dim3 nblks(nbx);
        const dim3 nthrs(ntx);
        CUDA_KERNEL_CALL((gatherMMKernel<IdType, DType>),
            nblks, nthrs, 0, thr_entry->stream,
            static_cast<DType*>(A->data),
            static_cast<DType*>(B->data),
            static_cast<DType*>(C->data),
            static_cast<IdType*>(idx_a->data),
            static_cast<IdType*>(idx_b->data),
            tot_num_rows,
            in_len, out_len);
    });
}

/* \brief Implementation of GatherMM operator. The indices of A (or B or C)
 * are looked up from idx_a (or idx_b or idx_c) when defined.
 */
template <int XPU, typename IdType, int bits>
void gatherMM_scatter(const NDArray A,
              const NDArray B,
              NDArray C,
              const NDArray idx_a,
              const NDArray idx_b,
              const NDArray idx_c,
              int num_rel, bool a_trans, bool b_trans) {
    SWITCH_BITS(bits, DType, {
        auto device = runtime::DeviceAPI::Get(A->ctx);
        auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        const IdType *idx_c_data = idx_c.Ptr<IdType>();
        int64_t out_len = B->shape[1];  // cols of B
        int64_t in_len = A->shape[1];  // cols of A
        if (!thr_entry->cublas_handle)
            CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
        CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle,
            thr_entry->stream));
        DType* B_trans_data = nullptr;
        if (b_trans) {
            int64_t B_offset = 0;
            const DType *B_data = B.Ptr<DType>();
            in_len = B->shape[0]/num_rel;
            B_trans_data = static_cast<DType*>(device->AllocWorkspace \
                (B->ctx, B->shape[0] * B->shape[1] * sizeof(DType)));
            // tranpose B per relation
            for (int rel = 0; rel < num_rel; ++rel) {
                _Transpose(thr_entry->cublas_handle, B_data + B_offset,
                    B_trans_data + B_offset, in_len, out_len);
                B_offset += in_len * out_len;
            }
            std::swap(in_len, out_len);
        }
        int64_t tot_num_rows = A->shape[0];
        const int ntx = 128;
        const int warp_size = 32;
        const int nbx =  ((tot_num_rows * warp_size + ntx - 1) / ntx);
        const dim3 nblks(nbx);
        const dim3 nthrs(ntx);

        if (idx_c_data) {
            // Custom kernel for W_grad[idx_c[i]] = H^T[i] * C.grad[i]
            // This kernel accesses rows of A in a transposed way w/o explicitly converting A
            CUDA_KERNEL_CALL((gatherMMScatterKernel<IdType, DType>),
                nblks, nthrs, 0, thr_entry->stream,
                static_cast<DType*>(A->data),
                static_cast<DType*>(B->data),
                static_cast<DType*>(C->data),
                static_cast<IdType*>(idx_a->data),
                static_cast<IdType*>(idx_b->data),
                static_cast<IdType*>(idx_c->data),
                tot_num_rows,
                in_len, out_len);
        } else {  // use generic gather_mm
                CUDA_KERNEL_CALL((gatherMMKernel<IdType, DType>),
                    nblks, nthrs, 0, thr_entry->stream,
                    static_cast<DType*>(A->data),
                    (b_trans) ? B_trans_data : static_cast<DType*>(B->data),
                    static_cast<DType*>(C->data),
                    static_cast<IdType*>(idx_a->data),
                    static_cast<IdType*>(idx_b->data),
                    tot_num_rows,
                    in_len, out_len);
        }
        if (b_trans)
            device->FreeWorkspace(B->ctx, B_trans_data);
    });
}

}  // namespace cuda

/* \brief Implementation of SegmentMM operator. Each segment calls cuBLAS
 * GEMM operator to multiply segment of A and B. When A or B needs to be
 * tranposed, cuBLAS GEMM switches it's transpose parameter (CUBLAS_OP_T).
 */
template <int XPU, typename IdType, int bits>
void segment_mm(const NDArray A,
              const NDArray B,
              NDArray C,
              const NDArray seglen_A,
              bool a_trans, bool b_trans) {
    SWITCH_BITS(bits, DType, {
        auto device = runtime::DeviceAPI::Get(A->ctx);
        const DType *A_data = A.Ptr<DType>();
        const DType *B_data = B.Ptr<DType>();
        const IdType* seglen_A_data = seglen_A.Ptr<IdType>();
        DType *C_data = C.Ptr<DType>();
        int64_t A_offset = 0, B_offset = 0, C_offset = 0;
        int64_t m, n, k;
        int64_t num_rel = seglen_A.NumElements();
        DType alpha = 1., beta = 0.;

        auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        if (!thr_entry->cublas_handle)
            CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
        CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle,
            thr_entry->stream));

        for (int etype = 0; etype < num_rel; ++etype) {
            IdType B_dim1 = B->shape[0] / num_rel;
            assert((a_trans) ? seglen_A_data[etype] : A->shape[1] ==  \
                (b_trans) ? B->shape[1] : B_dim1);
            m = seglen_A_data[etype];  // rows of A
            n = B->shape[1];  // cols of B
            k = A->shape[1];  // cols of A == rows of B
            int ldb = n, lda = k, ldc = n;
            cublasOperation_t transB = CUBLAS_OP_N;
            cublasOperation_t transA = CUBLAS_OP_N;
            if (a_trans) {
                transA = CUBLAS_OP_T;
                ldb = n, lda = k, ldc = n;
                std::swap(m, k);
            }
            if (b_trans) {
                transB = CUBLAS_OP_T;
                k = B_dim1;
                ldb = n, lda = n, ldc = k;
                std::swap(n, k);
            }
            CUBLAS_CALL(cublasGemm<DType>(
                thr_entry->cublas_handle,
                transB,
                transA,
                n, m, k,
                &alpha,
                B_data + B_offset, ldb,
                A_data + A_offset, lda,
                &beta,
                C_data + C_offset, ldc));
            A_offset += m * k;
            B_offset += k * n;
            C_offset += m * n;
        }
    });
}

/*!
 * \brief Implementation of Gather_mm operator. The input matrix A is
 *        expected to be sorted according to relation type.
 * \param A The input dense matrix of dimension m x k
 * \param B The input dense matrix of dimension k x n
 * \param C The output dense matrix of dimension m x n
 * \param seglen_A The input vector of size R. Each element
 *        is the length of segments of input ``A``
 * \param a_trans Matrix A to be transposed
 * \param b_trans Matrix B to be transposed
 */
template <int XPU, typename IdType, int bits>
void segmentMM(const NDArray A,
          const NDArray B,
          NDArray C,
          const NDArray seglen_A,
          bool a_trans, bool b_trans) {
    segment_mm<XPU, IdType, bits>(A, B, C, seglen_A, a_trans, b_trans);
}

/*!
 * \brief Implementation of Gather_mm operator. The input matrix A is
 *        expected to be sorted according to relation type.
 * \param A The input dense matrix of dimension m x k
 * \param B The input dense matrix of dimension k x n
 * \param C The output dense matrix of dimension m x n
 * \param idx_a The input vector to gather left hand operand on
 * \param idx_b The input vector to gather right hand operand on
 * \param num_rel The number of idx types in idx_b
 */
template <int XPU, typename IdType, int bits>
void gatherMM(const NDArray A,
          const NDArray B,
          NDArray C,
          const NDArray idx_a,
          const NDArray idx_b,
          const int num_rel) {
    cuda::gatherMM<XPU, IdType, bits>(A, B, C, idx_a, idx_b, num_rel);
}

/*!
 * \brief Implementation of Gather_mm operator. The input matrix A is
 *        expected to be sorted according to relation type.
 * \param A The input dense matrix of dimension m x k
 * \param B The input dense matrix of dimension k x n
 * \param C The output dense matrix of dimension m x n
 * \param idx_a The input vector to gather left hand operand on
 * \param idx_b The input vector to gather right hand operand on
 * \param idx_c The input vector to gather output operand on
 * \param num_rel The number of idx types in idx_b
 * \param a_trans Matrix A to be transposed
 * \param b_trans Matrix B to be transposed
 */
template <int XPU, typename IdType, int bits>
void gatherMM_scatter(const NDArray A,
          const NDArray B,
          NDArray C,
          const NDArray idx_a,
          const NDArray idx_b,
          const NDArray idx_c,
          const int num_rel,
          bool a_trans, bool b_trans) {
    cuda::gatherMM_scatter<XPU, IdType, bits>(A, B, C, idx_a, idx_b, idx_c,
        num_rel, a_trans, b_trans);
}


template void gatherMM<kDLGPU, int32_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const int num_rel);
template void gatherMM<kDLGPU, int64_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const int num_rel);
template void gatherMM<kDLGPU, int32_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const int num_rel);
template void gatherMM<kDLGPU, int64_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const int num_rel);
template void gatherMM<kDLGPU, int32_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const int num_rel);
template void gatherMM<kDLGPU, int64_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const int num_rel);

template void gatherMM_scatter<kDLGPU, int32_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c,
    const int num_rel, bool a_trans, bool b_trans);
template void gatherMM_scatter<kDLGPU, int64_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c,
    const int num_rel, bool a_trans, bool b_trans);
template void gatherMM_scatter<kDLGPU, int32_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c,
    const int num_rel, bool a_trans, bool b_trans);
template void gatherMM_scatter<kDLGPU, int64_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c,
    const int num_rel, bool a_trans, bool b_trans);
template void gatherMM_scatter<kDLGPU, int32_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c,
    const int num_rel, bool a_trans, bool b_trans);
template void gatherMM_scatter<kDLGPU, int64_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c,
    const int num_rel, bool a_trans, bool b_trans);

template void segmentMM<kDLGPU, int32_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void segmentMM<kDLGPU, int64_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void segmentMM<kDLGPU, int32_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void segmentMM<kDLGPU, int64_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void segmentMM<kDLGPU, int32_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void segmentMM<kDLGPU, int64_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);

}  // namespace aten
}  // namespace dgl
