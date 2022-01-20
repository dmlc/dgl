/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/gather_mm.cu
 * \brief SDDMM C APIs and definitions.
 */
#include <algorithm>    // std::swap
#include <dgl/array.h>
#include "./utils.h"
#include "./functor.cuh"

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
}  // namespace

/* Idea 1: tranpose W and compute dot product of each row vector of H
   and column vector of W. Reuse in H.
   Idea 2: multiply 1 element of H with a row of W and compute partial
   result of Out
*/

/* Implementation of Idea 2
One TB for one row of H. One WARP will multiply one element of H and
a row of W to compute partial result of 1 row of Out. H is loaded once,
W is used once. Only resuse is from Out which is put is shared
*/

/* uses shared mem to store output */
template <typename Idx, typename DType>
__global__ void gatherMMUnsortedEKernel_outInShmem(
    const DType* __restrict__ H,
    const DType* __restrict__ W,
    DType* __restrict__ out,
    const Idx* __restrict__ etype,
    int64_t num_rows,
    int64_t in_len, int64_t out_len) {

    unsigned int tId = threadIdx.x;
    unsigned int laneId = tId & 31;
    unsigned int gId = (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int warpId = gId >> 5;
    unsigned int row = warpId;
    if (row < num_rows) {
        unsigned int local_row = row & 3;  // hardcoded for TB size 128 (4 warps)
        // TODO(Israt): __shared__ does not take typename. Make compatible with double
        extern __shared__ float sh_out[];
        // global to shared
        for (unsigned int k = laneId; k < out_len; k += 32)
            sh_out[local_row * out_len + k] = 0;
        __syncthreads();

        int w_offset = etype[row] * in_len * out_len;  // assume all weights are of same dim
        /* iterate over elements of a row of H */
        for (unsigned int i = 0; i < in_len; i++) {
            DType h_val =  H[row * in_len + i];
            /* iterate over elements of a row of W in parallel */
            for (unsigned int k = laneId; k < out_len; k += 32) {
                sh_out[local_row * out_len + k] += h_val * W[w_offset + (i * out_len + k)];
                // out[row * out_len + k] += h_val * W[w_offset + (i * out_len + k)];
            }
        }
        __syncthreads();
        for (unsigned int k = laneId; k < out_len; k += 32) {
            out[row * out_len + k] = sh_out[local_row * out_len + k];
        }
    }
}

/* uses registers to store output */
template <typename Idx, typename DType>
__global__ void gatherMMUnsortedEKernel_outInReg(
    const DType* __restrict__ H,
    const DType* __restrict__ W,
    DType* __restrict__ out,
    const Idx* __restrict__ etype,
    int64_t num_rows,
    int64_t in_len, int64_t out_len) {

    unsigned int tId = threadIdx.x;
    unsigned int laneId = tId & 31;
    unsigned int gId = (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int warpId = gId >> 5;
    unsigned int row = warpId;
    if (row < num_rows) {
        int w_offset = etype[row] * in_len * out_len;  // assume all weights are of same dim
        /* iterate over elements of a row of H */
        for (unsigned int k_outloop = 0; k_outloop < out_len; k_outloop +=32) {
            float out_reg = 0;  // thread private
            unsigned int k = laneId;
            if (k < out_len) {
                for (unsigned int i = 0; i < in_len; i++) {
                    DType h_val =  H[row * in_len + i];
                    /* iterate over elements of a row of W in parallel */
                    out_reg += h_val * W[w_offset + (i * out_len + (k_outloop + k))];
                }
                out[row * out_len + (k_outloop + k)] = out_reg;
            }
        }
    }
}

/* uses registers to store output and shared mem to store input H
 Following version supports in_len <= 64. Change const int sh_h_tile to support
 larger in_len.
 */
template <typename Idx, typename DType>
__global__ void gatherMMUnsortedEKernel_optimal(
    const DType* __restrict__ H,
    const DType* __restrict__ W,
    DType* __restrict__ out,
    const Idx* __restrict__ etype,
    int64_t num_rows,
    int64_t in_len, int64_t out_len) {

    unsigned int tId = threadIdx.x;
    unsigned int laneId = tId & 31;
    unsigned int gId = (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int warpId = gId >> 5;
    unsigned int row = warpId;
    if (row < num_rows) {
        unsigned int local_row = row & 3;  // hardcoded for TB size 128 (4 warps)
        // TODO(Israt): Support inlen > 64
        const int sh_h_tile = 64;
        // TODO(Israt): Fix for inlen < 64.
        __shared__ float sh_H[4 * sh_h_tile];
        int h_tile = sh_h_tile;
        if (in_len < h_tile) h_tile = in_len;
        /* Load H in shared mem in a coalesced way */
        for (int h_outloop = 0; h_outloop < in_len; h_outloop += h_tile) {
            for (unsigned int l = laneId; l < h_tile; l += 32)
                sh_H[local_row * h_tile + l] = H[row * in_len + (h_outloop + l)];
            __syncthreads();

            int w_offset = etype[row] * in_len * out_len;  // assume all weights are of same dim
            /* iterate over elements of a row of H */
            for (unsigned int k_outloop = 0; k_outloop < out_len; k_outloop +=32) {
                float out_reg = 0;  // thread private
                unsigned int k = laneId;
                if (k < out_len) {
                    for (unsigned int i = 0; i < h_tile; i++) {
                        DType h_val =  sh_H[local_row * h_tile + i];
                        /* iterate over elements of a row of W in parallel */
                        out_reg += h_val * W[w_offset + (i * out_len + (k_outloop + k))];
                    }
                    out[row * out_len + (k_outloop + k)] += out_reg;
                }
            }
        }
    }
}

template <int XPU, typename IdType, int bits>
void gatherMM_UnsortedEtype(const NDArray h,
              const NDArray w,
              NDArray out,
              const NDArray E_per_rel,
              const NDArray etype) {
    SWITCH_BITS(bits, DType, {
        auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        int64_t num_rel = E_per_rel.NumElements();
        int n = w->shape[1];  // cols of B
        int k = h->shape[1];  // cols of A
        const IdType* E_per_rel_data = E_per_rel.Ptr<IdType>();
        IdType tot_num_rows = 0;
        for (int i = 0; i < num_rel; ++i)
            tot_num_rows += E_per_rel_data[i];

        const int ntx = 128;
        const int warp_size = 32;
        const int nbx =  ((tot_num_rows * warp_size + ntx - 1) / ntx);
        const dim3 nblks(nbx);
        const dim3 nthrs(ntx);
        const int warp_per_block = 4;  // ntx / warp_size;
        if (k <= 64) {
            CUDA_KERNEL_CALL((gatherMMUnsortedEKernel_optimal<IdType, DType>),
                nblks, nthrs, 0, thr_entry->stream,
                static_cast<DType*>(h->data),
                static_cast<DType*>(w->data),
                static_cast<DType*>(out->data),
                static_cast<IdType*>(etype->data),
                tot_num_rows,
                k, n);
        } else {  // Still can use the  *_optimal kernel if `const int sh_h_tile is changed
            CUDA_KERNEL_CALL((gatherMMUnsortedEKernel_outInReg<IdType, DType>),
                nblks, nthrs, 0, thr_entry->stream,
                static_cast<DType*>(h->data),
                static_cast<DType*>(w->data),
                static_cast<DType*>(out->data),
                static_cast<IdType*>(etype->data),
                tot_num_rows,
                k, n);
        }
    });
}

template <int XPU, typename IdType, int bits>
void gatherMM_SortedEtype(const NDArray h,
              const NDArray w,
              NDArray out,
              const NDArray E_per_rel,
              const NDArray h_etype,
              bool H_trans) {
    SWITCH_BITS(bits, DType, {
        auto device = runtime::DeviceAPI::Get(h->ctx);
        int64_t num_rel = E_per_rel.NumElements();
        int n = w->shape[1];  // cols of B
        int k = h->shape[1];  // cols of A = rows of B
        const DType *h_data = h.Ptr<DType>();
        const DType *w_data = w.Ptr<DType>();
        const IdType* E_per_rel_data = E_per_rel.Ptr<IdType>();
        DType *out_data = out.Ptr<DType>();

        int64_t h_offset = 0;
        int64_t w_offset = 0;
        int64_t out_offset = 0;
        DType alpha = 1., beta = 0.;

        auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        if (!thr_entry->cublas_handle)
            CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
        CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle,
            thr_entry->stream));
        int ldb = n,
            lda = k,
            ldc = n;
        for (int etype = 0; etype < num_rel; ++etype) {
            int m = E_per_rel_data[etype];  // rows of A
            k = h->shape[1];
            DType* h_trans_data = nullptr;
            if (H_trans) {
                // allocate matrix for temporary transposed output
                h_trans_data = static_cast<DType*>(device->AllocWorkspace \
                    (h->ctx, m * k * sizeof(DType)));
                CUBLAS_CALL(Xgeam<DType>(
                    thr_entry->cublas_handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    m, k,
                    &alpha, h_data + h_offset, k,
                    &beta, nullptr, m,
                    h_trans_data, m));
                std::swap(m, k);
            }
            CUBLAS_CALL(cublasGemm<DType>(
                thr_entry->cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n, m, k,
                &alpha,
                w_data + w_offset, n,
                (H_trans) ? h_trans_data : h_data + h_offset, k,
                &beta,
                out_data + out_offset, n));
            if (H_trans)
                device->FreeWorkspace(h->ctx, h_trans_data);
            h_offset += m * k;
            w_offset += k * n;
            out_offset += m * n;
        }
    });
}

template <int XPU, typename IdType, int bits>
void gatherMM(const NDArray h,
          const NDArray w,
          NDArray out,
          const NDArray E_per_rel,
          const NDArray etype,
          bool sortedE, bool H_trans, bool w_trans) {
    if (sortedE)  // similar to low-mem matmul
        gatherMM_SortedEtype<XPU, IdType, bits>(h, w, out, E_per_rel, etype, H_trans);
    else  // similar to bmm without copying w to edges
        gatherMM_UnsortedEtype<XPU, IdType, bits>(h, w, out, E_per_rel, etype);
}

template void gatherMM<kDLGPU, int32_t, 16>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray E_per_rel, const NDArray etype,
    bool sortedE, bool H_trans,  bool W_trans);
template void gatherMM<kDLGPU, int64_t, 16>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray E_per_rel, const NDArray etype,
    bool sortedE, bool H_trans,  bool W_trans);
template void gatherMM<kDLGPU, int32_t, 32>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray E_per_rel, const NDArray etype,
    bool sortedE, bool H_trans,  bool W_trans);
template void gatherMM<kDLGPU, int64_t, 32>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray E_per_rel, const NDArray etype,
    bool sortedE, bool H_trans,  bool W_trans);
template void gatherMM<kDLGPU, int32_t, 64>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray E_per_rel, const NDArray etype,
    bool sortedE, bool H_trans,  bool W_trans);
template void gatherMM<kDLGPU, int64_t, 64>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray E_per_rel, const NDArray etype,
    bool sortedE, bool H_trans,  bool W_trans);

}  // namespace aten
}  // namespace dgl
