/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/gather_mm.cu
 * \brief SDDMM C APIs and definitions.
 */
#include <dgl/array.h>
#include "./utils.h"
#include "./functor.cuh"

namespace dgl {
using namespace cuda;
namespace aten {


/*! \brief Call cuBLAS geam API for transpose operation for float and double. */
namespace {
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
}

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
template <typename Idx, typename DType>
__global__ void gatherMMUnsortedEKernel(
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
        int w_offset = etype[row] * in_len * out_len; // assume all weights are of same dim
        /* iterate over elements of a row of H */
        for (unsigned int i = 0; i < in_len; i++) {
            DType h_val =  H[row * in_len + i];
            /* iterate over elements of a row of W in parallel */
            for (unsigned int k = laneId; k < out_len; k += 32) {
                out[row * out_len + k] += h_val * W[w_offset + (i * out_len + k)];
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
    int n = w->shape[1]; // cols of B
    int k = h->shape[1]; // cols of A
    const IdType* E_per_rel_data = E_per_rel.Ptr<IdType>();
    IdType tot_num_rows = 0;
    for (int i = 0; i < num_rel; ++i)
        tot_num_rows += E_per_rel_data[i];

    const int ntx = 128;
    const int warp_size = 32;
    const int nbx =  ((tot_num_rows * warp_size + ntx - 1) / ntx);
    const dim3 nblks(nbx);
    const dim3 nthrs(ntx);
    CUDA_KERNEL_CALL((gatherMMUnsortedEKernel<IdType, DType>),
        nblks, nthrs, 0, thr_entry->stream,
        static_cast<DType*>(h->data),
        static_cast<DType*>(w->data),
        static_cast<DType*>(out->data),
        static_cast<IdType*>(etype->data),
        tot_num_rows,
        k, n);
    });
}

template <int XPU, typename IdType, int bits>
void gatherMM_SortedEtype(const NDArray h,
              const NDArray w,
              NDArray out,
              const NDArray E_per_rel,
              const NDArray h_etype) {
    SWITCH_BITS(bits, DType, {

        int64_t num_rel = E_per_rel.NumElements();
        int n = w->shape[1]; // cols of B
        int k = h->shape[1]; // cols of A = rows of B
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
            int m = E_per_rel_data[etype]; // rows of A
            CUBLAS_CALL(cublasGemm<DType>(
              thr_entry->cublas_handle,
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              n, m, k,
              &alpha,
              w_data + w_offset, ldb,
              h_data + h_offset, lda,
              &beta,
              out_data + out_offset, ldc));
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
          bool sortedE) {
    if (sortedE)  // similar to low-mem matmul
        gatherMM_SortedEtype<XPU, IdType, bits>(h, w, out, E_per_rel, etype);
    else  // similar to bmm without copying w to edges
        gatherMM_UnsortedEtype<XPU, IdType, bits>(h, w, out, E_per_rel, etype);
}

template void gatherMM<kDLGPU, int32_t, 16>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray E_per_rel, const NDArray etype, bool sortedE);
template void gatherMM<kDLGPU, int64_t, 16>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray E_per_rel, const NDArray etype, bool sortedE);
template void gatherMM<kDLGPU, int32_t, 32>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray E_per_rel, const NDArray etype, bool sortedE);
template void gatherMM<kDLGPU, int64_t, 32>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray E_per_rel, const NDArray etype, bool sortedE);
template void gatherMM<kDLGPU, int32_t, 64>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray E_per_rel, const NDArray etype, bool sortedE);
template void gatherMM<kDLGPU, int64_t, 64>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray E_per_rel, const NDArray etype, bool sortedE);

}  // namespace aten
}  // namespace dgl
