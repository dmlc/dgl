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

/*!
 * \brief CUDA implementation of g-SDDMM on Coo format.
 */

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

template <int XPU, typename IdType, int bits>
void gatherMM(const NDArray E_etype,
              const NDArray h,
              const NDArray w,
              NDArray out) {
  SWITCH_BITS(bits, DType, {

    int64_t num_rel = E_etype.NumElements();
    int n = w->shape[1]; // cols of B
    int k = h->shape[1]; // cols of A
    const DType *h_data = h.Ptr<DType>();
    const DType *w_data = w.Ptr<DType>();
    DType *out_data = out.Ptr<DType>();
    IdType* E_etype_data = static_cast<IdType*>(E_etype->data);
    int64_t h_offset = 0;
    int64_t w_offset = 0;
    int64_t out_offset = 0;
    DType alpha = 1., beta = 0.;

    auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
    if (!thr_entry->cublas_handle)
        CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
    CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle,
        thr_entry->stream));

    for (int etype = 0; etype < num_rel; ++etype) {
        int m = E_etype_data[etype]; // rows of A
        CUBLAS_CALL(cublasGemm<DType>(
          thr_entry->cublas_handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          m, n, k,
          &alpha,
          h_data + h_offset, m, //m x k
          w_data + w_offset, k, // k x n
          &beta,
          out_data + out_offset, m));
        h_offset += m * k;
        w_offset += k * n;
        out_offset += m * n;
    }
  });
}

template void gatherMM<kDLGPU, int32_t, 16>(
    const NDArray E_etype, const NDArray h,
    const NDArray w, NDArray out);
template void gatherMM<kDLGPU, int64_t, 16>(
    const NDArray E_etype, const NDArray h,
    const NDArray w, NDArray out);
template void gatherMM<kDLGPU, int32_t, 32>(
    const NDArray E_etype, const NDArray h,
    const NDArray w, NDArray out);
template void gatherMM<kDLGPU, int64_t, 32>(
    const NDArray E_etype, const NDArray h,
    const NDArray w, NDArray out);
template void gatherMM<kDLGPU, int32_t, 64>(
    const NDArray E_etype, const NDArray h,
    const NDArray w, NDArray out);
template void gatherMM<kDLGPU, int64_t, 64>(
    const NDArray E_etype, const NDArray h,
    const NDArray w, NDArray out);

}  // namespace aten
}  // namespace dgl
