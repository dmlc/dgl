/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/csr_mm.cu
 * \brief SpSpMM/SpGEMM C APIs and definitions.
 */
#include <dgl/array.h>
#include "./csr_mm.cuh"
#include <inttypes.h>
// #include "./ge_spmm.cuh"
#include "./functor.cuh"
#include "../../runtime/cuda/cuda_common.h"

namespace dgl {

using namespace cuda;

namespace aten {
namespace cusparse {

/*! Cusparse implementation of SpGEMM on Csr format. */
template <typename DType, typename IdType>
std::pair<CSRMatrix, NDArray> CusparseSpgemm(
    const DLContext& ctx,
    const CSRMatrix& A,
    const DType* A_weights, 
    const CSRMatrix& B,
    const DType* B_weights,
    const DLDataType C_idtype,
    const DLDataType C_dtype) {
  // We use Spgemm (SpSpMM) to perform following operation:
  // C = A x B, where A, B and C are sparse matrices in csr format.
  const int nnzA = A.indices->shape[0];
  const int nnzB = B.indices->shape[0];
  const DType alpha = 1.0;
  const DType beta = 0.0;
  auto transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  // device
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, thr_entry->stream));
  // all one data array
  DType* valptrA = nullptr;
  if (!A_weights) {
    valptrA = static_cast<DType*>(device->AllocWorkspace(ctx, nnzA * sizeof(DType)));
    _Fill(valptrA, nnzA, static_cast<DType>(1.));
  }
  DType* valptrB = nullptr;
  if (!B_weights) {
    valptrB = static_cast<DType*>(device->AllocWorkspace(ctx, nnzB * sizeof(DType)));
    _Fill(valptrB, nnzB, static_cast<DType>(1.));
  }
#if CUDART_VERSION >= 11000
  cusparseSpMatDescr_t matA, matB, matC;
  IdArray dC_csrOffsets = IdArray::Empty({A.num_rows+1}, C_idtype, A.indptr->ctx);
  IdType* dC_csrOffsets_data = dC_csrOffsets.Ptr<IdType>();
  constexpr auto dtype = cuda_dtype<DType>::value;
  constexpr auto idtype = cusparse_idtype<IdType>::value;
  // Create sparse matrix A, B and C in CSR format
  CUSPARSE_CALL(cusparseCreateCsr(&matA,
      A.num_rows, A.num_cols, nnzA,
      static_cast<IdType*>(A.indptr->data),
      static_cast<IdType*>(A.indices->data),
      const_cast<DType*>(valptrA? valptrA : A_weights), 
      idtype, idtype, CUSPARSE_INDEX_BASE_ZERO, dtype));
  CUSPARSE_CALL(cusparseCreateCsr(&matB,
      B.num_rows, B.num_cols, nnzB,
      static_cast<IdType*>(B.indptr->data),
      static_cast<IdType*>(B.indices->data),
      const_cast<DType*>(valptrB? valptrB : B_weights), 
      idtype, idtype, CUSPARSE_INDEX_BASE_ZERO, dtype));
  CUSPARSE_CALL(cusparseCreateCsr(&matC,
      A.num_rows, B.num_cols, 0,
      NULL, NULL, NULL, idtype, idtype,
      CUSPARSE_INDEX_BASE_ZERO, dtype));
  // SpGEMM Computation
  cusparseSpGEMMDescr_t spgemmDesc;
  CUSPARSE_CALL( cusparseSpGEMM_createDescr(&spgemmDesc) )
  size_t workspace_size1 = 0, workspace_size2 = 0;
  // ask bufferSize1 bytes for external memory
  CUSPARSE_CALL(cusparseSpGEMM_workEstimation(
      thr_entry->cusparse_handle, transA, transB,
      &alpha, matA, matB, &beta, matC, dtype, 
      CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &workspace_size1, 
      NULL));
  void* workspace1 = (device->AllocWorkspace(ctx, workspace_size1));
  // inspect the matrices A and B to understand the memory requiremnent
  // for the next step
  CUSPARSE_CALL(cusparseSpGEMM_workEstimation(
      thr_entry->cusparse_handle, transA, transB,
      &alpha, matA, matB, &beta, matC, dtype, 
      CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &workspace_size1, 
      workspace1));
  // ask bufferSize2 bytes for external memory
  CUSPARSE_CALL(cusparseSpGEMM_compute(thr_entry->cusparse_handle,
      transA, transB, &alpha, matA, matB, &beta, matC,
      dtype, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &workspace_size2,
      NULL));
  void* workspace2 = device->AllocWorkspace(ctx, workspace_size2);
  // compute the intermediate product of A * B
  CUSPARSE_CALL(cusparseSpGEMM_compute(thr_entry->cusparse_handle,
      transA, transB, &alpha, matA, matB, &beta, matC,
      dtype, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &workspace_size2,
      workspace2));
  // get matrix C non-zero entries C_nnz1
  int64_t C_num_rows1, C_num_cols1, C_nnz1;
  CUSPARSE_CALL( cusparseSpMatGetSize(matC, &C_num_rows1, 
    &C_num_cols1, &C_nnz1));
  IdArray dC_columns = IdArray::Empty({C_nnz1}, C_idtype, A.indptr->ctx);
  NDArray dC_weights = NDArray::Empty({C_nnz1}, C_dtype, A.indptr->ctx);
  IdType* dC_columns_data = dC_columns.Ptr<IdType>();
  DType* dC_weights_data = dC_weights.Ptr<DType>();
  // update matC with the new pointers
  CUSPARSE_CALL(cusparseCsrSetPointers(matC, dC_csrOffsets_data, 
     dC_columns_data, dC_weights_data));
  // copy the final products to the matrix C
  CUSPARSE_CALL(cusparseSpGEMM_copy(thr_entry->cusparse_handle,
      transA, transB, &alpha, matA, matB, &beta, matC,
      dtype, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

  device->FreeWorkspace(ctx, workspace1);
  device->FreeWorkspace(ctx, workspace2);
  // destroy matrix/vector descriptors
  CUSPARSE_CALL( cusparseSpGEMM_destroyDescr(spgemmDesc));
  CUSPARSE_CALL( cusparseDestroySpMat(matA));
  CUSPARSE_CALL( cusparseDestroySpMat(matB));
  CUSPARSE_CALL( cusparseDestroySpMat(matC));
#else
  LOG(FATAL) << "Not tested on CUDA < 11.0";
#endif
  if (valptrA)
    device->FreeWorkspace(ctx, valptrA);
  if (valptrB)
    device->FreeWorkspace(ctx, valptrB);
  return {CSRMatrix(A.num_rows, B.num_cols, dC_csrOffsets, dC_columns), dC_weights};
}
}  // namespace cusparse

/*!
 * \brief Determine whether cusparse SpGEMM function is applicable.
 */
template <typename IdType>
inline bool cusparse_available() {
  if (std::is_same<IdType, int64_t>::value)
    return false; // cusparse's SpGEMM does not allow 64 bits index type.
  return true;
}

/*!
 * \brief CUDA implementation of SpSpMM/SpGEMM on Csr format.
 */
template <int XPU, typename IdType, typename DType>
std::pair<CSRMatrix, NDArray> CSRMM(
    const CSRMatrix& A,
    NDArray A_weights,
    const CSRMatrix& B,
    NDArray B_weights) {
  if (cusparse_available<IdType>()) { 
    return cusparse::CusparseSpgemm<DType, IdType>(
      A.indptr->ctx, 
      A, static_cast<DType*>(A_weights->data),
      B, static_cast<DType*>(B_weights->data),
      A.indices->dtype, A_weights->dtype);
  } else {  
    LOG(FATAL) << "cuSPARSE SpGEMM does not support int64_t.";
  }
}

template std::pair<CSRMatrix, NDArray> CSRMM<kDLGPU, int32_t, float>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
template std::pair<CSRMatrix, NDArray> CSRMM<kDLGPU, int64_t, float>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
template std::pair<CSRMatrix, NDArray> CSRMM<kDLGPU, int32_t, double>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
template std::pair<CSRMatrix, NDArray> CSRMM<kDLGPU, int64_t, double>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);

}  // namespace aten
}  // namespace dgl
