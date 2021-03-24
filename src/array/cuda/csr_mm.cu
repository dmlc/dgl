/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/spmm.cu
 * \brief SPMM C APIs and definitions.
 */
#include <dgl/array.h>
#include "./csr_mm.cuh"
// #include "./ge_spmm.cuh"
#include "./functor.cuh"
#include "../../runtime/cuda/cuda_common.h"

namespace dgl {

using namespace cuda;

namespace aten {
namespace cusparse {

/*! Cusparse implementation of SpGEMM on Csr format. */
template <typename DType, typename IdType>
void CusparseSpgemm(
    const DLContext& ctx,
    const CSRMatrix& A,
    const DType* A_weights, 
    const CSRMatrix& B,
    const DType* B_weights,
    int& final_nnzC) {
  const int m = A.num_rows;
  const int n = A.num_cols;
  const int p = B.num_cols;
  const int nnzA = A.indices->shape[0];
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
  DType* valptr = nullptr;
  // if (!A_data) {
  //   valptr = static_cast<DType*>(device->AllocWorkspace(ctx, nnz * sizeof(DType)));
  //   _Fill(valptr, nnz, static_cast<DType>(1.));
  // }
#if CUDART_VERSION >= 11000
  cusparseSpMatDescr_t matA, matB, matC;
  void* dC_csrOffsets = device->AllocWorkspace(ctx, 
    (A.num_rows+1) * sizeof(IdType));
  constexpr auto dtype = cuda_dtype<DType>::value;
  constexpr auto idtype = cusparse_idtype<IdType>::value;
  // Create sparse matrix A, B and C in CSR format
  CUSPARSE_CALL(cusparseCreateCsr(&matA,
      A.num_rows, A.num_cols, A.indices->shape[0],
      static_cast<IdType*>(A.indptr->data),
      static_cast<IdType*>(A.indices->data),
      static_cast<DType*>(A.data->data), idtype, 
      idtype, CUSPARSE_INDEX_BASE_ZERO, dtype));
  CUSPARSE_CALL(cusparseCreateCsr(&matB,
      B.num_rows, B.num_cols, B.indices->shape[0],
      static_cast<IdType*>(B.indptr->data),
      static_cast<IdType*>(B.indices->data),
      static_cast<DType*>(B.data->data), idtype, 
      idtype, CUSPARSE_INDEX_BASE_ZERO, dtype));
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
  void* workspace1 = device->AllocWorkspace(ctx, workspace_size1);
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
  void* dC_columns = device->AllocWorkspace(ctx, C_nnz1 * sizeof(IdType));
  void* dC_values = device->AllocWorkspace(ctx, C_nnz1 * sizeof(dtype));
  // update matC with the new pointers
  CUSPARSE_CALL(cusparseCsrSetPointers(matC, dC_csrOffsets, 
     dC_columns, dC_values));
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
  CUSPARSE_CALL( cusparseDestroy(thr_entry->cusparse_handle));
#else
  LOG(FATAL) << "Not tested on CUDA < 11.2.1";
#endif
  final_nnzC = C_nnz1;
  if (valptr)
    device->FreeWorkspace(ctx, valptr);
}
}  // namespace cusparse

/*!
 * \brief Determine whether cusparse SpGEMM function is applicable.
 */
template <int bits, typename IdType>
inline bool cusparse_available() {
#if CUDART_VERSION < 11000
  if (std::is_same<IdType, int>::value)
    if (bits > 16)
      return true;
  return false;
#else
  if (bits == 16)
    return false;  // cusparse's SpMM on fp16 is slow, temporally disabled.
  return true;
#endif
}

/*!
 * \brief CUDA implementation of g-SpMM on Csr format.
 * \note use cusparse if the reduce operator is `sum` and there is
 *       no broadcast, use dgl's kernel in other cases.
 */
template <int XPU, typename IdType, typename DType>
std::pair<CSRMatrix, NDArray> CSRMM(
    const CSRMatrix& A,
    NDArray A_weights,
    const CSRMatrix& B,
    NDArray B_weights) {
  const int M = A.num_rows;
  const int P = B.num_cols;
  int nnz=0;
  IdArray C_indptr = IdArray::Empty({M + 1}, A.indptr->dtype, A.indptr->ctx);
  IdType* C_indptr_data = C_indptr.Ptr<IdType>();
  
  if (cusparse_available<32, IdType>()) {  // cusparse
    cusparse::CusparseSpgemm<DType, IdType>(
      A_weights->ctx, 
      A, static_cast<DType*>(A_weights->data),
      B, static_cast<DType*>(B_weights->data),
      nnz);
  } else {  
  }
  NDArray C_weights = NDArray::Empty({nnz}, A_weights->dtype, A_weights->ctx);
  IdArray C_indices = IdArray::Empty({nnz}, A.indices->dtype, A.indices->ctx);
 
  return {CSRMatrix(M, P, C_indptr, C_indices), C_weights};
}

template std::pair<CSRMatrix, NDArray> CSRMM<kDLGPU, int32_t, float>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
// template std::pair<CSRMatrix, NDArray> CSRMM<kDLGPU, int64_t, float>(
    // const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
template std::pair<CSRMatrix, NDArray> CSRMM<kDLGPU, int32_t, double>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
// template std::pair<CSRMatrix, NDArray> CSRMM<kDLGPU, int64_t, double>(
    // const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);

}  // namespace aten
}  // namespace dgl
