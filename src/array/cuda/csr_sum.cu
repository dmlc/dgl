/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/spmm.cu
 * \brief SPMM C APIs and definitions.
 */
#include <dgl/array.h>
#include "./csr_sum.cuh"
// #include "./ge_spmm.cuh"
#include "./functor.cuh"
#include "../../runtime/cuda/cuda_common.h"

namespace dgl {

using namespace cuda;

namespace aten {
namespace cusparse {

//Seperate cuSPARSE calls for float and double precision datatype

template <typename DType>
cusparseStatus_t cusparseXcsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const DType* alpha,
    const cusparseMatDescr_t matA, int nnzA, const DType * csrAVal, 
    const int* csrRowPtrA, const int* csrColIndA, const DType* beta, 
    const cusparseMatDescr_t matB, int nnzB, const DType * csrBVal, 
    const int* csrRowPtrB, const int* csrColIndB, 
    const cusparseMatDescr_t matC, const DType * csrCVal, 
    const int* csrRowPtrC, const int* csrColIndC,
    size_t* workspace_size){
  LOG(INFO) << "Not supported dtype";
  return CUSPARSE_STATUS_EXECUTION_FAILED;
}

template<>
cusparseStatus_t cusparseXcsrgeam2_bufferSizeExt<float>(
    cusparseHandle_t handle, int m, int n, const float* alpha,
    const cusparseMatDescr_t matA, int nnzA, const float * csrAVal, 
    const int* csrRowPtrA, const int* csrColIndA, const float* beta, 
    const cusparseMatDescr_t matB, int nnzB, const float * csrBVal, 
    const int* csrRowPtrB, const int* csrColIndB, 
    const cusparseMatDescr_t matC, const float * csrCVal, 
    const int* csrRowPtrC, const int* csrColIndC,
    size_t* workspace_size){
  return cusparseScsrgeam2_bufferSizeExt(handle, m, n, alpha,
    matA, nnzA, csrAVal, csrRowPtrA, csrColIndA, 
    beta, matB, nnzB, csrBVal, csrRowPtrB, 
    csrColIndB, matC, csrCVal, csrRowPtrC, 
    csrColIndC, workspace_size);
}

template<>
cusparseStatus_t cusparseXcsrgeam2_bufferSizeExt<double>(
    cusparseHandle_t handle, int m, int n, const double* alpha,
    const cusparseMatDescr_t matA, int nnzA, const double * csrAVal, 
    const int* csrRowPtrA, const int* csrColIndA, const double* beta, 
    const cusparseMatDescr_t matB, int nnzB, const double * csrBVal, 
    const int* csrRowPtrB, const int* csrColIndB, 
    const cusparseMatDescr_t matC, const double * csrCVal, 
    const int* csrRowPtrC, const int* csrColIndC,
    size_t* workspace_size){
  return cusparseDcsrgeam2_bufferSizeExt(handle, m, n, alpha,
    matA, nnzA, csrAVal, csrRowPtrA, csrColIndA, 
    beta, matB, nnzB, csrBVal, csrRowPtrB, 
    csrColIndB, matC, csrCVal, csrRowPtrC, 
    csrColIndC, workspace_size);
}

template <typename DType>
cusparseStatus_t cusparseXcsrgeam2(
    cusparseHandle_t handle, int m, int n, const DType* alpha,
    const cusparseMatDescr_t matA, int nnzA, const DType * csrAVal, 
    const int* csrRowPtrA, const int* csrColIndA, const DType* beta, 
    const cusparseMatDescr_t matB, int nnzB, const DType * csrBVal, 
    const int* csrRowPtrB, const int* csrColIndB, 
    const cusparseMatDescr_t matC, DType * csrCVal, 
    int* csrRowPtrC, int* csrColIndC,
    void* workspace){
  LOG(INFO) << "Not supported dtype";
  return CUSPARSE_STATUS_EXECUTION_FAILED;
}

template<>
cusparseStatus_t cusparseXcsrgeam2<float>(
    cusparseHandle_t handle, int m, int n, const float* alpha,
    const cusparseMatDescr_t matA, int nnzA, const float * csrAVal, 
    const int* csrRowPtrA, const int* csrColIndA, const float* beta, 
    const cusparseMatDescr_t matB, int nnzB, const float * csrBVal, 
    const int* csrRowPtrB, const int* csrColIndB, 
    const cusparseMatDescr_t matC, float * csrCVal, 
    int* csrRowPtrC, int* csrColIndC,
    void* workspace){
  return cusparseScsrgeam2(handle, m, n, alpha,
    matA, nnzA, csrAVal, csrRowPtrA, csrColIndA, 
    beta, matB, nnzB, csrBVal, csrRowPtrB, 
    csrColIndB, matC, csrCVal, csrRowPtrC, 
    csrColIndC, workspace);
}

template<>
cusparseStatus_t cusparseXcsrgeam2<double>(
    cusparseHandle_t handle, int m, int n, const double* alpha,
    const cusparseMatDescr_t matA, int nnzA, const double * csrAVal, 
    const int* csrRowPtrA, const int* csrColIndA, const double* beta, 
    const cusparseMatDescr_t matB, int nnzB, const double * csrBVal, 
    const int* csrRowPtrB, const int* csrColIndB, 
    const cusparseMatDescr_t matC, double * csrCVal, 
    int* csrRowPtrC, int* csrColIndC,
    void* workspace){
  return cusparseDcsrgeam2(handle, m, n, alpha,
    matA, nnzA, csrAVal, csrRowPtrA, csrColIndA, 
    beta, matB, nnzB, csrBVal, csrRowPtrB, 
    csrColIndB, matC, csrCVal, csrRowPtrC, 
    csrColIndC, workspace);
}

/*! Cusparse implementation of SpGEMM on Csr format. */
template <typename DType, typename IdType>
void CusparseCsrgeam2(
    const DLContext& ctx,
    const CSRMatrix& A,
    const DType* A_weights, 
    const CSRMatrix& B,
    const DType* B_weights,
    int& nnzC) {
  const int m = A.num_rows;
  const int n = A.num_cols;
  const int nnzA = A.indices->shape[0];
  const int nnzB = B.indices->shape[0];
  int baseC;
  int *nnzTotalDevHostPtr = &nnzC;
  const DType alpha = 1.0;
  const DType beta = 0.0;
  // device
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, thr_entry->stream));
  cusparseMatDescr_t matA, matB, matC;
  // all one data array
  DType* valptr = nullptr;
  // if (!A_data) {
  //   valptr = static_cast<DType*>(device->AllocWorkspace(ctx, nnz * sizeof(DType)));
  //   _Fill(valptr, nnz, static_cast<DType>(1.));
  // }
#if CUDART_VERSION >= 11000
  constexpr auto dtype = cuda_dtype<DType>::value;
  constexpr auto idtype = cusparse_idtype<IdType>::value;
  // Create sparse matrix A, B and C in CSR format 
  cusparseSetPointerMode(thr_entry->cusparse_handle, 
    CUSPARSE_POINTER_MODE_HOST);
  size_t workspace_size;
  IdType* dC_csrOffsets, *dC_columns; 
  DType *dC_values;
  dC_csrOffsets = (IdType*)(device->AllocWorkspace(ctx, 
    (A.num_rows+1) * sizeof(IdType)));
  /* prepare buffer */
  CUSPARSE_CALL(cusparseXcsrgeam2_bufferSizeExt<DType>(
     thr_entry->cusparse_handle, m, n, &alpha,
     matA, nnzA, static_cast<DType*>(A.data->data), 
     static_cast<IdType*>(A.indptr->data), 
     static_cast<IdType*>(A.indices->data), 
     &beta, matB, nnzB, static_cast<DType*>(B.data->data), 
     static_cast<IdType*>(B.indptr->data), 
     static_cast<IdType*>(B.indices->data),
     matC, dC_values, dC_csrOffsets, dC_columns,
     &workspace_size));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);
  CUSPARSE_CALL(cusparseXcsrgeam2Nnz(thr_entry->cusparse_handle, m, n,     
     matA, nnzA, static_cast<IdType*>(A.indptr->data), 
     static_cast<IdType*>(A.indices->data), matB, nnzB,
     static_cast<IdType*>(B.indptr->data), 
     static_cast<IdType*>(B.indices->data), matC, 
     static_cast<IdType*>(dC_csrOffsets), 
     nnzTotalDevHostPtr, workspace)); 
  if(NULL != nnzTotalDevHostPtr){
      nnzC = *nnzTotalDevHostPtr;
  }else{
      cudaMemcpy(&nnzC, dC_csrOffsets + m, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&baseC, dC_csrOffsets, sizeof(int), cudaMemcpyDeviceToHost);
      nnzC -= baseC;
  }
  dC_columns = (IdType*)(device->AllocWorkspace(ctx, nnzC * sizeof(IdType)));
  dC_values = (DType*)device->AllocWorkspace(ctx, nnzC * sizeof(dtype));
  // //TODO:: cudamalloc for outcsr inds, vals
  CUSPARSE_CALL(cusparseXcsrgeam2<DType>(
     thr_entry->cusparse_handle, m, n, &alpha,
     matA, nnzA, static_cast<DType*>(A.data->data), 
     static_cast<IdType*>(A.indptr->data), 
     static_cast<IdType*>(A.indices->data), 
     &beta, matB, nnzB, static_cast<DType*>(B.data->data), 
     static_cast<IdType*>(B.indptr->data), 
     static_cast<IdType*>(B.indices->data),
     matC, dC_values, dC_csrOffsets, dC_columns,
     workspace));
 
  device->FreeWorkspace(ctx, workspace);
  // destroy matrix/vector descriptors
  CUSPARSE_CALL( cusparseDestroyMatDescr(matA));
  CUSPARSE_CALL( cusparseDestroyMatDescr(matB));
  CUSPARSE_CALL( cusparseDestroyMatDescr(matC));
  CUSPARSE_CALL( cusparseDestroy(thr_entry->cusparse_handle));
#else
  LOG(FATAL) << "Not tested on CUDA < 11.2.1";
#endif
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
std::pair<CSRMatrix, NDArray> CSRSum(
    const std::vector<CSRMatrix>& list_A,
    const std::vector<NDArray>& list_A_weights) {
  CHECK(list_A.size() > 0) << "List of matrices can't be empty.";
  CHECK_EQ(list_A.size(), list_A_weights.size()) << "List of matrices and weights must have same length";
  const int64_t M = list_A[0].num_rows;
  const int64_t N = list_A[0].num_cols;
  const int64_t n = list_A.size();
  int nnz = 0;
  //Make it a for loop
  CHECK(list_A.size() == 2) << "CuSPASR csrgeam supports 2 matrices at a time.";
  const CSRMatrix& A = list_A[0];
  const NDArray& A_weights = list_A_weights[0];
  const CSRMatrix& B = list_A[1];
  const NDArray& B_weights = list_A_weights[1];

  IdArray C_indptr = IdArray::Empty({M + 1}, list_A[0].indptr->dtype, list_A[0].indptr->ctx);
  IdType* C_indptr_data = C_indptr.Ptr<IdType>();
  
  if (cusparse_available<32, IdType>()) {  // cusparse
    cusparse::CusparseCsrgeam2<DType, IdType>(
      A_weights->ctx, 
      A, static_cast<DType*>(A_weights->data),
      B, static_cast<DType*>(B_weights->data),
      nnz);
  } else {  
  }

  IdArray C_indices = IdArray::Empty({nnz}, list_A[0].indices->dtype, list_A[0].indices->ctx);
  NDArray C_weights = NDArray::Empty({nnz}, list_A_weights[0]->dtype, list_A_weights[0]->ctx);

  return {CSRMatrix(M, N, C_indptr, C_indices), C_weights};
}

template std::pair<CSRMatrix, NDArray> CSRSum<kDLGPU, int32_t, float>(
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&);
// template std::pair<CSRMatrix, NDArray> CSRSum<kDLGPU, int64_t, float>(
    // const std::vector<CSRMatrix>&, const std::vector<NDArray>&);
template std::pair<CSRMatrix, NDArray> CSRSum<kDLGPU, int32_t, double>(
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&);
// template std::pair<CSRMatrix, NDArray> CSRSum<kDLGPU, int64_t, double>(
    // const std::vector<CSRMatrix>&, const std::vector<NDArray>&);

}  // namespace aten
}  // namespace dgl
