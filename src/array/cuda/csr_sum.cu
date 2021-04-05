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

/*! Cusparse implementation of SpSum on Csr format. */
template <typename DType, typename IdType>
std::pair<CSRMatrix, NDArray> CusparseCsrgeam2(
    const DLContext& ctx,
    const CSRMatrix& A,
    const DType* A_weights, 
    const CSRMatrix& B,
    const DType* B_weights,
    const DLDataType C_idtype,
    const DLDataType C_dtype) {
  const int m = A.num_rows;
  const int n = A.num_cols;
  const int nnzA = A.indices->shape[0];
  const int nnzB = B.indices->shape[0];
  int baseC, nnzC;
  int *nnzTotalDevHostPtr = &nnzC;
  const DType alpha = 1.0;
  const DType beta = 1.0;
  //device
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, thr_entry->stream));
  cusparseMatDescr_t matA, matB, matC;
  CUSPARSE_CALL(cusparseCreateMatDescr(&matA));
  CUSPARSE_CALL(cusparseCreateMatDescr(&matB));
  CUSPARSE_CALL(cusparseCreateMatDescr(&matC));
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
  constexpr auto dtype = cuda_dtype<DType>::value;
  constexpr auto idtype = cusparse_idtype<IdType>::value;
  cusparseSetPointerMode(thr_entry->cusparse_handle, 
    CUSPARSE_POINTER_MODE_HOST);
  size_t workspace_size = 0;
  /* prepare output C */
  IdArray dC_csrOffsets = IdArray::Empty({A.num_rows+1}, C_idtype, A.indptr->ctx);
  IdType* dC_csrOffsets_data = dC_csrOffsets.Ptr<IdType>();
  IdArray dC_columns; 
  NDArray dC_weights; 
  IdType* dC_columns_data = dC_columns.Ptr<IdType>();
  DType* dC_weights_data = dC_weights.Ptr<DType>();
  /* prepare buffer */
  CUSPARSE_CALL(cusparseXcsrgeam2_bufferSizeExt<DType>(
     thr_entry->cusparse_handle, m, n, &alpha,
     matA, nnzA, const_cast<DType*>(valptrA? valptrA : A_weights), 
     static_cast<IdType*>(A.indptr->data), 
     static_cast<IdType*>(A.indices->data), 
     &beta, matB, nnzB, const_cast<DType*>(valptrB? valptrB : B_weights), 
     static_cast<IdType*>(B.indptr->data), 
     static_cast<IdType*>(B.indices->data),
     matC, dC_weights_data, dC_csrOffsets_data, dC_columns_data,
     &workspace_size));
  
  void *workspace = (device->AllocWorkspace(ctx, workspace_size)); //sizeof(char)
  CUSPARSE_CALL(cusparseXcsrgeam2Nnz(thr_entry->cusparse_handle, 
     m, n, matA, nnzA, 
     static_cast<IdType*>(A.indptr->data), 
     static_cast<IdType*>(A.indices->data), 
     matB, nnzB,
     static_cast<IdType*>(B.indptr->data), 
     static_cast<IdType*>(B.indices->data), 
     matC, dC_csrOffsets_data, nnzTotalDevHostPtr, workspace)); 
  
  if(NULL != nnzTotalDevHostPtr){
      nnzC = *nnzTotalDevHostPtr;
  }else{
      cudaMemcpy(&nnzC, dC_csrOffsets_data + m, sizeof(IdType), cudaMemcpyDeviceToHost);
      cudaMemcpy(&baseC, dC_csrOffsets_data, sizeof(IdType), cudaMemcpyDeviceToHost);
      nnzC -= baseC;
  }
  dC_columns = IdArray::Empty({nnzC}, C_idtype, A.indptr->ctx);
  dC_weights = NDArray::Empty({nnzC}, C_dtype, A.indptr->ctx);
  dC_columns_data = dC_columns.Ptr<IdType>();
  dC_weights_data = dC_weights.Ptr<DType>();

  CUSPARSE_CALL(cusparseXcsrgeam2<DType>(
     thr_entry->cusparse_handle, m, n, &alpha,
     matA, nnzA, const_cast<DType*>(valptrA? valptrA : A_weights),  
     static_cast<IdType*>(A.indptr->data), 
     static_cast<IdType*>(A.indices->data), 
     &beta, matB, nnzB, const_cast<DType*>(valptrB? valptrB : B_weights),  
     static_cast<IdType*>(B.indptr->data), 
     static_cast<IdType*>(B.indices->data),
     matC, dC_weights_data, dC_csrOffsets_data, dC_columns_data,
     workspace));

  device->FreeWorkspace(ctx, workspace);
  // destroy matrix/vector descriptors
  CUSPARSE_CALL( cusparseDestroyMatDescr(matA));
  CUSPARSE_CALL( cusparseDestroyMatDescr(matB));
  CUSPARSE_CALL( cusparseDestroyMatDescr(matC));
#else
  LOG(FATAL) << "Not tested on CUDA < 11.0.2";
#endif
  if (valptrA)
    device->FreeWorkspace(ctx, valptrA);
  if (valptrB)
    device->FreeWorkspace(ctx, valptrB);
  return {CSRMatrix(A.num_rows, A.num_cols, dC_csrOffsets, dC_columns), 
    dC_weights};
}
}  // namespace cusparse

/*!
 * \brief Determine whether cusparse CSRGEAM function is applicable.
 */
template <typename IdType>
inline bool cusparse_available() {
  if (std::is_same<IdType, int64_t>::value)
    return false; // cusparse's SpGEMM does not allow 64 bits index type.
  return true;
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
  CHECK(list_A.size() == 2) << "CuSPARSE csrgeam supports 2 matrices at a time.";
  const CSRMatrix& A = list_A[0];
  const NDArray& A_weights = list_A_weights[0];
  const CSRMatrix& B = list_A[1];
  const NDArray& B_weights = list_A_weights[1];
  
  if (cusparse_available<IdType>()) { // cusparse
    return cusparse::CusparseCsrgeam2<DType, int32_t>(
      A_weights->ctx, 
      A, static_cast<DType*>(A_weights->data),
      B, static_cast<DType*>(B_weights->data),
      A.indices->dtype, A_weights->dtype);
  } else {  
      LOG(FATAL) << "DOBULECHECK!! cuSPARSE SpGEMM does not support int64_t.";
  }
}

template std::pair<CSRMatrix, NDArray> CSRSum<kDLGPU, int32_t, float>(
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&);
template std::pair<CSRMatrix, NDArray> CSRSum<kDLGPU, int64_t, float>(
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&);
template std::pair<CSRMatrix, NDArray> CSRSum<kDLGPU, int32_t, double>(
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&);
template std::pair<CSRMatrix, NDArray> CSRSum<kDLGPU, int64_t, double>(
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&);

}  // namespace aten
}  // namespace dgl
