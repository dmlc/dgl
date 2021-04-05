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
void CusparseCsrgeam2(
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
  const DType beta = 0.0;
  //device
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, thr_entry->stream));
  cusparseMatDescr_t matA, matB, matC;
  // all one data array
  DType* valptrA = nullptr;
  if (!A_weights) {
    // valptrA = static_cast<DType*>(device->AllocWorkspace(ctx, nnzA * sizeof(DType)));
    CUDA_CALL( cudaMalloc((void**) &valptrA, (nnzA) * sizeof(DType)) );
    _Fill(valptrA, nnzA, static_cast<DType>(1.));
  }
  DType* valptrB = nullptr;
  if (!B_weights) {
     CUDA_CALL( cudaMalloc((void**) &valptrB, (nnzB) * sizeof(DType)) );
    // valptrB = static_cast<DType*>(device->AllocWorkspace(ctx, nnzB * sizeof(DType)));
    _Fill(valptrB, nnzB, static_cast<DType>(1.));
  }
#if CUDART_VERSION >= 11000
  constexpr auto dtype = cuda_dtype<DType>::value;
  constexpr auto idtype = cusparse_idtype<IdType>::value;
  cusparseSetPointerMode(thr_entry->cusparse_handle, 
    CUSPARSE_POINTER_MODE_HOST);
  size_t workspace_size = 0;
  IdType *dC_columns, *dC_csrOffsets; 
  DType *dC_values;
  // IdArray dC_csrOffsets = IdArray::Empty({A.num_rows+1}, C_idtype, A.indptr->ctx);
  // IdType* dC_csrOffsets_data = dC_csrOffsets.Ptr<IdType>();
  // /* prepare buffer */
  
  CUDA_CALL( cudaMalloc((void**) &dC_csrOffsets,
    (m + 1) * sizeof(IdType)) );
  CUSPARSE_CALL(cusparseXcsrgeam2_bufferSizeExt<DType>(
     thr_entry->cusparse_handle, m, n, &alpha,
     matA, nnzA, const_cast<DType*>(valptrA? valptrA : A_weights), 
     static_cast<IdType*>(A.indptr->data), 
     static_cast<IdType*>(A.indices->data), 
     &beta, matB, nnzB, const_cast<DType*>(valptrB? valptrB : B_weights), 
     static_cast<IdType*>(B.indptr->data), 
     static_cast<IdType*>(B.indices->data),
     matC, dC_values, dC_csrOffsets, dC_columns,
     &workspace_size));
  
  char *workspace = NULL;
  CUDA_CALL(cudaMalloc((void**) &workspace, sizeof(char) * workspace_size) );
    // = device->AllocWorkspace(ctx, workspace_size);
  CUSPARSE_CALL(cusparseXcsrgeam2Nnz(thr_entry->cusparse_handle, 
     m, n, matA, nnzA, 
     static_cast<IdType*>(A.indptr->data), 
     static_cast<IdType*>(A.indices->data), 
     matB, nnzB,
     static_cast<IdType*>(B.indptr->data), 
     static_cast<IdType*>(B.indices->data), 
     matC, dC_csrOffsets, nnzTotalDevHostPtr, workspace)); 
  
  if(NULL != nnzTotalDevHostPtr){
      nnzC = *nnzTotalDevHostPtr;
  }else{
      cudaMemcpy(&nnzC, dC_csrOffsets + m, sizeof(IdType), cudaMemcpyDeviceToHost);
      cudaMemcpy(&baseC, dC_csrOffsets, sizeof(IdType), cudaMemcpyDeviceToHost);
      nnzC -= baseC;
  }
  // IdArray dC_columns = IdArray::Empty({nnzC}, C_idtype, A.indptr->ctx);
  // NDArray dC_weights = NDArray::Empty({nnzC}, C_dtype, A.indptr->ctx);
  // IdType* dC_columns_data = dC_columns.Ptr<IdType>();
  // DType* dC_weights_data = dC_weights.Ptr<DType>();

  // dC_columns = (IdType*)(device->AllocWorkspace(ctx, nnzC * sizeof(IdType)));
  // dC_values = (DType*)device->AllocWorkspace(ctx, nnzC * sizeof(dtype));
  // //TODO:: cudamalloc for outcsr inds, vals
  // CUSPARSE_CALL(cusparseXcsrgeam2<DType>(
  //    thr_entry->cusparse_handle, m, n, &alpha,
  //    matA, nnzA, const_cast<DType*>(valptrA? valptrA : A_weights),  
  //    static_cast<IdType*>(A.indptr->data), 
  //    static_cast<IdType*>(A.indices->data), 
  //    &beta, matB, nnzB, const_cast<DType*>(valptrB? valptrB : B_weights),  
  //    static_cast<IdType*>(B.indptr->data), 
  //    static_cast<IdType*>(B.indices->data),
  //    matC, dC_values, dC_csrOffsets_data, dC_columns,
  //    workspace));
 
  // device->FreeWorkspace(ctx, workspace);
  // destroy matrix/vector descriptors
  CUSPARSE_CALL( cusparseDestroyMatDescr(matA));
  CUSPARSE_CALL( cusparseDestroyMatDescr(matB));
  CUSPARSE_CALL( cusparseDestroyMatDescr(matC));
  cudaFree(workspace);
  cudaFree(dC_csrOffsets);
  // cudaFree(dC_columns);
  // cudaFree(dC_values);
#else
  LOG(FATAL) << "Not tested on CUDA < 11.2.1";
#endif
    cudaFree(valptrA);
    cudaFree(valptrB);
  // if (valptrA)
  //   device->FreeWorkspace(ctx, valptrA);
  // if (valptrB)
  //   device->FreeWorkspace(ctx, valptrB);
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
  
  //created hard coded matrices for debugging
  const int A_num_rows = 4;
  const int A_num_cols = 4;
  const int A_nnz      = 9;
  const int B_num_rows = 4;
  const int B_num_cols = 4;
  const int B_nnz      = 9;
  
  IdType   hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
  IdType   hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
  DType hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                            6.0f, 7.0f, 8.0f, 9.0f };
  IdType   hB_csrOffsets[] = { 0, 2, 4, 7, 8 };
  IdType   hB_columns[]    = { 0, 3, 1, 3, 0, 1, 2, 1 };
  DType hB_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                            6.0f, 7.0f, 8.0f };
  IdType   hC_csrOffsets[] = { 0, 4, 6, 10, 12 };
  IdType   hC_columns[]    = { 0, 1, 2, 3, 1, 3, 0, 1, 2, 3, 1, 3 };
  DType hC_values[]     = { 11.0f, 36.0f, 14.0f, 2.0f,  12.0f,
                            16.0f, 35.0f, 92.0f, 42.0f, 10.0f,
                            96.0f, 32.0f };

  IdArray dA_csrOffsets = IdArray::Empty({M + 1}, list_A[0].indptr->dtype, list_A[0].indptr->ctx);
  IdType* dA_csrOffsets_data = dA_csrOffsets.Ptr<IdType>();
  IdArray dA_columns = IdArray::Empty({A_nnz}, list_A[0].indices->dtype, list_A[0].indices->ctx);
  IdType* dA_columns_data = dA_columns.Ptr<IdType>();
  NDArray dA_values = NDArray::Empty({A_nnz}, list_A_weights[0]->dtype, list_A_weights[0]->ctx);
  DType* dA_values_data = dA_values.Ptr<DType>(); 
  
  IdArray dB_csrOffsets = IdArray::Empty({M + 1}, list_A[1].indptr->dtype, list_A[0].indptr->ctx);
  IdType* dB_csrOffsets_data = dB_csrOffsets.Ptr<IdType>();
  IdArray dB_columns = IdArray::Empty({B_nnz}, list_A[0].indices->dtype, list_A[0].indices->ctx);
  IdType* dB_columns_data = dB_columns.Ptr<IdType>(); 
  NDArray dB_values = NDArray::Empty({B_nnz}, list_A_weights[0]->dtype, list_A_weights[0]->ctx);
  DType* dB_values_data = dB_values.Ptr<DType>(); 

  // copy A
  CUDA_CALL( cudaMemcpy(dA_csrOffsets_data, hA_csrOffsets,
                         (A_num_rows + 1) * sizeof(IdType),
                         cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMemcpy(dA_columns_data, hA_columns, A_nnz * sizeof(IdType),
                         cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMemcpy(dA_values_data, hA_values,
                         A_nnz * sizeof(DType), cudaMemcpyHostToDevice) );
  // copy B
  CUDA_CALL( cudaMemcpy(dB_csrOffsets_data, hB_csrOffsets,
                         (B_num_rows + 1) * sizeof(IdType),
                         cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMemcpy(dB_columns_data, hB_columns, B_nnz * sizeof(IdType),
                         cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMemcpy(dB_values_data, hB_values,
                         B_nnz * sizeof(DType), cudaMemcpyHostToDevice) );
  
  CSRMatrix tmpA = CSRMatrix(M, N, dA_csrOffsets, dA_columns);
  CSRMatrix tmpB = CSRMatrix(M, N, dB_csrOffsets, dB_columns);

  IdArray C_indptr = IdArray::Empty({M + 1}, list_A[0].indptr->dtype, list_A[0].indptr->ctx);
  IdType* C_indptr_data = C_indptr.Ptr<IdType>();
 
  // if (cusparse_available<IdType>()) { // cusparse
  //   std::cout << "DTPE check " << B.indices->dtype << std::endl;
  //   cusparse::CusparseCsrgeam2<DType, int32_t>(
  //     A_weights->ctx, 
  //     A, static_cast<DType*>(A_weights->data),
  //     A, static_cast<DType*>(A_weights->data),
  //     A.indices->dtype, A_weights->dtype);
  // } else {  
  //     LOG(FATAL) << "DOBULECHECK!! cuSPARSE SpGEMM does not support int64_t.";
  // }
  if (cusparse_available<IdType>()) { // cusparse
    std::cout << "DTYPE check " << B.indices->dtype << " " <<  A_weights->dtype <<std::endl;
    cusparse::CusparseCsrgeam2<DType, int32_t>(
      A_weights->ctx, 
      tmpA, dA_values_data,
      tmpB, dB_values_data,
      A.indices->dtype, A_weights->dtype);
  } else {  
      LOG(FATAL) << "Not supported";
  }

  IdArray C_indices = IdArray::Empty({nnz}, list_A[0].indices->dtype, list_A[0].indices->ctx);
  NDArray C_weights = NDArray::Empty({nnz}, list_A_weights[0]->dtype, list_A_weights[0]->ctx);

  return {CSRMatrix(M, N, C_indptr, C_indices), C_weights};
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
