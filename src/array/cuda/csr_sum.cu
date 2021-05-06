/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/spmm.cu
 * \brief SpGEAM C APIs and definitions.
 */
#include <dgl/array.h>
#include <dgl/runtime/device_api.h>
#include "./functor.cuh"
#include "./cusparse_dispatcher.cuh"
#include "../../runtime/cuda/cuda_common.h"

#include <type_traits>

namespace dgl {

using namespace dgl::runtime;

namespace aten {
namespace cusparse {

/*! Cusparse implementation of SpSum on Csr format. */
template <typename DType, typename IdType>
std::pair<CSRMatrix, NDArray> CusparseCsrgeam2(
    const CSRMatrix& A,
    const NDArray A_weights_array,
    const CSRMatrix& B,
    const NDArray B_weights_array) {
  const int m = A.num_rows;
  const int n = A.num_cols;
  const int nnzA = A.indices->shape[0];
  const int nnzB = B.indices->shape[0];
  int nnzC;
  const DType alpha = 1.0;
  const DType beta = 1.0;
  auto ctx = A.indptr->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const DType* A_weights = A_weights_array.Ptr<DType>();
  const DType* B_weights = B_weights_array.Ptr<DType>();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle)
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, thr_entry->stream));

  cusparseMatDescr_t matA, matB, matC;
  cusparse_create_descr(&matA);
  cusparse_create_descr(&matB);
  cusparse_create_descr(&matC);

  cusparseSetPointerMode(thr_entry->cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
  size_t workspace_size = 0;
  /* prepare output C */
  IdArray dC_csrOffsets = IdArray::Empty({m + 1}, A.indptr->dtype, ctx);
  IdType* dC_csrOffsets_data = dC_csrOffsets.Ptr<IdType>();
  IdArray dC_columns;
  NDArray dC_weights;
  IdType* dC_columns_data = dC_columns.Ptr<IdType>();
  DType* dC_weights_data = dC_weights.Ptr<DType>();
  /* prepare buffer */
  CUSPARSE_CALL(CSRGEAM<DType>::bufferSizeExt(
      thr_entry->cusparse_handle, m, n, &alpha,
      matA, nnzA, A_weights,
      A.indptr.Ptr<IdType>(),
      A.indices.Ptr<IdType>(),
      &beta, matB, nnzB, B_weights,
      B.indptr.Ptr<IdType>(),
      B.indices.Ptr<IdType>(),
      matC, dC_weights_data, dC_csrOffsets_data, dC_columns_data,
      &workspace_size));

  void *workspace = device->AllocWorkspace(ctx, workspace_size);
  CUSPARSE_CALL(CSRGEAM<DType>::nnz(thr_entry->cusparse_handle,
      m, n, matA, nnzA,
      A.indptr.Ptr<IdType>(),
      A.indices.Ptr<IdType>(),
      matB, nnzB,
      B.indptr.Ptr<IdType>(),
      B.indices.Ptr<IdType>(),
      matC, dC_csrOffsets_data, &nnzC, workspace));

  std::cout << m << ' ' << n << std::endl;
  std::cout << A.indptr << std::endl;
  std::cout << A.indices << std::endl;
  std::cout << B.indptr << std::endl;
  std::cout << B.indices << std::endl;
  std::cout << nnzA << ' ' << nnzB << ' ' << nnzC << std::endl;
  std::cout << std::is_same<IdType, int32_t>::value << ' ' << std::is_same<DType, float>::value << std::endl;
  std::cout << dC_csrOffsets << std::endl;
  dC_columns = IdArray::Empty({nnzC}, A.indptr->dtype, ctx);
  dC_weights = NDArray::Empty({nnzC}, A_weights_array->dtype, ctx);
  dC_columns_data = dC_columns.Ptr<IdType>();
  dC_weights_data = dC_weights.Ptr<DType>();

  CUSPARSE_CALL(CSRGEAM<DType>::compute(
      thr_entry->cusparse_handle, m, n, &alpha,
      matA, nnzA, A_weights,
      A.indptr.Ptr<IdType>(),
      A.indices.Ptr<IdType>(),
      &beta, matB, nnzB, B_weights,
      B.indptr.Ptr<IdType>(),
      B.indices.Ptr<IdType>(),
      matC, dC_weights_data, dC_csrOffsets_data, dC_columns_data,
      workspace));

  device->FreeWorkspace(ctx, workspace);
  // destroy matrix/vector descriptors
  CUSPARSE_CALL(cusparseDestroyMatDescr(matA));
  CUSPARSE_CALL(cusparseDestroyMatDescr(matB));
  CUSPARSE_CALL(cusparseDestroyMatDescr(matC));
  return {CSRMatrix(A.num_rows, A.num_cols, dC_csrOffsets, dC_columns),
    dC_weights};
}
}  // namespace cusparse

template <int XPU, typename IdType, typename DType>
std::pair<CSRMatrix, NDArray> CSRSum(
    const std::vector<CSRMatrix>& As,
    const std::vector<NDArray>& A_weights) {
  const int64_t M = As[0].num_rows;
  const int64_t N = As[0].num_cols;
  const int64_t n = As.size();

  // Cast 64 bit indices to 32 bit
  std::vector<CSRMatrix> newAs;
  bool cast = false;
  if (As[0].indptr->dtype.bits == 64) {
    newAs.reserve(n);
    for (int i = 0; i < n; ++i)
      newAs.emplace_back(
        As[i].num_rows, As[i].num_cols, AsNumBits(As[i].indptr, 32),
        AsNumBits(As[i].indices, 32), AsNumBits(As[i].data, 32));
    cast = true;
  }
  const std::vector<CSRMatrix> &As_ref = cast ? newAs : As;

  // Reorder weights if A[i] has edge IDs
  std::vector<NDArray> A_weights_reordered(n);
  for (int i = 0; i < n; ++i) {
    if (CSRHasData(As[i]))
      A_weights_reordered[i] = IndexSelect(A_weights[i], As[i].data);
    else
      A_weights_reordered[i] = A_weights[i];
  }

  // Loop and sum
  auto result = std::make_pair(
      CSRMatrix(
        As_ref[0].num_rows, As_ref[0].num_cols,
        As_ref[0].indptr, As_ref[0].indices),
      A_weights_reordered[0]);  // Weights already reordered so we don't need As[0].data
  for (int64_t i = 1; i < n; ++i)
    result = cusparse::CusparseCsrgeam2<DType, int32_t>(
        result.first, result.second, As_ref[i], A_weights_reordered[i]);

  // Cast 32 bit indices back to 64 bit if necessary
  if (cast) {
    CSRMatrix C = result.first;
    return {
      CSRMatrix(C.num_rows, C.num_cols, AsNumBits(C.indptr, 64), AsNumBits(C.indices, 64)),
      result.second};
  } else {
    return result;
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
