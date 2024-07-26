/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/csr_transpose.cc
 * @brief CSR transpose (convert to CSC)
 */
#include <dgl/array.h>

#include "../../runtime/cuda/cuda_common.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRTranspose(CSRMatrix csr) {
  LOG(FATAL) << "Unreachable codes";
  return {};
}

template <>
CSRMatrix CSRTranspose<kDGLCUDA, int32_t>(CSRMatrix csr) {
#if CUDART_VERSION < 12000
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, stream));

  NDArray indptr = csr.indptr, indices = csr.indices, data = csr.data;
  const int64_t nnz = indices->shape[0];
  const auto& ctx = indptr->ctx;
  const auto bits = indptr->dtype.bits;
  if (aten::IsNullArray(data)) data = aten::Range(0, nnz, bits, ctx);
  const int32_t* indptr_ptr = static_cast<int32_t*>(indptr->data);
  const int32_t* indices_ptr = static_cast<int32_t*>(indices->data);
  const void* data_ptr = data->data;

  // (BarclayII) csr2csc doesn't seem to clear the content of cscColPtr if nnz
  // == 0. We need to do it ourselves.
  NDArray t_indptr = aten::Full(0, csr.num_cols + 1, bits, ctx);
  NDArray t_indices = aten::NewIdArray(nnz, ctx, bits);
  NDArray t_data = aten::NewIdArray(nnz, ctx, bits);
  int32_t* t_indptr_ptr = static_cast<int32_t*>(t_indptr->data);
  int32_t* t_indices_ptr = static_cast<int32_t*>(t_indices->data);
  void* t_data_ptr = t_data->data;

#if CUDART_VERSION >= 10010
  auto device = runtime::DeviceAPI::Get(csr.indptr->ctx);
  // workspace
  size_t workspace_size;
  CUSPARSE_CALL(cusparseCsr2cscEx2_bufferSize(
      thr_entry->cusparse_handle, csr.num_rows, csr.num_cols, nnz, data_ptr,
      indptr_ptr, indices_ptr, t_data_ptr, t_indptr_ptr, t_indices_ptr,
      CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
      CUSPARSE_CSR2CSC_ALG1,  // see cusparse doc for reference
      &workspace_size));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);
  CUSPARSE_CALL(cusparseCsr2cscEx2(
      thr_entry->cusparse_handle, csr.num_rows, csr.num_cols, nnz, data_ptr,
      indptr_ptr, indices_ptr, t_data_ptr, t_indptr_ptr, t_indices_ptr,
      CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
      CUSPARSE_CSR2CSC_ALG1,  // see cusparse doc for reference
      workspace));
  device->FreeWorkspace(ctx, workspace);
#else
  CUSPARSE_CALL(cusparseScsr2csc(
      thr_entry->cusparse_handle, csr.num_rows, csr.num_cols, nnz,
      static_cast<const float*>(data_ptr), indptr_ptr, indices_ptr,
      static_cast<float*>(t_data_ptr), t_indices_ptr, t_indptr_ptr,
      CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO));
#endif

  return CSRMatrix(
      csr.num_cols, csr.num_rows, t_indptr, t_indices, t_data, false);
#else
  return COOToCSR(COOTranspose(CSRToCOO(csr, false)));
#endif
}

template <>
CSRMatrix CSRTranspose<kDGLCUDA, int64_t>(CSRMatrix csr) {
  return COOToCSR(COOTranspose(CSRToCOO(csr, false)));
}

template CSRMatrix CSRTranspose<kDGLCUDA, int32_t>(CSRMatrix csr);
template CSRMatrix CSRTranspose<kDGLCUDA, int64_t>(CSRMatrix csr);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
