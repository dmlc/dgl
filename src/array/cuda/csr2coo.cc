/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/csr2coo.cc
 * \brief CSR2COO
 */
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
COOMatrix CSRToCOO(CSRMatrix csr) {
  CHECK(sizeof(IdType) == 4) << "CUDA CSRToCOO does not support int64.";
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, thr_entry->stream));

  NDArray indptr = csr.indptr, indices = csr.indices, data = csr.data;
  const int32_t* indptr_ptr = static_cast<int32_t*>(indptr->data);
  NDArray row = aten::NewIdArray(indices->shape[0], indptr->ctx, indptr->dtype.bits);
  int32_t* row_ptr = static_cast<int32_t*>(row->data);

  CUSPARSE_CALL(cusparseXcsr2coo(
      thr_entry->cusparse_handle,
      indptr_ptr,
      indices->shape[0],
      csr.num_rows,
      row_ptr,
      CUSPARSE_INDEX_BASE_ZERO));

  return COOMatrix(csr.num_rows, csr.num_cols,
                   row, indices, data,
                   true, csr.sorted);
}

template COOMatrix CSRToCOO<kDLGPU, int32_t>(CSRMatrix csr);
template COOMatrix CSRToCOO<kDLGPU, int64_t>(CSRMatrix csr);

template <DLDeviceType XPU, typename IdType>
COOMatrix CSRToCOODataAsOrder(CSRMatrix csr) {
  COOMatrix coo = CSRToCOO<XPU, IdType>(csr);
  if (aten::IsNullArray(coo.data))
    return coo;

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  auto device = runtime::DeviceAPI::Get(coo.row->ctx);
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, thr_entry->stream));

  NDArray row = coo.row, col = coo.col, data = coo.data;
  int32_t* row_ptr = static_cast<int32_t*>(row->data);
  int32_t* col_ptr = static_cast<int32_t*>(col->data);
  int32_t* data_ptr = static_cast<int32_t*>(data->data);

  size_t workspace_size = 0;
  CUSPARSE_CALL(cusparseXcoosort_bufferSizeExt(
      thr_entry->cusparse_handle,
      coo.num_rows, coo.num_cols,
      row->shape[0],
      data_ptr,
      row_ptr,
      &workspace_size));
  void* workspace = device->AllocWorkspace(row->ctx, workspace_size);
  CUSPARSE_CALL(cusparseXcoosortByRow(
      thr_entry->cusparse_handle,
      coo.num_rows, coo.num_cols,
      row->shape[0],
      data_ptr,
      row_ptr,
      col_ptr,
      workspace));
  device->FreeWorkspace(row->ctx, workspace);

  return coo;
}

template COOMatrix CSRToCOODataAsOrder<kDLGPU, int32_t>(CSRMatrix csr);
template COOMatrix CSRToCOODataAsOrder<kDLGPU, int64_t>(CSRMatrix csr);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
