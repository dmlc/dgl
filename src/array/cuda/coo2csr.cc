/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/coo2csr.cc
 * \brief COO2CSR
 */
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
CSRMatrix COOToCSR(COOMatrix coo) {
  CHECK(sizeof(IdType) == 4) << "CUDA COOToCSR does not support int64.";
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
  int32_t* data_ptr = aten::IsNullArray(data) ? nullptr : static_cast<int32_t*>(data->data);

  if (!coo.row_sorted) {
    // make a copy of row and col because sort is done in-place
    row = row.CopyTo(row->ctx);
    col = col.CopyTo(col->ctx);
    row_ptr = static_cast<int32_t*>(row->data);
    col_ptr = static_cast<int32_t*>(col->data);
    if (aten::IsNullArray(data)) {
      // create the index array
      data = aten::Range(0, row->shape[0], row->dtype.bits, row->ctx);
      data_ptr = static_cast<int32_t*>(data->data);
    }
    // sort row
    size_t workspace_size = 0;
    CUSPARSE_CALL(cusparseXcoosort_bufferSizeExt(
        thr_entry->cusparse_handle,
        coo.num_rows, coo.num_cols,
        row->shape[0],
        row_ptr,
        col_ptr,
        &workspace_size));
    void* workspace = device->AllocWorkspace(row->ctx, workspace_size);
    CUSPARSE_CALL(cusparseXcoosortByRow(
        thr_entry->cusparse_handle,
        coo.num_rows, coo.num_cols,
        row->shape[0],
        row_ptr,
        col_ptr,
        data_ptr,
        workspace));
    device->FreeWorkspace(row->ctx, workspace);
  }

  NDArray indptr = aten::NewIdArray(coo.num_rows + 1, row->ctx, row->dtype.bits);
  int32_t* indptr_ptr = static_cast<int32_t*>(indptr->data);
  CUSPARSE_CALL(cusparseXcoo2csr(
        thr_entry->cusparse_handle,
        row_ptr,
        row->shape[0],
        coo.num_rows,
        indptr_ptr,
        CUSPARSE_INDEX_BASE_ZERO));

  return CSRMatrix(coo.num_rows, coo.num_cols,
                   indptr, col, data, false);
}

template CSRMatrix COOToCSR<kDLGPU, int32_t>(COOMatrix coo);
template CSRMatrix COOToCSR<kDLGPU, int64_t>(COOMatrix coo);


}  // namespace impl
}  // namespace aten
}  // namespace dgl
