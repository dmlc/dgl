/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/coo_sort.cc
 * \brief Sort COO index
 */
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
COOMatrix COOSort(COOMatrix coo, bool sort_column) {
  CHECK(sizeof(IdType) == 4) << "CUDA COOSort does not support int64.";
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  auto device = runtime::DeviceAPI::Get(coo.row->ctx);
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, thr_entry->stream));


  NDArray row = coo.row.CopyTo(coo.row->ctx);
  NDArray col = coo.col.CopyTo(coo.col->ctx);
  NDArray data;
  if (aten::IsNullArray(coo.data)) {
    // create the index array
    data = aten::Range(0, row->shape[0], row->dtype.bits, row->ctx);
  } else {
    data = coo.data.CopyTo(coo.data->ctx);
  }
  int32_t* row_ptr = static_cast<int32_t*>(row->data);
  int32_t* col_ptr = static_cast<int32_t*>(col->data);
  int32_t* data_ptr = static_cast<int32_t*>(data->data);

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

  if (sort_column) {
    // First create a row indptr array and then call csrsort
    int32_t* indptr = static_cast<int32_t*>(
        device->AllocWorkspace(row->ctx, (coo.num_rows + 1) * sizeof(IdType)));
    CUSPARSE_CALL(cusparseXcoo2csr(
          thr_entry->cusparse_handle,
          row_ptr,
          row->shape[0],
          coo.num_rows,
          indptr,
          CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CALL(cusparseXcsrsort_bufferSizeExt(
          thr_entry->cusparse_handle,
          coo.num_rows,
          coo.num_cols,
          row->shape[0],
          indptr,
          col_ptr,
          &workspace_size));
    void* workspace = device->AllocWorkspace(row->ctx, workspace_size);
    cusparseMatDescr_t descr;
    CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
    CUSPARSE_CALL(cusparseXcsrsort(
          thr_entry->cusparse_handle,
          coo.num_rows,
          coo.num_cols,
          row->shape[0],
          descr,
          indptr,
          col_ptr,
          data_ptr,
          workspace));
    CUSPARSE_CALL(cusparseDestroyMatDescr(descr));
    device->FreeWorkspace(row->ctx, workspace);
    device->FreeWorkspace(row->ctx, indptr);
  }

  return COOMatrix(coo.num_rows, coo.num_cols,
                   row, col, data, true, sort_column);
}

template COOMatrix COOSort<kDLGPU, int32_t>(COOMatrix coo, bool sort_column);
template COOMatrix COOSort<kDLGPU, int64_t>(COOMatrix coo, bool sort_column);


}  // namespace impl
}  // namespace aten
}  // namespace dgl
