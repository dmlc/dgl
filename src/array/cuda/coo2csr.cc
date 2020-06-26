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
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, thr_entry->stream));

  bool row_sorted = coo.row_sorted;
  bool col_sorted = coo.col_sorted;
  if (!row_sorted) {
    // It is possible that the flag is simply not set (default value is false),
    // so we still perform a linear scan to check the flag.
    std::tie(row_sorted, col_sorted) = COOIsSorted(coo);
  }
  if (!row_sorted) {
    coo = COOSort(coo);
  }

  const int64_t nnz = coo.row->shape[0];
  // TODO(minjie): Many of our current implementation assumes that CSR must have
  //   a data array. This is a temporary workaround. Remove this after:
  //   - The old immutable graph implementation is deprecated.
  //   - The old binary reduce kernel is deprecated.
  if (!COOHasData(coo))
    coo.data = aten::Range(0, nnz, coo.row->dtype.bits, coo.row->ctx);

  NDArray indptr = aten::NewIdArray(coo.num_rows + 1, coo.row->ctx, coo.row->dtype.bits);
  int32_t* indptr_ptr = static_cast<int32_t*>(indptr->data);
  CUSPARSE_CALL(cusparseXcoo2csr(
        thr_entry->cusparse_handle,
        coo.row.Ptr<int32_t>(),
        nnz,
        coo.num_rows,
        indptr_ptr,
        CUSPARSE_INDEX_BASE_ZERO));

  return CSRMatrix(coo.num_rows, coo.num_cols,
                   indptr, coo.col, coo.data, col_sorted);
}

template CSRMatrix COOToCSR<kDLGPU, int32_t>(COOMatrix coo);
template CSRMatrix COOToCSR<kDLGPU, int64_t>(COOMatrix coo);


}  // namespace impl
}  // namespace aten
}  // namespace dgl
