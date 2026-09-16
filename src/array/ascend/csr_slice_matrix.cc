#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
#define ASCEND_CALL(func)                                                \
  {                                                                      \
    aclError e = (func);                                                 \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e;              \
  }

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_csr_slice_matrix_mark_valid_col_int32(
    uint32_t blockDim, aclrtStream stream,
    void* cols, void* valid_col, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_slice_matrix_mark_valid_col_int64(
    uint32_t blockDim, aclrtStream stream,
    void* cols, void* valid_col, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_slice_matrix_prefix_int32(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* indices,
    void* rows, void* valid_col,
    void* ret_indptr, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_slice_matrix_prefix_int64(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* indices,
    void* rows, void* valid_col,
    void* ret_indptr, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_slice_matrix_copy_int32(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* indices, void* data_or_null,
    void* rows, void* valid_col,
    void* ret_indptr, void* ret_indices, void* ret_data, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_slice_matrix_copy_int64(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* indices, void* data_or_null,
    void* rows, void* valid_col,
    void* ret_indptr, void* ret_indices, void* ret_data, void* tiling);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}
}
#endif

#include <dgl/array.h>
#include <dgl/runtime/device_api.h>
#include "../array_op.h"

namespace dgl {
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceMatrix(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols) {
#ifdef DGL_USE_ASCEND
  auto ctx = csr.indptr->ctx;
  CHECK(ctx.device_type == kDGLAscend);

  const int64_t new_nrows = rows->shape[0];
  const int64_t new_ncols = cols->shape[0];
  const bool has_data = CSRHasData(csr);

  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  auto stream = dgl::runtime::getCurrentAscendStream();

  // Allocate valid_col lookup table on NPU (int64, initialized by kernel)
  uint32_t num_cols_u32 = static_cast<uint32_t>(csr.num_cols);
  NDArray valid_col_arr = NDArray::Empty(
      {csr.num_cols}, DGLDataType{kDGLInt, 64, 1}, ctx);
  void* valid_col_data = valid_col_arr->data;

  // Allocate ret_indptr (count pass writes to it)
  IdArray ret_indptr =
      IdArray::Empty({new_nrows + 1}, csr.indptr->dtype, ctx);

  // --- Count pass: aclrtMemsetAsync init + 2 kernel launches (mark / prefix) ---
  // aclrtMemsetAsync uses the DMA engine (no AI Core scheduling issue).
  // Stream ordering serializes DMA → mark kernel → prefix kernel, providing a barrier.
  {
    size_t bytes = static_cast<size_t>(csr.num_cols) * sizeof(int64_t);
    ASCEND_CALL(aclrtMemsetAsync(valid_col_data, bytes, 0xFF, bytes, stream));
  }

  // Launch 1: mark valid_col[cols[k]] = k (single block — Ascend HW may not dispatch all blocks for small blockDim)
  {
    uint32_t ncols_u32 = static_cast<uint32_t>(new_ncols);
    uint32_t block_dim_mark = 1;
    uint32_t tiling_data[2] = {ncols_u32, num_cols_u32};
    void* tiling_dev = nullptr;
    ASCEND_CALL(aclrtMalloc(&tiling_dev, sizeof(tiling_data), ACL_MEM_MALLOC_HUGE_FIRST));
    ASCEND_CALL(aclrtMemcpy(tiling_dev, sizeof(tiling_data),
                             tiling_data, sizeof(tiling_data),
                             ACL_MEMCPY_HOST_TO_DEVICE));
    if (std::is_same<IdType, int32_t>::value) {
      aclError err = aclrtlaunch_csr_slice_matrix_mark_valid_col_int32(
          block_dim_mark, stream, cols->data, valid_col_data, tiling_dev);
      CHECK(err == ACL_SUCCESS) << "csr_slice_matrix_mark_valid_col_int32 failed: " << err;
    } else {
      aclError err = aclrtlaunch_csr_slice_matrix_mark_valid_col_int64(
          block_dim_mark, stream, cols->data, valid_col_data, tiling_dev);
      CHECK(err == ACL_SUCCESS) << "csr_slice_matrix_mark_valid_col_int64 failed: " << err;
    }
    ASCEND_CALL(aclrtSynchronizeStream(stream));
    ASCEND_CALL(aclrtFree(tiling_dev));
  }

  // Launch 2: prefix sum (single block, reads valid_col, writes ret_indptr)
  {
    uint32_t nrows_u32 = static_cast<uint32_t>(new_nrows);
    uint32_t orig_nrows_u32 = static_cast<uint32_t>(csr.num_rows);
    uint32_t orig_nnz_u32 = static_cast<uint32_t>(csr.indices->shape[0]);
    uint32_t tiling_data[4] = {nrows_u32, num_cols_u32, orig_nrows_u32, orig_nnz_u32};
    void* tiling_dev = nullptr;
    ASCEND_CALL(aclrtMalloc(&tiling_dev, sizeof(tiling_data), ACL_MEM_MALLOC_HUGE_FIRST));
    ASCEND_CALL(aclrtMemcpy(tiling_dev, sizeof(tiling_data),
                             tiling_data, sizeof(tiling_data),
                             ACL_MEMCPY_HOST_TO_DEVICE));
    if (std::is_same<IdType, int32_t>::value) {
      aclError err = aclrtlaunch_csr_slice_matrix_prefix_int32(
          1, stream,
          csr.indptr->data, csr.indices->data,
          rows->data, valid_col_data,
          ret_indptr->data, tiling_dev);
      CHECK(err == ACL_SUCCESS) << "csr_slice_matrix_prefix_int32 failed: " << err;
    } else {
      aclError err = aclrtlaunch_csr_slice_matrix_prefix_int64(
          1, stream,
          csr.indptr->data, csr.indices->data,
          rows->data, valid_col_data,
          ret_indptr->data, tiling_dev);
      CHECK(err == ACL_SUCCESS) << "csr_slice_matrix_prefix_int64 failed: " << err;
    }
    ASCEND_CALL(aclrtSynchronizeStream(stream));
    ASCEND_CALL(aclrtFree(tiling_dev));
  }

  // Read total nnz from ret_indptr[new_nrows]
  IdType total_nnz = 0;
  auto device = runtime::DeviceAPI::Get(ctx);
  device->CopyDataFromTo(
      static_cast<IdType*>(ret_indptr->data) + new_nrows, 0,
      &total_nnz, 0, sizeof(IdType),
      ctx, DGLContext{kDGLCPU, 0}, ret_indptr->dtype);
  // total_nnz is read from ret_indptr to allocate output arrays

  // Allocate output arrays
  IdArray ret_indices =
      IdArray::Empty({static_cast<int64_t>(total_nnz)}, csr.indices->dtype, ctx);
  DGLDataType data_dtype = has_data ? csr.data->dtype : csr.indices->dtype;
  IdArray ret_data_arr =
      NDArray::Empty({static_cast<int64_t>(total_nnz)}, data_dtype, ctx);

  if (total_nnz == 0) {
    return CSRMatrix(
        new_nrows, new_ncols, ret_indptr, ret_indices, ret_data_arr, csr.sorted);
  }

  // Pass 2 tiling: [nrows, has_data, num_cols, orig_nrows, orig_nnz]
  uint32_t orig_nrows_u32 = static_cast<uint32_t>(csr.num_rows);
  uint32_t orig_nnz_u32 = static_cast<uint32_t>(csr.indices->shape[0]);
  uint32_t pass2_tiling[5] = {
      static_cast<uint32_t>(new_nrows),
      has_data ? 1u : 0u,
      num_cols_u32,
      orig_nrows_u32,
      orig_nnz_u32};
  uint32_t block_dim2 = 1;  // single block (Ascend may not dispatch all blocks)
  void* tiling_dev2 = nullptr;
  ASCEND_CALL(aclrtMalloc(&tiling_dev2, sizeof(pass2_tiling), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(tiling_dev2, sizeof(pass2_tiling),
                           pass2_tiling, sizeof(pass2_tiling),
                           ACL_MEMCPY_HOST_TO_DEVICE));

  void* data_ptr = has_data ? csr.data->data : csr.indices->data;

  if (std::is_same<IdType, int32_t>::value) {
    aclError err = aclrtlaunch_csr_slice_matrix_copy_int32(
        block_dim2, stream,
        csr.indptr->data, csr.indices->data, data_ptr,
        rows->data, valid_col_data,
        ret_indptr->data, ret_indices->data, ret_data_arr->data, tiling_dev2);
    CHECK(err == ACL_SUCCESS) << "csr_slice_matrix_copy_int32 failed: " << err;
  } else {
    aclError err = aclrtlaunch_csr_slice_matrix_copy_int64(
        block_dim2, stream,
        csr.indptr->data, csr.indices->data, data_ptr,
        rows->data, valid_col_data,
        ret_indptr->data, ret_indices->data, ret_data_arr->data, tiling_dev2);
    CHECK(err == ACL_SUCCESS) << "csr_slice_matrix_copy_int64 failed: " << err;
  }

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(tiling_dev2));

  return CSRMatrix(
      new_nrows, new_ncols, ret_indptr, ret_indices, ret_data_arr, csr.sorted);
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}

template CSRMatrix CSRSliceMatrix<kDGLAscend, int32_t>(
    CSRMatrix, runtime::NDArray, runtime::NDArray);
template CSRMatrix CSRSliceMatrix<kDGLAscend, int64_t>(
    CSRMatrix, runtime::NDArray, runtime::NDArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl

