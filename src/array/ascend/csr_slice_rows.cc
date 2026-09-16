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

extern "C" uint32_t aclrtlaunch_csr_slice_rows_scalar_int32(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* ret_indptr, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_slice_rows_scalar_int64(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* ret_indptr, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_slice_rows_int32(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* indices, void* data_or_null,
    void* rows, void* ret_indptr,
    void* ret_indices, void* ret_data, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_slice_rows_int64(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* indices, void* data_or_null,
    void* rows, void* ret_indptr,
    void* ret_indices, void* ret_data, void* tiling);

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
CSRMatrix CSRSliceRows(CSRMatrix csr, int64_t start, int64_t end) {
#ifdef DGL_USE_ASCEND
  auto ctx = csr.indptr->ctx;
  CHECK(ctx.device_type == kDGLAscend);

  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  const int64_t num_rows = end - start;

  // Read indptr[start] and indptr[end] to get base offset and nnz
  IdType base = 0, end_val = 0;
  auto device = runtime::DeviceAPI::Get(ctx);
  device->CopyDataFromTo(
      const_cast<IdType*>(indptr + start), 0, &base, 0, sizeof(IdType),
      ctx, DGLContext{kDGLCPU, 0}, csr.indptr->dtype);
  device->CopyDataFromTo(
      const_cast<IdType*>(indptr + end), 0, &end_val, 0, sizeof(IdType),
      ctx, DGLContext{kDGLCPU, 0}, csr.indptr->dtype);
  const int64_t nnz = end_val - base;

  // Allocate ret_indptr on NPU
  IdArray ret_indptr =
      IdArray::Empty({num_rows + 1}, csr.indptr->dtype, ctx);

  // Launch kernel to compute shifted indptr
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));
  auto stream = dgl::runtime::getCurrentAscendStream();

  uint32_t tiling_data[2] = {
      static_cast<uint32_t>(start),
      static_cast<uint32_t>(num_rows)};
  void* tiling_dev = nullptr;
  ASCEND_CALL(aclrtMalloc(&tiling_dev, sizeof(tiling_data), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(tiling_dev, sizeof(tiling_data),
                           tiling_data, sizeof(tiling_data),
                           ACL_MEMCPY_HOST_TO_DEVICE));

  if (std::is_same<IdType, int32_t>::value) {
    aclError err = aclrtlaunch_csr_slice_rows_scalar_int32(
        1, stream, csr.indptr->data, ret_indptr->data, tiling_dev);
    CHECK(err == ACL_SUCCESS) << "csr_slice_rows_scalar_int32 failed: " << err;
  } else {
    aclError err = aclrtlaunch_csr_slice_rows_scalar_int64(
        1, stream, csr.indptr->data, ret_indptr->data, tiling_dev);
    CHECK(err == ACL_SUCCESS) << "csr_slice_rows_scalar_int64 failed: " << err;
  }

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(tiling_dev));

  // indices and data use CreateView (zero-copy on NPU)
  IdArray ret_indices = csr.indices.CreateView(
      {nnz}, csr.indices->dtype, base * sizeof(IdType));
  IdArray ret_data;
  if (CSRHasData(csr))
    ret_data = csr.data.CreateView(
        {nnz}, csr.data->dtype, base * sizeof(IdType));
  else
    ret_data = aten::Range(
        base, base + nnz, csr.indptr->dtype.bits, ctx);

  return CSRMatrix(
      num_rows, csr.num_cols, ret_indptr, ret_indices, ret_data, csr.sorted);
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}

template CSRMatrix CSRSliceRows<kDGLAscend, int32_t>(CSRMatrix, int64_t, int64_t);
template CSRMatrix CSRSliceRows<kDGLAscend, int64_t>(CSRMatrix, int64_t, int64_t);

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceRows(CSRMatrix csr, NDArray rows) {
#ifdef DGL_USE_ASCEND
  auto ctx = csr.indptr->ctx;
  CHECK(ctx.device_type == kDGLAscend);

  const int64_t len = rows->shape[0];
  const int64_t nnz_total = csr.indices->shape[0];

  // Allocate output arrays on NPU
  IdArray ret_indptr = IdArray::Empty({len + 1}, csr.indptr->dtype, ctx);
  IdArray ret_indices = IdArray::Empty({nnz_total}, csr.indices->dtype, ctx);
  IdArray ret_data = IdArray::Empty({nnz_total}, csr.indptr->dtype, ctx);

  ASCEND_CALL(aclrtSetDevice(ctx.device_id));
  auto stream = dgl::runtime::getCurrentAscendStream();

  uint32_t tiling_data[3] = {
      static_cast<uint32_t>(len),
      static_cast<uint32_t>(csr.num_cols),
      CSRHasData(csr) ? 1u : 0u};
  void* tiling_dev = nullptr;
  ASCEND_CALL(aclrtMalloc(&tiling_dev, sizeof(tiling_data), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(tiling_dev, sizeof(tiling_data),
                           tiling_data, sizeof(tiling_data),
                           ACL_MEMCPY_HOST_TO_DEVICE));

  void* data_ptr = CSRHasData(csr) ? csr.data->data : csr.indices->data;

  if (std::is_same<IdType, int32_t>::value) {
    aclError err = aclrtlaunch_csr_slice_rows_int32(
        1, stream,
        csr.indptr->data, csr.indices->data, data_ptr,
        rows->data, ret_indptr->data,
        ret_indices->data, ret_data->data, tiling_dev);
    CHECK(err == ACL_SUCCESS) << "csr_slice_rows_int32 failed: " << err;
  } else {
    aclError err = aclrtlaunch_csr_slice_rows_int64(
        1, stream,
        csr.indptr->data, csr.indices->data, data_ptr,
        rows->data, ret_indptr->data,
        ret_indices->data, ret_data->data, tiling_dev);
    CHECK(err == ACL_SUCCESS) << "csr_slice_rows_int64 failed: " << err;
  }

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(tiling_dev));

  // Read actual nnz from ret_indptr[len] to trim output arrays
  IdType total_nnz = 0;
  auto device = runtime::DeviceAPI::Get(ctx);
  device->CopyDataFromTo(
      static_cast<IdType*>(ret_indptr->data) + len, 0,
      &total_nnz, 0, sizeof(IdType),
      ctx, DGLContext{kDGLCPU, 0}, ret_indptr->dtype);

  // Create trimmed views
  IdArray out_indices = ret_indices.CreateView(
      {static_cast<int64_t>(total_nnz)}, ret_indices->dtype, 0);
  IdArray out_data = ret_data.CreateView(
      {static_cast<int64_t>(total_nnz)}, ret_data->dtype, 0);

  return CSRMatrix(
      len, csr.num_cols, ret_indptr, out_indices, out_data, csr.sorted);
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}

template CSRMatrix CSRSliceRows<kDGLAscend, int32_t>(CSRMatrix, NDArray);
template CSRMatrix CSRSliceRows<kDGLAscend, int64_t>(CSRMatrix, NDArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl

