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

extern "C" uint32_t aclrtlaunch_csr_get_row_nnz_int32(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* rows, void* out, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_get_row_nnz_int64(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* rows, void* out, void* tiling);

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
int64_t CSRGetRowNNZ(CSRMatrix csr, int64_t row) {
  if (row >= csr.num_rows) return 0;
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  IdType start = 0, end = 0;
  auto device = runtime::DeviceAPI::Get(csr.indptr->ctx);
  device->CopyDataFromTo(
      indptr_data + row, 0, &start, 0, sizeof(IdType),
      csr.indptr->ctx, DGLContext{kDGLCPU, 0}, csr.indptr->dtype);
  device->CopyDataFromTo(
      indptr_data + row + 1, 0, &end, 0, sizeof(IdType),
      csr.indptr->ctx, DGLContext{kDGLCPU, 0}, csr.indptr->dtype);
  return static_cast<int64_t>(end - start);
}

template int64_t CSRGetRowNNZ<kDGLAscend, int32_t>(CSRMatrix, int64_t);
template int64_t CSRGetRowNNZ<kDGLAscend, int64_t>(CSRMatrix, int64_t);

template <DGLDeviceType XPU, typename IdType>
NDArray CSRGetRowNNZ(CSRMatrix csr, NDArray rows) {
#ifdef DGL_USE_ASCEND
  auto ctx = rows->ctx;
  CHECK(ctx.device_type == kDGLAscend)
      << "Expected Ascend device context for CSRGetRowNNZ";

  const int64_t len = rows->shape[0];
  NDArray rst = NDArray::Empty({len}, rows->dtype, ctx);
  if (len == 0) return rst;

  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  auto stream = dgl::runtime::getCurrentAscendStream();

  uint32_t n = static_cast<uint32_t>(len);
  uint32_t orig_rows = static_cast<uint32_t>(csr.num_rows);
  uint32_t block_dim = 1;
  uint32_t tiling_data[2] = {n, orig_rows};
  void* tiling_dev = nullptr;
  ASCEND_CALL(aclrtMalloc(&tiling_dev, sizeof(tiling_data), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(tiling_dev, sizeof(tiling_data),
                           tiling_data, sizeof(tiling_data),
                           ACL_MEMCPY_HOST_TO_DEVICE));

  if (std::is_same<IdType, int32_t>::value) {
    aclError err = aclrtlaunch_csr_get_row_nnz_int32(
        block_dim, stream, csr.indptr->data, rows->data, rst->data, tiling_dev);
    CHECK(err == ACL_SUCCESS) << "csr_get_row_nnz_int32 launch failed: " << err;
  } else {
    aclError err = aclrtlaunch_csr_get_row_nnz_int64(
        block_dim, stream, csr.indptr->data, rows->data, rst->data, tiling_dev);
    CHECK(err == ACL_SUCCESS) << "csr_get_row_nnz_int64 launch failed: " << err;
  }

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  if (tiling_dev) ASCEND_CALL(aclrtFree(tiling_dev));

  return rst;
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}

template NDArray CSRGetRowNNZ<kDGLAscend, int32_t>(CSRMatrix, NDArray);
template NDArray CSRGetRowNNZ<kDGLAscend, int64_t>(CSRMatrix, NDArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl

