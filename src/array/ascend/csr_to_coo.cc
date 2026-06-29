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

extern "C" uint32_t aclrtlaunch_csr_to_coo_int32(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* ret_row, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_to_coo_int64(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* ret_row, void* tiling);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}
}
#endif

#include <dgl/array.h>
#include "../array_op.h"

namespace dgl {
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRToCOO(CSRMatrix csr) {
#ifdef DGL_USE_ASCEND
  auto ctx = csr.indptr->ctx;
  CHECK(ctx.device_type == kDGLAscend);

  const int64_t nnz = csr.indices->shape[0];
  NDArray ret_row =
      NDArray::Empty({nnz}, csr.indices->dtype, ctx);

  ASCEND_CALL(aclrtSetDevice(ctx.device_id));
  auto stream = dgl::runtime::getCurrentAscendStream();

  uint32_t num_rows = static_cast<uint32_t>(csr.indptr->shape[0] - 1);
  uint32_t nnz_u32 = static_cast<uint32_t>(nnz);
  uint32_t block_dim = 1;
  uint32_t tiling_data[2] = {num_rows, nnz_u32};
  void* tiling_dev = nullptr;
  ASCEND_CALL(aclrtMalloc(&tiling_dev, sizeof(tiling_data), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(tiling_dev, sizeof(tiling_data),
                           tiling_data, sizeof(tiling_data),
                           ACL_MEMCPY_HOST_TO_DEVICE));

  if (std::is_same<IdType, int32_t>::value) {
    aclError err = aclrtlaunch_csr_to_coo_int32(
        block_dim, stream, csr.indptr->data, ret_row->data, tiling_dev);
    CHECK(err == ACL_SUCCESS) << "csr_to_coo_int32 failed: " << err;
  } else {
    aclError err = aclrtlaunch_csr_to_coo_int64(
        block_dim, stream, csr.indptr->data, ret_row->data, tiling_dev);
    CHECK(err == ACL_SUCCESS) << "csr_to_coo_int64 failed: " << err;
  }

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(tiling_dev));

  // col and data are zero-copy views of csr.indices and csr.data
  return COOMatrix(
      csr.num_rows, csr.num_cols, ret_row, csr.indices, csr.data, true,
      csr.sorted);
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}

template COOMatrix CSRToCOO<kDGLAscend, int32_t>(CSRMatrix);
template COOMatrix CSRToCOO<kDGLAscend, int64_t>(CSRMatrix);

}  // namespace impl
}  // namespace aten
}  // namespace dgl

