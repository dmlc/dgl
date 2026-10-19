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
  CHECK(ctx.device_type == kDGLAscend)
      << "Expected Ascend device context for CSRToCOO";

  const int64_t nnz = csr.indices->shape[0];
  NDArray ret_row = NDArray::Empty({nnz}, csr.indices->dtype, ctx);
  if (nnz == 0) {
    // col and data are zero-copy views of csr.indices and csr.data
    return COOMatrix(csr.num_rows, csr.num_cols, ret_row, csr.indices,
                     csr.data, true, csr.sorted);
  }

  ASCEND_CALL(aclrtSetDevice(ctx.device_id));
  auto stream = dgl::runtime::getCurrentAscendStream();

  const uint32_t num_rows =
      static_cast<uint32_t>(csr.indptr->shape[0] - 1);
  if (num_rows == 0) {
    return COOMatrix(csr.num_rows, csr.num_cols, ret_row, csr.indices,
                     csr.data, true, csr.sorted);
  }

  // Rows are partitioned across blocks; cap the block count by the number of
  // available vector cores and the row count.
  uint32_t block_dim = 40;
  {
    int64_t core_num = 0;
    if (aclrtGetDeviceInfo(
            ctx.device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &core_num) ==
            ACL_SUCCESS &&
        core_num > 0 && core_num <= 4096) {
      block_dim = static_cast<uint32_t>(core_num);
    }
  }
  if (block_dim > num_rows) block_dim = num_rows;

  uint32_t tiling_data[2] = {num_rows, static_cast<uint32_t>(nnz)};
  void* tiling_dev = nullptr;
  ASCEND_CALL(aclrtMalloc(&tiling_dev, sizeof(tiling_data),
                          ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(tiling_dev, sizeof(tiling_data), tiling_data,
                          sizeof(tiling_data), ACL_MEMCPY_HOST_TO_DEVICE));

  if (std::is_same<IdType, int32_t>::value) {
    aclError err = aclrtlaunch_csr_to_coo_int32(
        block_dim, stream, csr.indptr->data, ret_row->data, tiling_dev);
    CHECK(err == ACL_SUCCESS) << "csr_to_coo_int32 launch failed: " << err;
  } else {
    aclError err = aclrtlaunch_csr_to_coo_int64(
        block_dim, stream, csr.indptr->data, ret_row->data, tiling_dev);
    CHECK(err == ACL_SUCCESS) << "csr_to_coo_int64 launch failed: " << err;
  }

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(tiling_dev));

  // col and data are zero-copy views of csr.indices and csr.data
  return COOMatrix(csr.num_rows, csr.num_cols, ret_row, csr.indices,
                   csr.data, true, csr.sorted);
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
