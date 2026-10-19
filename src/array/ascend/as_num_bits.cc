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

extern "C" uint32_t aclrtlaunch_as_num_bits_i32_to_i64(
    uint32_t blockDim, aclrtStream stream, void* src, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_as_num_bits_i64_to_i32(
    uint32_t blockDim, aclrtStream stream, void* src, void* dst, void* tiling);

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

#ifdef DGL_USE_ASCEND

#define ASCEND_AS_NUM_BITS_IMPL(type)                                          \
  template <>                                                                  \
  IdArray AsNumBits<kDGLAscend, type>(IdArray arr, uint8_t bits) {             \
    auto ctx = arr->ctx;                                                       \
    CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context";  \
    ASCEND_CALL(aclrtSetDevice(ctx.device_id));                                 \
    const int64_t len = arr->shape[0];                                         \
    IdArray ret = NewIdArray(len, arr->ctx, bits);                             \
    if (len == 0) return ret;                                                  \
    uint32_t n_host = static_cast<uint32_t>(len);                              \
    void* n_dev = nullptr;                                                     \
    ASCEND_CALL(aclrtMalloc(&n_dev, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST)); \
    ASCEND_CALL(aclrtMemcpy(n_dev, sizeof(uint32_t),                           \
                             &n_host, sizeof(uint32_t),                        \
                             ACL_MEMCPY_HOST_TO_DEVICE));                      \
    auto stream = dgl::runtime::getCurrentAscendStream();                     \
    uint32_t block_dim = 1; \
    if (bits == 32) {                                                          \
      aclrtlaunch_as_num_bits_i64_to_i32(block_dim, stream, arr->data, ret->data, n_dev); \
    } else {                                                                   \
      aclrtlaunch_as_num_bits_i32_to_i64(block_dim, stream, arr->data, ret->data, n_dev); \
    }                                                                          \
    ASCEND_CALL(aclrtSynchronizeStream(stream));                              \
    if (n_dev) ASCEND_CALL(aclrtFree(n_dev));                                  \
    return ret;                                                                \
  }

ASCEND_AS_NUM_BITS_IMPL(int32_t)
ASCEND_AS_NUM_BITS_IMPL(int64_t)

#else  // DGL_USE_ASCEND

template <>
IdArray AsNumBits<kDGLAscend, int32_t>(IdArray arr, uint8_t bits) {
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
}

template <>
IdArray AsNumBits<kDGLAscend, int64_t>(IdArray arr, uint8_t bits) {
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
}

#endif  // DGL_USE_ASCEND

}  // namespace impl
}  // namespace aten
}  // namespace dgl

