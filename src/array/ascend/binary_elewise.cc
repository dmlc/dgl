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

extern "C" uint32_t aclrtlaunch_binary_l_lt_i32(
    uint32_t blockDim, aclrtStream stream, void* lhs, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_binary_l_gt_i32(
    uint32_t blockDim, aclrtStream stream, void* lhs, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_binary_l_le_i32(
    uint32_t blockDim, aclrtStream stream, void* lhs, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_binary_l_ge_i32(
    uint32_t blockDim, aclrtStream stream, void* lhs, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_binary_l_eq_i32(
    uint32_t blockDim, aclrtStream stream, void* lhs, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_binary_l_ne_i32(
    uint32_t blockDim, aclrtStream stream, void* lhs, void* dst, void* tiling);

extern "C" uint32_t aclrtlaunch_binary_l_lt_i64(
    uint32_t blockDim, aclrtStream stream, void* lhs, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_binary_l_gt_i64(
    uint32_t blockDim, aclrtStream stream, void* lhs, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_binary_l_le_i64(
    uint32_t blockDim, aclrtStream stream, void* lhs, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_binary_l_ge_i64(
    uint32_t blockDim, aclrtStream stream, void* lhs, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_binary_l_eq_i64(
    uint32_t blockDim, aclrtStream stream, void* lhs, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_binary_l_ne_i64(
    uint32_t blockDim, aclrtStream stream, void* lhs, void* dst, void* tiling);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}
}
#endif

#include <dgl/array.h>
#include <dgl/runtime/device_api.h>

#include "../array_op.h"
#include "../arith.h"

namespace dgl {
namespace aten {
namespace impl {

namespace {

template <typename IdType, typename Op>
struct BinaryElewiseLKernelLauncher;

#define DEFINE_L_SPECIALIZATION(type, type_suffix, op_name, op)            \
  template <>                                                              \
  struct BinaryElewiseLKernelLauncher<type, arith::op> {                   \
    static void Launch(uint32_t blockDim, aclrtStream stream,              \
                       void* lhs, void* dst, void* tiling) {              \
      aclrtlaunch_binary_l_##op_name##_##type_suffix(                      \
          blockDim, stream, lhs, dst, tiling);                            \
    }                                                                      \
  };

DEFINE_L_SPECIALIZATION(int32_t, i32, lt, LT)
DEFINE_L_SPECIALIZATION(int32_t, i32, gt, GT)
DEFINE_L_SPECIALIZATION(int32_t, i32, le, LE)
DEFINE_L_SPECIALIZATION(int32_t, i32, ge, GE)
DEFINE_L_SPECIALIZATION(int32_t, i32, eq, EQ)
DEFINE_L_SPECIALIZATION(int32_t, i32, ne, NE)

DEFINE_L_SPECIALIZATION(int64_t, i64, lt, LT)
DEFINE_L_SPECIALIZATION(int64_t, i64, gt, GT)
DEFINE_L_SPECIALIZATION(int64_t, i64, le, LE)
DEFINE_L_SPECIALIZATION(int64_t, i64, ge, GE)
DEFINE_L_SPECIALIZATION(int64_t, i64, eq, EQ)
DEFINE_L_SPECIALIZATION(int64_t, i64, ne, NE)

}  // anonymous namespace

template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdArray lhs, IdType rhs) {
#ifdef DGL_USE_ASCEND
  auto ctx = lhs->ctx;
  CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  const int64_t len = lhs->shape[0];
  IdArray ret = NewIdArray(lhs->shape[0], lhs->ctx, lhs->dtype.bits);
  if (len == 0) return ret;

  struct TilingData {
    uint32_t n;
    uint32_t scalar_hi;
    uint32_t scalar_lo;
  };
  int64_t scalar_val = static_cast<int64_t>(rhs);
  TilingData tiling_host;
  tiling_host.n = static_cast<uint32_t>(len);
  tiling_host.scalar_hi = static_cast<uint32_t>(static_cast<uint64_t>(scalar_val) >> 32);
  tiling_host.scalar_lo = static_cast<uint32_t>(static_cast<uint64_t>(scalar_val));

  void* tiling_dev = nullptr;
  ASCEND_CALL(aclrtMalloc(&tiling_dev, sizeof(TilingData), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(tiling_dev, sizeof(TilingData),
                           &tiling_host, sizeof(TilingData),
                           ACL_MEMCPY_HOST_TO_DEVICE));

  auto stream = dgl::runtime::getCurrentAscendStream();
  uint32_t block_dim = 1;
  BinaryElewiseLKernelLauncher<IdType, Op>::Launch(
      block_dim, stream, lhs->data, ret->data, tiling_dev);

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  if (tiling_dev) ASCEND_CALL(aclrtFree(tiling_dev));
  return ret;
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}

#define INSTANTIATE_L(type, op)                                            \
  template IdArray BinaryElewise<kDGLAscend, type, arith::op>(            \
      IdArray lhs, type rhs);

INSTANTIATE_L(int32_t, LT)
INSTANTIATE_L(int32_t, GT)
INSTANTIATE_L(int32_t, LE)
INSTANTIATE_L(int32_t, GE)
INSTANTIATE_L(int32_t, EQ)
INSTANTIATE_L(int32_t, NE)

INSTANTIATE_L(int64_t, LT)
INSTANTIATE_L(int64_t, GT)
INSTANTIATE_L(int64_t, LE)
INSTANTIATE_L(int64_t, GE)
INSTANTIATE_L(int64_t, EQ)
INSTANTIATE_L(int64_t, NE)

}  // namespace impl
}  // namespace aten
}  // namespace dgl

