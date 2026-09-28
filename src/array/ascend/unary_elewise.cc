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

extern "C" uint32_t aclrtlaunch_unary_neg_i32(
    uint32_t blockDim, aclrtStream stream, void* x, void* y, void* tiling);
extern "C" uint32_t aclrtlaunch_unary_neg_i64(
    uint32_t blockDim, aclrtStream stream, void* x, void* y, void* tiling);

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
struct UnaryElewiseKernelLauncher;

#define DEFINE_UNARY_SPECIALIZATION(type, type_suffix, op_name, op)      \
  template <>                                                            \
  struct UnaryElewiseKernelLauncher<type, arith::op> {                   \
    static void Launch(uint32_t blockDim, aclrtStream stream,            \
                       void* x, void* y, void* tiling) {                \
      ASCEND_CALL(aclrtlaunch_unary_##op_name##_##type_suffix(           \
          blockDim, stream, x, y, tiling));                             \
    }                                                                    \
  };

DEFINE_UNARY_SPECIALIZATION(int32_t, i32, neg, Neg)
DEFINE_UNARY_SPECIALIZATION(int64_t, i64, neg, Neg)

}  // anonymous namespace

#ifdef DGL_USE_ASCEND

// RAII guard for the UnaryElewise host path: restores the thread's Ascend
// device and frees tiling_dev on every exit path (best-effort, never throws).
class AscendUnaryResourceGuard {
 public:
  AscendUnaryResourceGuard()
      : saved_(aclrtGetDevice(&prev_device_) == ACL_SUCCESS) {}
  ~AscendUnaryResourceGuard() {
    if (tiling_dev != nullptr) {
      aclrtFree(tiling_dev);  // best effort; the happy path frees with CHECK
    }
    if (saved_) {
      aclrtSetDevice(prev_device_);  // best effort; never masks main-flow errors
    }
  }
  void* tiling_dev = nullptr;

 private:
  int32_t prev_device_ = -1;
  bool saved_ = false;
};

// Multi-core launch: query the Vector core num (fall back to 8 on failure),
// gated to at least 1024 elements per core (same as range.cc:59-73).
static uint32_t CalcBlockDim(int32_t device_id, int64_t length) {
  int64_t available_cores = 8;
  aclError ret = aclrtGetDeviceInfo(
      static_cast<uint32_t>(device_id),
      ACL_DEV_ATTR_VECTOR_CORE_NUM, &available_cores);
  if (ret != ACL_SUCCESS || available_cores <= 0) {
    available_cores = 8;
  }
  const uint32_t MIN_ELEMENTS_PER_CORE = 1024;
  uint32_t needed = static_cast<uint32_t>(
      (length + MIN_ELEMENTS_PER_CORE - 1) / MIN_ELEMENTS_PER_CORE);
  uint32_t cores = static_cast<uint32_t>(available_cores);
  uint32_t block_dim = needed < cores ? needed : cores;
  return block_dim > 0 ? block_dim : 1;
}

#endif  // DGL_USE_ASCEND

template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray UnaryElewise(IdArray lhs) {
#ifdef DGL_USE_ASCEND
  auto ctx = lhs->ctx;
  CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context";
  AscendUnaryResourceGuard guard;  // saves the current device, restores on exit
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  const int64_t len = lhs->shape[0];
  CHECK_LE(len, static_cast<int64_t>(UINT32_MAX))
      << "UnaryElewise: element count exceeds the uint32 tiling limit";
  IdArray ret = NewIdArray(lhs->shape[0], lhs->ctx, lhs->dtype.bits);
  if (len == 0) return ret;

  struct TilingData {
    uint32_t n;
  };
  TilingData tiling_host;
  tiling_host.n = static_cast<uint32_t>(len);

  void* tiling_dev = nullptr;
  ASCEND_CALL(aclrtMalloc(&tiling_dev, sizeof(TilingData),
                          ACL_MEM_MALLOC_HUGE_FIRST));
  guard.tiling_dev = tiling_dev;  // freed on every exit path from here on
  ASCEND_CALL(aclrtMemcpy(tiling_dev, sizeof(TilingData),
                          &tiling_host, sizeof(TilingData),
                          ACL_MEMCPY_HOST_TO_DEVICE));

  auto stream = dgl::runtime::getCurrentAscendStream();
  uint32_t block_dim = CalcBlockDim(ctx.device_id, len);
  UnaryElewiseKernelLauncher<IdType, Op>::Launch(
      block_dim, stream, lhs->data, ret->data, tiling_dev);

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  if (tiling_dev) ASCEND_CALL(aclrtFree(tiling_dev));
  guard.tiling_dev = nullptr;  // already freed with CHECK above
  return ret;
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}

#define INSTANTIATE_UNARY(type, op)                                       \
  template IdArray UnaryElewise<kDGLAscend, type, arith::op>(            \
      IdArray lhs);

INSTANTIATE_UNARY(int32_t, Neg)
INSTANTIATE_UNARY(int64_t, Neg)

}  // namespace impl
}  // namespace aten
}  // namespace dgl
