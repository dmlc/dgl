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

extern "C" uint32_t aclrtlaunch_range_i32(
    uint32_t blockDim, aclrtStream stream, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_range_i64(
    uint32_t blockDim, aclrtStream stream, void* dst, void* tiling);

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

// 持久化 tiling buffer：分配一次、复用，消除 per-call 的 malloc/free/sync
// 同一流上的 aclrtMemcpy → kernel launch 是有序的，无需同步
struct TilingData {
  uint32_t n;
  uint32_t low_hi;
  uint32_t low_lo;
};

static thread_local void* tiling_dev_buf = nullptr;

static void* GetTilingDev(const TilingData& tiling) {
  if (tiling_dev_buf == nullptr) {
    ASCEND_CALL(aclrtMalloc(&tiling_dev_buf, sizeof(TilingData),
                             ACL_MEM_MALLOC_HUGE_FIRST));
  }
  ASCEND_CALL(aclrtMemcpy(tiling_dev_buf, sizeof(TilingData),
                           &tiling, sizeof(TilingData),
                           ACL_MEMCPY_HOST_TO_DEVICE));
  return tiling_dev_buf;
}

// 根据数据量和可用 Vector Core 数动态计算 block_dim
static uint32_t CalcBlockDim(int32_t device_id, int64_t length) {
  int64_t available_cores = 8;
  aclError ret = aclrtGetDeviceInfo(
      static_cast<uint32_t>(device_id),
      ACL_DEV_ATTR_VECTOR_CORE_NUM, &available_cores);
  if (ret != ACL_SUCCESS || available_cores <= 0) {
    available_cores = 8;
  }
  const uint32_t MIN_ELEMENTS_PER_CORE = 1024;
  uint32_t needed = (static_cast<uint32_t>(length) + MIN_ELEMENTS_PER_CORE - 1)
                    / MIN_ELEMENTS_PER_CORE;
  uint32_t cores = static_cast<uint32_t>(available_cores);
  uint32_t block_dim = needed < cores ? needed : cores;
  return block_dim > 0 ? block_dim : 1;
}

template <>
IdArray Range<kDGLAscend, int32_t>(int32_t low, int32_t high, DGLContext ctx) {
  CHECK(high >= low) << "high must be bigger than low";
  CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  const int64_t length = static_cast<int64_t>(high) - static_cast<int64_t>(low);
  IdArray ret = NewIdArray(length, ctx, sizeof(int32_t) * 8);
  if (length == 0) return ret;

  TilingData tiling;
  tiling.n = static_cast<uint32_t>(length);
  int64_t low_val = static_cast<int64_t>(low);
  tiling.low_hi = static_cast<uint32_t>(static_cast<uint64_t>(low_val) >> 32);
  tiling.low_lo = static_cast<uint32_t>(static_cast<uint64_t>(low_val));

  void* tiling_dev = GetTilingDev(tiling);

  auto stream = dgl::runtime::getCurrentAscendStream();
  uint32_t block_dim = CalcBlockDim(ctx.device_id, length);
  aclrtlaunch_range_i32(block_dim, stream, ret->data, tiling_dev);

  return ret;
}

template <>
IdArray Range<kDGLAscend, int64_t>(int64_t low, int64_t high, DGLContext ctx) {
  CHECK(high >= low) << "high must be bigger than low";
  CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  const int64_t length = high - low;
  IdArray ret = NewIdArray(length, ctx, sizeof(int64_t) * 8);
  if (length == 0) return ret;

  TilingData tiling;
  tiling.n = static_cast<uint32_t>(length);
  tiling.low_hi = static_cast<uint32_t>(static_cast<uint64_t>(low) >> 32);
  tiling.low_lo = static_cast<uint32_t>(static_cast<uint64_t>(low));

  void* tiling_dev = GetTilingDev(tiling);

  auto stream = dgl::runtime::getCurrentAscendStream();
  uint32_t block_dim = CalcBlockDim(ctx.device_id, length);
  aclrtlaunch_range_i64(block_dim, stream, ret->data, tiling_dev);

  return ret;
}

#else  // DGL_USE_ASCEND

template <>
IdArray Range<kDGLAscend, int32_t>(int32_t low, int32_t high, DGLContext ctx) {
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
}

template <>
IdArray Range<kDGLAscend, int64_t>(int64_t low, int64_t high, DGLContext ctx) {
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
}

#endif  // DGL_USE_ASCEND

}  // namespace impl
}  // namespace aten
}  // namespace dgl
