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

#include "full_tiling.h"

// NOTE: the AscendC packaging flow's generated host stubs take tiling
// struct args by POINTER (device side still receives them by value).
// Keep the declaration in sync with the generated stub ABI.
extern "C" uint32_t aclrtlaunch_full_i32(
    uint32_t blockDim, aclrtStream stream, void* dst,
    const FullTilingData* tiling);
extern "C" uint32_t aclrtlaunch_full_i64(
    uint32_t blockDim, aclrtStream stream, void* dst,
    const FullTilingData* tiling);
extern "C" uint32_t aclrtlaunch_full_f32(
    uint32_t blockDim, aclrtStream stream, void* dst,
    const FullTilingData* tiling);
extern "C" uint32_t aclrtlaunch_full_f64(
    uint32_t blockDim, aclrtStream stream, void* dst,
    const FullTilingData* tiling);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}
}
#endif

#include <dgl/array.h>
#include <dgl/runtime/device_api.h>

#include "../array_op.h"

#include <cstring>

namespace dgl {
namespace aten {
namespace impl {

#ifdef DGL_USE_ASCEND

namespace {

// Bit-cast the typed fill value into the tiling's 64-bit pattern slot.
// 32-bit types use the low 32 bits (two's complement / IEEE bits kept).
template <typename CppType>
uint64_t BitsOf(CppType val) {
  if constexpr (sizeof(CppType) <= sizeof(uint32_t)) {
    uint32_t bits = 0;
    std::memcpy(&bits, &val, sizeof(CppType));
    return static_cast<uint64_t>(bits);
  } else {
    uint64_t bits = 0;
    std::memcpy(&bits, &val, sizeof(CppType));
    return bits;
  }
}

// 根据数据量和可用 Vector Core 数动态计算 block_dim（与 range.cc 一致）。
// Full 的 kernel 按元素连续分片，每核至少分到一个 tile 才有意义。
static uint32_t CalcBlockDim(int32_t device_id, int64_t length) {
  int64_t available_cores = 8;
  aclError ret = aclrtGetDeviceInfo(
      static_cast<uint32_t>(device_id),
      ACL_DEV_ATTR_VECTOR_CORE_NUM, &available_cores);
  if (ret != ACL_SUCCESS || available_cores <= 0) {
    available_cores = 8;
  }
  const int64_t needed =
      (length + static_cast<int64_t>(kFullTileLength) - 1) / kFullTileLength;
  const uint32_t cores = static_cast<uint32_t>(available_cores);
  uint32_t block_dim =
      needed < static_cast<int64_t>(cores) ? static_cast<uint32_t>(needed)
                                           : cores;
  return block_dim > 0 ? block_dim : 1;
}

template <typename CppType>
using LaunchFullFn = uint32_t (*)(uint32_t, aclrtStream, void*,
                                  const FullTilingData*);

template <typename CppType, LaunchFullFn<CppType> LaunchFn>
NDArray FullAscendImpl(CppType val, int64_t length, DGLContext ctx) {
  CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));
  NDArray ret =
      NDArray::Empty({length}, DGLDataTypeTraits<CppType>::dtype, ctx);
  if (length == 0) return ret;
  FullTilingData tiling;
  tiling.n = static_cast<uint64_t>(length);
  tiling.val = BitsOf<CppType>(val);
  auto stream = dgl::runtime::getCurrentAscendStream();
  const uint32_t block_dim = CalcBlockDim(ctx.device_id, length);
  LaunchFn(block_dim, stream, ret->data, &tiling);
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  return ret;
}
}  // namespace

#define ASCEND_FULL_SPEC(CppType, launch_name)                            \
  template <>                                                             \
  NDArray Full<kDGLAscend, CppType>(CppType val, int64_t length,         \
                                    DGLContext ctx) {                     \
    return FullAscendImpl<CppType, launch_name>(val, length, ctx);       \
  }

ASCEND_FULL_SPEC(int32_t, aclrtlaunch_full_i32)
ASCEND_FULL_SPEC(int64_t, aclrtlaunch_full_i64)
ASCEND_FULL_SPEC(float, aclrtlaunch_full_f32)
ASCEND_FULL_SPEC(double, aclrtlaunch_full_f64)

#else  // DGL_USE_ASCEND

template <>
NDArray Full<kDGLAscend, int32_t>(int32_t, int64_t, DGLContext) {
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
}
template <>
NDArray Full<kDGLAscend, int64_t>(int64_t, int64_t, DGLContext) {
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
}
template <>
NDArray Full<kDGLAscend, float>(float, int64_t, DGLContext) {
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
}
template <>
NDArray Full<kDGLAscend, double>(double, int64_t, DGLContext) {
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
}

#endif  // DGL_USE_ASCEND

}  // namespace impl
}  // namespace aten
}  // namespace dgl
