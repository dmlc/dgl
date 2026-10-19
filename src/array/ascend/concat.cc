#include <dgl/array.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "../array_op.h"
#include "concat_tiling.h"

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>

#define ASCEND_CALL(func)                                    \
  {                                                          \
    aclError e = (func);                                     \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e; \
  }

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_concat_kernel_f32(
    uint32_t blockDim, aclrtStream stream, void* dst, void* control);
extern "C" uint32_t aclrtlaunch_concat_kernel_f16(
    uint32_t blockDim, aclrtStream stream, void* dst, void* control);
extern "C" uint32_t aclrtlaunch_concat_kernel_i32(
    uint32_t blockDim, aclrtStream stream, void* dst, void* control);
extern "C" uint32_t aclrtlaunch_concat_kernel_i64(
    uint32_t blockDim, aclrtStream stream, void* dst, void* control);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}  // namespace runtime
}  // namespace dgl
#endif

namespace dgl {
namespace aten {
namespace impl {

#ifdef DGL_USE_ASCEND
namespace {

constexpr int64_t kDefaultVectorCoreNum = 48;

int64_t GetVectorCoreNum(int device_id) {
  int64_t core_num = 0;
  if (aclrtGetDeviceInfo(
          device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &core_num) != ACL_SUCCESS ||
      core_num <= 0) {
    core_num = kDefaultVectorCoreNum;
  }
  return core_num;
}

template <typename DType>
struct ConcatKernelLauncher;

template <>
struct ConcatKernelLauncher<float> {
  static uint32_t Launch(
      uint32_t blockDim, aclrtStream stream, void* dst, void* control) {
    return ACLRT_LAUNCH_KERNEL(concat_kernel_f32)(
        blockDim, stream, dst, control);
  }
};

template <>
struct ConcatKernelLauncher<uint16_t> {
  static uint32_t Launch(
      uint32_t blockDim, aclrtStream stream, void* dst, void* control) {
    return ACLRT_LAUNCH_KERNEL(concat_kernel_f16)(
        blockDim, stream, dst, control);
  }
};

template <>
struct ConcatKernelLauncher<int32_t> {
  static uint32_t Launch(
      uint32_t blockDim, aclrtStream stream, void* dst, void* control) {
    return ACLRT_LAUNCH_KERNEL(concat_kernel_i32)(
        blockDim, stream, dst, control);
  }
};

template <>
struct ConcatKernelLauncher<int64_t> {
  static uint32_t Launch(
      uint32_t blockDim, aclrtStream stream, void* dst, void* control) {
    return ACLRT_LAUNCH_KERNEL(concat_kernel_i64)(
        blockDim, stream, dst, control);
  }
};

}  // namespace
#endif

template <typename DType>
void ConcatNDAscend(
    const std::vector<NDArray>& arrays, NDArray out, int64_t axis) {
#ifdef DGL_USE_ASCEND
  const DGLContext ctx = out->ctx;
  CHECK_EQ(ctx.device_type, kDGLAscend) << "Expected Ascend device context";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  int64_t outer = 1;
  int64_t inner = 1;
  int64_t out_numel = 1;
  for (int64_t d = 0; d < out->ndim; ++d) {
    out_numel *= out->shape[d];
    if (d < axis) {
      outer *= out->shape[d];
    } else if (d > axis) {
      inner *= out->shape[d];
    }
  }
  if (out_numel == 0) return;

  CHECK_LT(out_numel, static_cast<int64_t>(1) << 32)
      << "ConcatND: output has " << out_numel
      << " elements, exceeding the 2^32 offset limit of the Ascend kernel";
  CHECK_LE(arrays.size(), std::numeric_limits<uint32_t>::max())
      << "ConcatND: too many input arrays";

  ASCEND_CALL(aclrtMemInvalidate(
      out->data, static_cast<size_t>(out_numel) * sizeof(DType)));
  ASCEND_CALL(aclrtMemset(
      out->data, static_cast<size_t>(out_numel) * sizeof(DType), 0,
      static_cast<size_t>(out_numel) * sizeof(DType)));

  const uint32_t seg_count = static_cast<uint32_t>(outer);
  const uint32_t dst_stride =
      static_cast<uint32_t>(out->shape[axis] * inner);
  const uint32_t chunk_elems =
      static_cast<uint32_t>(CONCAT_CHUNK_BYTES / sizeof(DType));

  std::vector<ConcatInputDesc> inputs;
  inputs.reserve(arrays.size());
  uint64_t total_chunks = 0;
  int64_t prefix = 0;
  for (const NDArray& array : arrays) {
    const int64_t len = array->shape[axis];
    const uint32_t seg_elems = static_cast<uint32_t>(len * inner);
    if (seg_elems != 0) {
      const uint32_t chunks_per_seg = static_cast<uint32_t>(
          (static_cast<uint64_t>(seg_elems) + chunk_elems - 1) /
          chunk_elems);
      CHECK_LE(total_chunks, std::numeric_limits<uint32_t>::max())
          << "ConcatND: too many chunks";
      inputs.push_back(ConcatInputDesc{
          reinterpret_cast<uint64_t>(array->data), seg_elems,
          static_cast<uint32_t>(prefix * inner), chunks_per_seg,
          static_cast<uint32_t>(total_chunks)});
      total_chunks += static_cast<uint64_t>(seg_count) * chunks_per_seg;
    }
    prefix += len;
  }
  CHECK_LE(total_chunks, std::numeric_limits<uint32_t>::max())
      << "ConcatND: too many chunks";
  CHECK(!inputs.empty()) << "ConcatND: non-empty output has no input data";

  const int64_t core_num = GetVectorCoreNum(ctx.device_id);
  const uint32_t block_dim = static_cast<uint32_t>(
      std::max<int64_t>(1, std::min<int64_t>(total_chunks, core_num)));
  ConcatTilingData tiling{
      static_cast<uint32_t>(inputs.size()), seg_count, dst_stride, chunk_elems,
      static_cast<uint32_t>(total_chunks),
      static_cast<uint32_t>((total_chunks + block_dim - 1) / block_dim)};

  const size_t control_bytes =
      sizeof(ConcatTilingData) + inputs.size() * sizeof(ConcatInputDesc);
  std::vector<uint8_t> control(control_bytes);
  std::memcpy(control.data(), &tiling, sizeof(tiling));
  std::memcpy(
      control.data() + sizeof(tiling), inputs.data(),
      inputs.size() * sizeof(ConcatInputDesc));

  void* control_dev = nullptr;
  ASCEND_CALL(
      aclrtMalloc(&control_dev, control_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(
      control_dev, control_bytes, control.data(), control_bytes,
      ACL_MEMCPY_HOST_TO_DEVICE));

  auto stream = dgl::runtime::getCurrentAscendStream();
  const uint32_t launch_error = ConcatKernelLauncher<DType>::Launch(
      block_dim, stream, out->data, control_dev);
  if (launch_error != ACL_SUCCESS) {
    ASCEND_CALL(aclrtFree(control_dev));
    LOG(FATAL) << "ConcatND kernel launch failed with error code: "
               << launch_error;
  }

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(control_dev));
#else
  LOG(FATAL) << "Ascend support is not compiled. Please compile with "
                "-DUSE_ASCEND=ON";
#endif
}

template <>
void ConcatND<kDGLAscend, float>(
    const std::vector<NDArray>& arrays, NDArray out, int64_t dim) {
  ConcatNDAscend<float>(arrays, out, dim);
}

template <>
void ConcatND<kDGLAscend, uint16_t>(
    const std::vector<NDArray>& arrays, NDArray out, int64_t dim) {
  ConcatNDAscend<uint16_t>(arrays, out, dim);
}

template <>
void ConcatND<kDGLAscend, int32_t>(
    const std::vector<NDArray>& arrays, NDArray out, int64_t dim) {
  ConcatNDAscend<int32_t>(arrays, out, dim);
}

template <>
void ConcatND<kDGLAscend, int64_t>(
    const std::vector<NDArray>& arrays, NDArray out, int64_t dim) {
  ConcatNDAscend<int64_t>(arrays, out, dim);
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl
