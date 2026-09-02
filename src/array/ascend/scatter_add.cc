/*
 * @file src/array/ascend/scatter_add.cc
 * @brief Host launcher + template specializations for Scatter Add on Ascend.
 *
 * Mirrors segment_reduce.cc:
 *   - aclrtlaunch_scatter_add: tiling malloc'd on GM + aclrtMemcpy H2D +
 *     launch + aclrtSynchronizeStream + aclrtFree.
 *   - int64 idx -> int32 conversion via CopyTo(CPU) + AsNumBits(32) + CopyTo(NPU),
 *     identical to segment_reduce.cc:196-199.
 *   - dtype coverage: int32/int64 x float32 supported; half(uint16_t)/double
 *     LOG(FATAL) placeholders (matches segment_reduce.cc specializations).
 *
 * NOTE on BFloat16: under the non-CUDA build (USE_CUDA=OFF), the
 * ATEN_FLOAT_TYPE_SWITCH_16BITS macro LOG(FATAL)s Ascend+BFloat16 before it
 * reaches any specialization, so no BFloat16 specialization is required here,
 * consistent with segment_reduce.cc.
 */
#include <dgl/array.h>
#include "../kernel_decl.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
#define ASCEND_CALL(func)                                    \
  {                                                          \
    aclError e = (func);                                     \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e; \
  }

struct ScatterAddTilingData {
  uint32_t N;        // feat rows
  uint32_t M;        // out rows
  uint32_t featDim;  // elements per row
};

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

// The ACLRT_LAUNCH_KERNEL macro expands to aclrtlaunch_<kernel_func_name>.
// The device kernel function is named `kernel_scatter_add` (see
// scatter_add_kernel.cpp), so the host-side launcher symbol is
// `aclrtlaunch_kernel_scatter_add`.
extern "C" uint32_t aclrtlaunch_kernel_scatter_add(
    uint32_t blockDim, aclrtStream stream, void* feat, void* idx,
    void* out, void* tiling);
#endif  // DGL_USE_ASCEND

namespace dgl {
namespace aten {

#ifdef DGL_USE_ASCEND
namespace {

// blockDim = min(N, 40), consistent with segment_reduce.cc:75-76.
template <typename IdType, typename DType>
void LaunchScatterAddKernel(
    int64_t num_items, int64_t num_segments, int64_t feat_dim,
    const void* feat_ptr, const void* idx_ptr, void* out_ptr,
    aclrtStream stream) {
  uint32_t block_dim = static_cast<uint32_t>(
      num_items > 0 ? std::min<int64_t>(num_items, 40) : 1);

  ScatterAddTilingData tiling = {
      static_cast<uint32_t>(num_items),
      static_cast<uint32_t>(num_segments),
      static_cast<uint32_t>(feat_dim)};

  void* tiling_device = nullptr;
  ASCEND_CALL(aclrtMalloc(
      &tiling_device, sizeof(ScatterAddTilingData),
      ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(
      tiling_device, sizeof(ScatterAddTilingData), &tiling,
      sizeof(ScatterAddTilingData), ACL_MEMCPY_HOST_TO_DEVICE));

  aclError launch_err = aclrtlaunch_kernel_scatter_add(
      block_dim, stream, const_cast<void*>(feat_ptr),
      const_cast<void*>(idx_ptr), out_ptr, tiling_device);
  if (launch_err != ACL_SUCCESS) {
    ASCEND_CALL(aclrtFree(tiling_device));
    LOG(FATAL) << "ScatterAdd kernel launch failed with error code: "
               << launch_err;
  }

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(tiling_device));
}

// Shared implementation for int32 idx x float feat.
template <typename IdType, typename DType>
void ScatterAddAscendImpl(NDArray feat, NDArray idx, NDArray out) {
  DGLContext ctx = feat->ctx;
  ASCEND_CALL(
      aclrtSynchronizeDevice());  // ensure PyTorch NPU ops complete before
                                  // touching feat/idx/out
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  int64_t num_items = feat->shape[0];
  int64_t num_segments = out->shape[0];
  int64_t feat_dim = 1;
  for (int i = 1; i < out->ndim; ++i) feat_dim *= out->shape[i];

  // Use the default ACL stream (nullptr). Correctness is ensured by the
  // aclrtSynchronizeDevice() above, not by stream alignment. Matches
  // segment_reduce.cc:139.
  aclrtStream stream = nullptr;

  LaunchScatterAddKernel<IdType, DType>(
      num_items, num_segments, feat_dim, feat->data, idx->data, out->data,
      stream);
}

}  // namespace
#endif  // DGL_USE_ASCEND

// ---- Supported: int32/int64 x float32 ----

template <>
void ScatterAdd<kDGLAscend, int32_t, float>(
    NDArray feat, NDArray idx, NDArray out) {
#ifdef DGL_USE_ASCEND
  ScatterAddAscendImpl<int32_t, float>(feat, idx, out);
#else
  LOG(FATAL) << "Ascend support is not compiled. Please compile with "
                "-DUSE_ASCEND=ON";
#endif
}

template <>
void ScatterAdd<kDGLAscend, int64_t, float>(
    NDArray feat, NDArray idx, NDArray out) {
#ifdef DGL_USE_ASCEND
  // int64 idx -> int32 (AsNumBits is a width-preserving cast, not truncation).
  // Same pattern as segment_reduce.cc:196-199.
  NDArray idx_cpu = idx.CopyTo(DGLContext{kDGLCPU, 0});
  IdArray idx_i32_cpu = aten::AsNumBits(idx_cpu, 32);
  NDArray idx_i32 = idx_i32_cpu.CopyTo(idx->ctx);
  ScatterAddAscendImpl<int32_t, float>(feat, idx_i32, out);
#else
  LOG(FATAL) << "Ascend support is not compiled. Please compile with "
                "-DUSE_ASCEND=ON";
#endif
}

// ---- Unsupported dtypes: half(uint16_t) / double placeholders ----

template <>
void ScatterAdd<kDGLAscend, int32_t, uint16_t>(
    NDArray feat, NDArray idx, NDArray out) {
  LOG(FATAL) << "Current Ascend scatter_add kernel only supports float "
                "features.";
}

template <>
void ScatterAdd<kDGLAscend, int64_t, uint16_t>(
    NDArray feat, NDArray idx, NDArray out) {
  LOG(FATAL) << "Current Ascend scatter_add kernel only supports float "
                "features.";
}

template <>
void ScatterAdd<kDGLAscend, int32_t, double>(
    NDArray feat, NDArray idx, NDArray out) {
  LOG(FATAL) << "Current Ascend scatter_add kernel only supports float "
                "features.";
}

template <>
void ScatterAdd<kDGLAscend, int64_t, double>(
    NDArray feat, NDArray idx, NDArray out) {
  LOG(FATAL) << "Current Ascend scatter_add kernel only supports float "
                "features.";
}

}  // namespace aten
}  // namespace dgl
