/*
 * @file src/array/ascend/backward_segment_cmp.cc
 * @brief Host launcher + template specializations for BackwardSegmentCmp
 *        on Ascend NPU (scatter-assign for segment min/max backward).
 *
 * Mirrors scatter_add.cc / segment_reduce.cc:
 *   - aclrtlaunch_*: tiling malloc'd on GM + aclrtMemcpy H2D + launch +
 *     aclrtSynchronizeStream + aclrtFree.
 *   - int64 arg -> int32 conversion via CopyTo(CPU) + AsNumBits(32) +
 *     CopyTo(NPU).
 *   - dtype coverage: int32/int64 x float32 supported; half/double
 *     LOG(FATAL) placeholders.
 */
#ifdef DGL_USE_PYTORCH_NPU_STREAM
// Include torch_npu stream headers FIRST: c10 redefines LOG/CHECK macros;
// dmlc's definitions below must win so LOG(FATAL) throws (catchable) instead
// of aborting the process.
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#endif

#include <dgl/array.h>
#include "../kernel_decl.h"

#include <algorithm>
#include <cstdint>

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
#define ASCEND_CALL(func)                                    \
  {                                                          \
    aclError e = (func);                                     \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e; \
  }

struct BwdCmpTilingData {
  uint32_t N;
  uint32_t M;
  uint32_t featDim;
};

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_kernel_backward_segment_cmp(
    uint32_t blockDim, aclrtStream stream, void* feat, void* arg, void* out,
    void* tiling_ptr);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}  // namespace runtime
}  // namespace dgl

namespace {

// Resolve the stream to launch on. torch_npu executes eager ops on its own
// NPU stream, so DGL must use the same stream for cross-framework ordering
// (review: stream-granularity sync instead of aclrtSynchronizeDevice).
#ifdef DGL_USE_PYTORCH_NPU_STREAM
inline aclrtStream ResolveStream(int64_t device_id) {
  return c10_npu::getCurrentNPUStream(
             static_cast<c10::DeviceIndex>(device_id)).stream();
}
// torch_npu runs on its own ACL context; aclrtSynchronizeStream from DGL's
// default context does not reliably wait on torch streams, so use torch's
// own wrapper (stream-granularity, review: not device-wide sync).
inline void SyncStream(int64_t device_id) {
  c10_npu::stream_synchronize(ResolveStream(device_id));
}
#else
inline aclrtStream ResolveStream(int64_t) {
  return dgl::runtime::getCurrentAscendStream();
}
inline void SyncStream(int64_t) {
  aclrtStream s = dgl::runtime::getCurrentAscendStream();
  aclError e = aclrtSynchronizeStream(s);
  CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e;
}
#endif

// int64 -> int32 index narrowing.
//
// Two constraints make the naive routes unusable here:
//   - the framework's blocking D2H memcpy (NDArray::CopyTo) reads stale
//     bytes from torch-written buffers across torch_npu's ACL context;
//   - the AsNumBits AscendC kernel only converts the first
//     n / GetBlockNum() elements (latent bug, block_dim=1 -> first 16).
// Stream-ordered async copies on the torch stream are verified reliable,
// so narrow on host between an ordered D2H and H2D.
#ifdef DGL_USE_PYTORCH_NPU_STREAM
inline NDArray ConvertIdx64To32(const NDArray& src, int64_t device_id) {
  int64_t n = src.NumElements();
  NDArray ret = dgl::aten::NewIdArray(n, src->ctx, 32);
  if (n == 0) return ret;
  size_t bytes64 = static_cast<size_t>(n) * sizeof(int64_t);
  size_t bytes32 = static_cast<size_t>(n) * sizeof(int32_t);
  std::vector<int64_t> host64(static_cast<size_t>(n));
  std::vector<int32_t> host32(static_cast<size_t>(n));
  aclrtStream ts = ResolveStream(device_id);
  ASCEND_CALL(aclrtMemcpyAsync(host64.data(), bytes64, src->data, bytes64,
                               ACL_MEMCPY_DEVICE_TO_HOST, ts));
  c10_npu::stream_synchronize(ts);
  for (int64_t i = 0; i < n; ++i) {
    host32[static_cast<size_t>(i)] =
        static_cast<int32_t>(host64[static_cast<size_t>(i)]);
  }
  ASCEND_CALL(aclrtMemcpyAsync(ret->data, bytes32, host32.data(), bytes32,
                               ACL_MEMCPY_HOST_TO_DEVICE, ts));
  c10_npu::stream_synchronize(ts);
  return ret;
}
#else
inline NDArray ConvertIdx64To32(const NDArray& src, int64_t device_id) {
  ASCEND_CALL(aclrtSetDevice(device_id));
  NDArray cpu = src.CopyTo(DGLContext{kDGLCPU, 0});
  IdArray cpu_i32 = dgl::aten::AsNumBits(cpu, 32);
  return cpu_i32.CopyTo(src->ctx);
}
#endif

}  // namespace
#endif  // DGL_USE_ASCEND

namespace dgl {
namespace aten {

#ifdef DGL_USE_ASCEND
namespace {

inline uint32_t ComputeBlockDim(int64_t n) {
  return static_cast<uint32_t>(n > 0 ? std::min<int64_t>(n, 40) : 1);
}

template <typename DType>
void LaunchBackwardSegmentCmpKernel(
    int64_t n, int64_t m, int64_t feat_dim, const void* feat_ptr,
    const void* arg_ptr, void* out_ptr, aclrtStream stream) {
  uint32_t block_dim = ComputeBlockDim(n);

  BwdCmpTilingData tiling = {
      static_cast<uint32_t>(n), static_cast<uint32_t>(m),
      static_cast<uint32_t>(feat_dim)};

  void* tiling_device = nullptr;
  ASCEND_CALL(aclrtMalloc(
      &tiling_device, sizeof(BwdCmpTilingData),
      ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(
      tiling_device, sizeof(BwdCmpTilingData), &tiling,
      sizeof(BwdCmpTilingData), ACL_MEMCPY_HOST_TO_DEVICE));

  aclError launch_err = aclrtlaunch_kernel_backward_segment_cmp(
      block_dim, stream, const_cast<void*>(feat_ptr),
      const_cast<void*>(arg_ptr), out_ptr, tiling_device);
  if (launch_err != ACL_SUCCESS) {
    ASCEND_CALL(aclrtFree(tiling_device));
    LOG(FATAL) << "BackwardSegmentCmp kernel launch failed with error code: "
               << launch_err;
  }

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(tiling_device));
}

template <typename DType>
void BackwardSegmentCmpAscendImpl(NDArray feat, NDArray arg, NDArray out) {
  DGLContext ctx = feat->ctx;
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  // Stream-granularity sync (review: aclrtSynchronizeDevice waits on all
  // streams device-wide, blocking far more than needed).
  // Launch on DGL's default stream: our kernels are mutually ordered there.
  // torch-boundary waits (H2D visibility) use the torch stream (SyncStream).
  aclrtStream stream = nullptr;
  SyncStream(ctx.device_id);

  int64_t n = feat->shape[0];
  int64_t m = out->shape[0];
  int64_t dim = 1;
  for (int i = 1; i < out->ndim; ++i) dim *= out->shape[i];

  LaunchBackwardSegmentCmpKernel<DType>(
      n, m, dim, feat->data, arg->data, out->data, stream);
}

}  // namespace
#endif  // DGL_USE_ASCEND

// ---- Supported: int32/int64 x float32 ----

template <>
void BackwardSegmentCmp<kDGLAscend, int32_t, float>(
    NDArray feat, NDArray arg, NDArray out) {
#ifdef DGL_USE_ASCEND
  BackwardSegmentCmpAscendImpl<float>(feat, arg, out);
#else
  LOG(FATAL) << "Ascend support is not compiled. Please compile with "
                 "-DUSE_ASCEND=ON";
#endif
}

template <>
void BackwardSegmentCmp<kDGLAscend, int64_t, float>(
    NDArray feat, NDArray arg, NDArray out) {
#ifdef DGL_USE_ASCEND
  // int64 arg -> int32 conversion, performed on-device via the AsNumBits
  // Ascend kernel (no host round-trip: the blocking D2H memcpy of
  // torch-written buffers is unreliable under torch_npu's own ACL context /
  // copy streams). Ordering is guaranteed because the AsNumBits kernel and
  // the consumer kernel both run on the same resolved stream.
  ASCEND_CALL(aclrtSetDevice(arg->ctx.device_id));
  SyncStream(arg->ctx.device_id);
  NDArray arg_i32 = ConvertIdx64To32(arg, arg->ctx.device_id);
  BackwardSegmentCmpAscendImpl<float>(feat, arg_i32, out);
#else
  LOG(FATAL) << "Ascend support is not compiled. Please compile with "
                 "-DUSE_ASCEND=ON";
#endif
}

// ---- Unsupported dtypes ----

template <>
void BackwardSegmentCmp<kDGLAscend, int32_t, uint16_t>(
    NDArray feat, NDArray arg, NDArray out) {
  LOG(FATAL)
      << "Current Ascend BackwardSegmentCmp kernel only supports float features.";
}

template <>
void BackwardSegmentCmp<kDGLAscend, int64_t, uint16_t>(
    NDArray feat, NDArray arg, NDArray out) {
  LOG(FATAL)
      << "Current Ascend BackwardSegmentCmp kernel only supports float features.";
}

template <>
void BackwardSegmentCmp<kDGLAscend, int32_t, double>(
    NDArray feat, NDArray arg, NDArray out) {
  LOG(FATAL)
      << "Current Ascend BackwardSegmentCmp kernel only supports float features.";
}

template <>
void BackwardSegmentCmp<kDGLAscend, int64_t, double>(
    NDArray feat, NDArray arg, NDArray out) {
  LOG(FATAL)
      << "Current Ascend BackwardSegmentCmp kernel only supports float features.";
}

}  // namespace aten
}  // namespace dgl
