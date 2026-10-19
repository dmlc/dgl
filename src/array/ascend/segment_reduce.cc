#include <dgl/array.h>
#include "../kernel_decl.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_op.h>
#include <acl/acl_rt.h>
#define ASCEND_CALL(func)                                    \
  {                                                          \
    aclError e = (func);                                     \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e; \
  }

struct SegmentReduceTilingData {
  uint32_t numItems;
  uint32_t numSegments;
  uint32_t featDim;
};

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_segment_reduce_sum(
    uint32_t blockDim, aclrtStream stream, void* offsets, void* feat,
    void* output, void* segment_split, void* tiling);
extern "C" uint32_t aclrtlaunch_segment_reduce_max(
    uint32_t blockDim, aclrtStream stream, void* offsets, void* feat,
    void* output, void* segment_split, void* tiling);
extern "C" uint32_t aclrtlaunch_segment_reduce_min(
    uint32_t blockDim, aclrtStream stream, void* offsets, void* feat,
    void* output, void* segment_split, void* tiling);
#endif

namespace dgl {
namespace aten {

#ifdef DGL_USE_ASCEND
namespace {

// 负载均衡
std::vector<uint32_t> BuildSegmentSplit(
    const uint32_t* offsets_ptr, int64_t num_segments, uint32_t block_dim,
    aclrtStream stream) {
  // Use synchronous copy to ensure data is ready
  std::vector<uint32_t> host_offsets(num_segments + 1, 0);
  ASCEND_CALL(aclrtMemcpy(
      host_offsets.data(), sizeof(uint32_t) * host_offsets.size(), offsets_ptr,
      sizeof(uint32_t) * host_offsets.size(), ACL_MEMCPY_DEVICE_TO_HOST));

  std::vector<uint32_t> segment_split(block_dim + 1, 0);
  segment_split[block_dim] = static_cast<uint32_t>(num_segments);
  uint32_t total_weight = host_offsets.back();
  for (uint32_t part = 1; part < block_dim; ++part) {
    uint32_t target_weight = static_cast<uint32_t>(
        (static_cast<uint64_t>(part) * total_weight) / block_dim);
    segment_split[part] = static_cast<uint32_t>(
        std::lower_bound(
            host_offsets.begin(), host_offsets.end(), target_weight) -
        host_offsets.begin());
  }
  return segment_split;
}

template <typename LaunchFn>
void LaunchSegmentReduceKernel(
    LaunchFn launch_fn, int device_id, int64_t num_items,
    int64_t num_segments, int64_t feat_dim, const uint32_t* offsets_ptr,
    const float* feat_ptr, float* out_ptr, aclrtStream stream) {
  int64_t coreNum = 0;
  aclError err = aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreNum);
  if (err != ACL_SUCCESS || coreNum <= 0) {
    coreNum = 0;
    err = aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_AICORE_CORE_NUM, &coreNum);
    if (err != ACL_SUCCESS || coreNum <= 0) coreNum = 6;
  }
  uint32_t block_dim = static_cast<uint32_t>(
      num_segments > 0 ? std::min<int64_t>(num_segments, coreNum) : 1);
  std::vector<uint32_t> segment_split =
      BuildSegmentSplit(offsets_ptr, num_segments, block_dim, stream);
  size_t segment_split_bytes = sizeof(uint32_t) * segment_split.size();
  void* segment_split_device = nullptr;
  ASCEND_CALL(aclrtMalloc(
      &segment_split_device, segment_split_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(
      segment_split_device, segment_split_bytes, segment_split.data(),
      segment_split_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

  SegmentReduceTilingData tiling = {
      static_cast<uint32_t>(num_items), static_cast<uint32_t>(num_segments),
      static_cast<uint32_t>(feat_dim)};

  void* tiling_device = nullptr;
  ASCEND_CALL(aclrtMalloc(
      &tiling_device, sizeof(SegmentReduceTilingData),
      ACL_MEM_MALLOC_HUGE_FIRST));

  ASCEND_CALL(aclrtMemcpy(
      tiling_device, sizeof(SegmentReduceTilingData), &tiling,
      sizeof(SegmentReduceTilingData), ACL_MEMCPY_HOST_TO_DEVICE));

  aclError launch_err = launch_fn(
      block_dim, stream, const_cast<void*>(static_cast<const void*>(offsets_ptr)),
      const_cast<void*>(static_cast<const void*>(feat_ptr)), out_ptr,
      segment_split_device, tiling_device);
  if (launch_err != ACL_SUCCESS) {
    ASCEND_CALL(aclrtFree(segment_split_device));
    ASCEND_CALL(aclrtFree(tiling_device));
    LOG(FATAL) << "SegmentReduce kernel launch failed with error code: "
               << launch_err;
  }

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(segment_split_device));
  ASCEND_CALL(aclrtFree(tiling_device));
}

}  // namespace
#endif

template <typename IdType, typename DType>
void SegmentReduceAscend(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg) {
#ifdef DGL_USE_ASCEND
  DGLContext ctx = feat->ctx;
  ASCEND_CALL(aclrtSynchronizeDevice());  // Ensure PyTorch NPU ops complete before reading offsets
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  if (op == "sum" || op == "max" || op == "min") {
    int64_t num_items = feat->shape[0];
    int64_t num_segments = out->shape[0];
    int64_t feat_dim = 1;
    for (int i = 1; i < feat->ndim; ++i) feat_dim *= feat->shape[i];

    // Use the default ACL stream (nullptr).  Note: this is NOT the same as
    // PyTorch NPU's default stream -- they are decoupled.  Correctness is
    // ensured by the aclrtSynchronizeDevice() call above, not by stream
    // alignment.  TODO: run DGL kernels on PyTorch's current NPU stream
    // (via to_dgl_stream_handle) to avoid device-level sync.
    aclrtStream stream = nullptr;

    const uint32_t* offsets_ptr =
        static_cast<const uint32_t*>(static_cast<const void*>(offsets->data));
    const float* feat_ptr =
        static_cast<const float*>(static_cast<const void*>(feat->data));
    float* out_ptr = static_cast<float*>(out->data);

    if (op == "sum") {
      LaunchSegmentReduceKernel(
          aclrtlaunch_segment_reduce_sum, ctx.device_id, num_items,
          num_segments, feat_dim, offsets_ptr, feat_ptr, out_ptr, stream);
    } else if (op == "max") {
      LaunchSegmentReduceKernel(
          aclrtlaunch_segment_reduce_max, ctx.device_id, num_items,
          num_segments, feat_dim, offsets_ptr, feat_ptr, out_ptr, stream);
    } else {
      LaunchSegmentReduceKernel(
          aclrtlaunch_segment_reduce_min, ctx.device_id, num_items,
          num_segments, feat_dim, offsets_ptr, feat_ptr, out_ptr, stream);
    }
  } else {
    LOG(FATAL) << "Unsupported reduce function " << op;
  }
#else
  LOG(FATAL)
      << "Ascend support is not compiled. Please compile with -DUSE_ASCEND=ON";
#endif
}

template <>
void SegmentReduce<kDGLAscend, int32_t, uint16_t>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg) {
  LOG(FATAL)
      << "Current Ascend segment_reduce kernel only supports float features.";
}

template <>
void SegmentReduce<kDGLAscend, int64_t, uint16_t>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg) {
  LOG(FATAL)
      << "Current Ascend segment_reduce kernel only supports float features.";
}

template <>
void SegmentReduce<kDGLAscend, int32_t, float>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg) {
  SegmentReduceAscend<int32_t, float>(op, feat, offsets, out, arg);
}

template <>
void SegmentReduce<kDGLAscend, int64_t, float>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg) {
  NDArray offsets_cpu = offsets.CopyTo(DGLContext{kDGLCPU, 0});
  IdArray offsets_i32_cpu = aten::AsNumBits(offsets_cpu, 32);
  NDArray offsets_i32 = offsets_i32_cpu.CopyTo(offsets->ctx);
  SegmentReduceAscend<int32_t, float>(op, feat, offsets_i32, out, arg);
}

template <>
void SegmentReduce<kDGLAscend, int32_t, double>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg) {
  LOG(FATAL)
      << "Current Ascend segment_reduce kernel only supports float features.";
}

template <>
void SegmentReduce<kDGLAscend, int64_t, double>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg) {
  LOG(FATAL) << "Current Ascend segment_reduce kernel only supports int32 "
                "offsets and float features.";
}

}  // namespace aten
}  // namespace dgl
