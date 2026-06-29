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

// int32 data kernels
extern "C" uint32_t aclrtlaunch_index_select_i32_i64(
    uint32_t blockDim, aclrtStream stream,
    void* src, void* idx, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_index_select_i32_i32(
    uint32_t blockDim, aclrtStream stream,
    void* src, void* idx, void* dst, void* tiling);

// int64 data kernels
extern "C" uint32_t aclrtlaunch_index_select_i64_i64(
    uint32_t blockDim, aclrtStream stream,
    void* src, void* idx, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_index_select_i64_i32(
    uint32_t blockDim, aclrtStream stream,
    void* src, void* idx, void* dst, void* tiling);

// float data kernels
extern "C" uint32_t aclrtlaunch_index_select_f32_i64(
    uint32_t blockDim, aclrtStream stream,
    void* src, void* idx, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_index_select_f32_i32(
    uint32_t blockDim, aclrtStream stream,
    void* src, void* idx, void* dst, void* tiling);

// double data kernels
extern "C" uint32_t aclrtlaunch_index_select_f64_i64(
    uint32_t blockDim, aclrtStream stream,
    void* src, void* idx, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_index_select_f64_i32(
    uint32_t blockDim, aclrtStream stream,
    void* src, void* idx, void* dst, void* tiling);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}
}
#endif

#include <dgl/array.h>
#include <dgl/runtime/device_api.h>

namespace dgl {
namespace aten {
namespace impl {

namespace {

template <typename DType, typename IdType>
struct IndexSelectKernelLauncher;

template <>
struct IndexSelectKernelLauncher<int32_t, int64_t> {
  static void Launch(uint32_t blockDim, aclrtStream stream,
                     void* src, void* idx, void* dst, void* tiling) {
    aclrtlaunch_index_select_i32_i64(blockDim, stream, src, idx, dst, tiling);
  }
};

template <>
struct IndexSelectKernelLauncher<int64_t, int64_t> {
  static void Launch(uint32_t blockDim, aclrtStream stream,
                     void* src, void* idx, void* dst, void* tiling) {
    aclrtlaunch_index_select_i64_i64(blockDim, stream, src, idx, dst, tiling);
  }
};

template <>
struct IndexSelectKernelLauncher<float, int64_t> {
  static void Launch(uint32_t blockDim, aclrtStream stream,
                     void* src, void* idx, void* dst, void* tiling) {
    aclrtlaunch_index_select_f32_i64(blockDim, stream, src, idx, dst, tiling);
  }
};

template <>
struct IndexSelectKernelLauncher<int32_t, int32_t> {
  static void Launch(uint32_t blockDim, aclrtStream stream,
                     void* src, void* idx, void* dst, void* tiling) {
    aclrtlaunch_index_select_i32_i32(blockDim, stream, src, idx, dst, tiling);
  }
};

template <>
struct IndexSelectKernelLauncher<int64_t, int32_t> {
  static void Launch(uint32_t blockDim, aclrtStream stream,
                     void* src, void* idx, void* dst, void* tiling) {
    aclrtlaunch_index_select_i64_i32(blockDim, stream, src, idx, dst, tiling);
  }
};

template <>
struct IndexSelectKernelLauncher<float, int32_t> {
  static void Launch(uint32_t blockDim, aclrtStream stream,
                     void* src, void* idx, void* dst, void* tiling) {
    aclrtlaunch_index_select_f32_i32(blockDim, stream, src, idx, dst, tiling);
  }
};

template <>
struct IndexSelectKernelLauncher<double, int64_t> {
  static void Launch(uint32_t blockDim, aclrtStream stream,
                     void* src, void* idx, void* dst, void* tiling) {
    aclrtlaunch_index_select_f64_i64(blockDim, stream, src, idx, dst, tiling);
  }
};

template <>
struct IndexSelectKernelLauncher<double, int32_t> {
  static void Launch(uint32_t blockDim, aclrtStream stream,
                     void* src, void* idx, void* dst, void* tiling) {
    aclrtlaunch_index_select_f64_i32(blockDim, stream, src, idx, dst, tiling);
  }
};

}  // anonymous namespace

template <DGLDeviceType XPU, typename DType, typename IdType>
NDArray IndexSelect(NDArray array, IdArray index) {
  CHECK_EQ(array->shape[0], array.NumElements())
      << "Only support tensor"
      << " whose first dimension equals number of elements, e.g. (5,), (5, 1)";

#ifdef DGL_USE_ASCEND
  auto ctx = array->ctx;
  CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));
  auto stream = dgl::runtime::getCurrentAscendStream();

  const int64_t len = index->shape[0];
  NDArray ret = NDArray::Empty({len}, array->dtype, ctx);
  if (len == 0) return ret;

  uint32_t n = static_cast<uint32_t>(len);
  uint32_t block_dim = 1;
  void* tiling_dev = nullptr;
  ASCEND_CALL(aclrtMalloc(&tiling_dev, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(tiling_dev, sizeof(uint32_t),
                           &n, sizeof(uint32_t),
                           ACL_MEMCPY_HOST_TO_DEVICE));

  IndexSelectKernelLauncher<DType, IdType>::Launch(
      block_dim, stream, array->data, index->data, ret->data, tiling_dev);

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  if (tiling_dev) ASCEND_CALL(aclrtFree(tiling_dev));

  return ret;
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}

template NDArray IndexSelect<kDGLAscend, int32_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLAscend, int32_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLAscend, int64_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLAscend, int64_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLAscend, float, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLAscend, float, int64_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLAscend, double, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLAscend, double, int64_t>(NDArray, IdArray);

template <DGLDeviceType XPU, typename DType>
DType IndexSelect(NDArray array, int64_t index) {
  CHECK(array->ctx.device_type == kDGLAscend) << "Expected Ascend device context";
  const DType* data = static_cast<DType*>(array->data);
  DType ret = 0;
  auto device = runtime::DeviceAPI::Get(array->ctx);
  device->CopyDataFromTo(
      data + index, 0, &ret, 0, sizeof(DType),
      array->ctx, DGLContext{kDGLCPU, 0}, array->dtype);
  return ret;
}

template int32_t IndexSelect<kDGLAscend, int32_t>(NDArray, int64_t);
template int64_t IndexSelect<kDGLAscend, int64_t>(NDArray, int64_t);
template float IndexSelect<kDGLAscend, float>(NDArray, int64_t);
template double IndexSelect<kDGLAscend, double>(NDArray, int64_t);

}  // namespace impl
}  // namespace aten
}  // namespace dgl

