/**
 *  Copyright (c) 2017-2022 by Contributors
 * @file cuda_device_api.cc
 * @brief GPU specific API
 */
#include <cuda_runtime.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/tensordispatch.h>
#include <dmlc/thread_local.h>

#include "cuda_common.h"

namespace dgl {
namespace runtime {

class CUDADeviceAPI final : public DeviceAPI {
 public:
  CUDADeviceAPI() {
    int count;
    auto err = cudaGetDeviceCount(&count);
    switch (err) {
      case cudaSuccess:
        break;
      default:
        count = 0;
        cudaGetLastError();
    }
    is_available_ = count > 0;
  }

  bool IsAvailable() final { return is_available_; }

  void SetDevice(DGLContext ctx) final {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
  }
  void GetAttr(DGLContext ctx, DeviceAttrKind kind, DGLRetValue* rv) final {
    int value = 0;
    switch (kind) {
      case kExist:
        value =
            (cudaDeviceGetAttribute(
                 &value, cudaDevAttrMaxThreadsPerBlock, ctx.device_id) ==
             cudaSuccess);
        break;
      case kMaxThreadsPerBlock: {
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrMaxThreadsPerBlock, ctx.device_id));
        break;
      }
      case kWarpSize: {
        CUDA_CALL(
            cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize, ctx.device_id));
        break;
      }
      case kMaxSharedMemoryPerBlock: {
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrMaxSharedMemoryPerBlock, ctx.device_id));
        break;
      }
      case kComputeVersion: {
        std::ostringstream os;
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrComputeCapabilityMajor, ctx.device_id));
        os << value << ".";
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrComputeCapabilityMinor, ctx.device_id));
        os << value;
        *rv = os.str();
        return;
      }
      case kDeviceName: {
        cudaDeviceProp props;
        CUDA_CALL(cudaGetDeviceProperties(&props, ctx.device_id));
        *rv = std::string(props.name);
        return;
      }
      case kMaxClockRate: {
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrClockRate, ctx.device_id));
        break;
      }
      case kMultiProcessorCount: {
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrMultiProcessorCount, ctx.device_id));
        break;
      }
      case kMaxThreadDimensions: {
        int dims[3];
        CUDA_CALL(cudaDeviceGetAttribute(
            &dims[0], cudaDevAttrMaxBlockDimX, ctx.device_id));
        CUDA_CALL(cudaDeviceGetAttribute(
            &dims[1], cudaDevAttrMaxBlockDimY, ctx.device_id));
        CUDA_CALL(cudaDeviceGetAttribute(
            &dims[2], cudaDevAttrMaxBlockDimZ, ctx.device_id));

        std::stringstream ss;  // use json string to return multiple int values;
        ss << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
        *rv = ss.str();
        return;
      }
    }
    *rv = value;
  }
  void* AllocDataSpace(
      DGLContext ctx, size_t nbytes, size_t alignment,
      DGLDataType type_hint) final {
    SetDevice(ctx);
    // Redirect to PyTorch's allocator when available.
    TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
    if (tensor_dispatcher->IsAvailable()) {
      return tensor_dispatcher->CUDAAllocWorkspace(
          nbytes, getCurrentCUDAStream());
    }
    CHECK_EQ(256 % alignment, 0U) << "CUDA space is aligned at 256 bytes";
    void* ret;
    CUDA_CALL(cudaMalloc(&ret, nbytes));
    return ret;
  }

  void FreeDataSpace(DGLContext ctx, void* ptr) final {
    SetDevice(ctx);
    TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
    if (tensor_dispatcher->IsAvailable()) {
      return tensor_dispatcher->CUDAFreeWorkspace(ptr);
    }
    CUDA_CALL(cudaFree(ptr));
  }

  void CopyDataFromTo(
      const void* from, size_t from_offset, void* to, size_t to_offset,
      size_t size, DGLContext ctx_from, DGLContext ctx_to,
      DGLDataType type_hint, DGLStreamHandle stream) {
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;
    if (ctx_from.device_type == kDGLCUDA && ctx_to.device_type == kDGLCUDA) {
      CUDA_CALL(cudaSetDevice(ctx_from.device_id));
      if (ctx_from.device_id == ctx_to.device_id) {
        GPUCopy(from, to, size, cudaMemcpyDeviceToDevice, cu_stream);
      } else {
        CUDA_CALL(cudaMemcpyPeerAsync(
            to, ctx_to.device_id, from, ctx_from.device_id, size, cu_stream));
      }
    } else if (
        ctx_from.device_type == kDGLCUDA && ctx_to.device_type == kDGLCPU) {
      CUDA_CALL(cudaSetDevice(ctx_from.device_id));
      GPUCopy(from, to, size, cudaMemcpyDeviceToHost, cu_stream);
    } else if (
        ctx_from.device_type == kDGLCPU && ctx_to.device_type == kDGLCUDA) {
      CUDA_CALL(cudaSetDevice(ctx_to.device_id));
      GPUCopy(from, to, size, cudaMemcpyHostToDevice, cu_stream);
    } else {
      LOG(FATAL) << "expect copy from/to GPU or between GPU";
    }
  }

  void CopyDataFromTo(
      const void* from, size_t from_offset, void* to, size_t to_offset,
      size_t size, DGLContext ctx_from, DGLContext ctx_to,
      DGLDataType type_hint) final {
    auto stream = GetStream();
    CopyDataFromTo(
        from, from_offset, to, to_offset, size, ctx_from, ctx_to, type_hint,
        stream);
  }

  // To ensure correct behavior, `record_event` must be invoked anytime a
  // pointer from PyTorch CachingHostAllocator is used in a cudaMemcpyAsync
  // call. It provides a way to re-use freed pinned (page-locked) memory
  // allocations and avoid device sync due to cudaFreeHost calls.
  void RecordedCopyDataFromTo(
      void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
      DGLContext ctx_from, DGLContext ctx_to, DGLDataType type_hint,
      void* pytorch_ctx) final {
    auto stream = GetStream();
    CopyDataFromTo(
        from, from_offset, to, to_offset, size, ctx_from, ctx_to, type_hint,
        stream);
    auto tensor_dispatcher = TensorDispatcher::Global();
    if (tensor_dispatcher->IsAvailable()) {
      auto custream = static_cast<cudaStream_t>(stream);
      void* ptr = ctx_to.device_type == kDGLCPU ? to : from;
      int id =
          ctx_to.device_type == kDGLCPU ? ctx_from.device_id : ctx_to.device_id;
      tensor_dispatcher->CUDARecordHostAlloc(ptr, pytorch_ctx, custream, id);
    }
  }

  DGLStreamHandle CreateStream(DGLContext ctx) {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    cudaStream_t retval;
    // make sure the legacy default stream won't block on this stream
    CUDA_CALL(cudaStreamCreateWithFlags(&retval, cudaStreamNonBlocking));
    return static_cast<DGLStreamHandle>(retval);
  }

  void FreeStream(DGLContext ctx, DGLStreamHandle stream) {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    CUDA_CALL(cudaStreamDestroy(cu_stream));
  }

  void SyncStreamFromTo(
      DGLContext ctx, DGLStreamHandle event_src, DGLStreamHandle event_dst) {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    cudaStream_t src_stream = static_cast<cudaStream_t>(event_src);
    cudaStream_t dst_stream = static_cast<cudaStream_t>(event_dst);
    cudaEvent_t evt;
    CUDA_CALL(cudaEventCreate(&evt));
    CUDA_CALL(cudaEventRecord(evt, src_stream));
    CUDA_CALL(cudaStreamWaitEvent(dst_stream, evt, 0));
    CUDA_CALL(cudaEventDestroy(evt));
  }

  void StreamSync(DGLContext ctx, DGLStreamHandle stream) final {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  }

  /** NOTE: If the backend is PyTorch, we will use PyTorch's stream management,
   *        so just avoid calling our SetStream/CreateStream unless
   *        you really need advanced stream control.
   * TODO(Xin): Redirect this to PyTorch or remove it.
   * PyTorch allows external CUDA streams to be set as current since v1.11.
   */
  void SetStream(DGLContext ctx, DGLStreamHandle stream) final {}

  DGLStreamHandle GetStream() const final {
    return static_cast<DGLStreamHandle>(getCurrentCUDAStream());
  }

  /** NOTE: cudaHostRegister can be called from an arbitrary GPU device,
   *        so we don't need to specify a ctx.
   *        The pinned memory can be seen by all CUDA contexts,
   *        not just the one that performed the allocation
   */
  bool PinData(void* ptr, size_t nbytes) override {
    // prevent users from pinning empty tensors or graphs
    if (ptr == nullptr || nbytes == 0) return false;
    TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
    // Minimize the pinned memory pool allocated by backend (via tensoradapter)
    // to preserve enough memory for DGL inherited in-place pin-memory operation
    if (tensor_dispatcher->IsAvailable()) {
      tensor_dispatcher->CUDAHostAllocatorEmptyCache();
    }
    CUDA_CALL(cudaHostRegister(ptr, nbytes, cudaHostRegisterDefault));
    return true;
  }

  void UnpinData(void* ptr) {
    if (ptr == nullptr) return;
    CUDA_CALL(cudaHostUnregister(ptr));
  }

  void* AllocPinnedDataSpace(
      size_t nbytes, void** ctx, void** deleter) override {
    // prevent pinning empty tensors or graphs
    if (nbytes == 0) return nullptr;
    TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
    CHECK(tensor_dispatcher->IsAvailable())
        << "CachingHostAllocator is not available in the current backend "
           "PyTorch. Please update the PyTorch version to 1.11+";
    return tensor_dispatcher->CUDAAllocHostWorkspace(nbytes, ctx, deleter);
  }

  void FreePinnedDataSpace(void** deleter) override {
    TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
    CHECK(tensor_dispatcher->IsAvailable())
        << "CachingHostAllocator is not available in the current backend "
           "PyTorch. Please update the PyTorch version to 1.11+";
    tensor_dispatcher->CUDAFreeHostWorkspace(deleter);
  }

  bool IsPinned(const void* ptr) override {
    // can't be a pinned tensor if CUDA context is unavailable.
    if (!is_available_) return false;

    cudaPointerAttributes attr;
    cudaError_t status = cudaPointerGetAttributes(&attr, ptr);
    bool result = false;

    switch (status) {
      case cudaErrorInvalidValue:
        // might be a normal CPU tensor in CUDA 10.2-
        cudaGetLastError();  // clear error
        break;
      case cudaSuccess:
        result = (attr.type == cudaMemoryTypeHost);
        break;
      case cudaErrorInitializationError:
      case cudaErrorNoDevice:
      case cudaErrorInsufficientDriver:
      case cudaErrorInvalidDevice:
        // We don't want to fail in these particular cases since this function
        // can be called when users only want to run on CPU even if CUDA API is
        // enabled, or in a forked subprocess where CUDA context cannot be
        // initialized.  So we just mark the CUDA context to unavailable and
        // return.
        is_available_ = false;
        cudaGetLastError();  // clear error
        break;
      default:
        LOG(FATAL) << "error while determining memory status: "
                   << cudaGetErrorString(status);
        break;
    }

    return result;
  }

  void* AllocWorkspace(
      DGLContext ctx, size_t size, DGLDataType type_hint) final {
    SetDevice(ctx);
    // Redirect to PyTorch's allocator when available.
    TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
    if (tensor_dispatcher->IsAvailable())
      return tensor_dispatcher->CUDAAllocWorkspace(
          size, getCurrentCUDAStream());

    return CUDAThreadEntry::ThreadLocal()->pool.AllocWorkspace(ctx, size);
  }

  void FreeWorkspace(DGLContext ctx, void* data) final {
    SetDevice(ctx);
    TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
    if (tensor_dispatcher->IsAvailable())
      return tensor_dispatcher->CUDAFreeWorkspace(data);

    CUDAThreadEntry::ThreadLocal()->pool.FreeWorkspace(ctx, data);
  }

  static const std::shared_ptr<CUDADeviceAPI>& Global() {
    static std::shared_ptr<CUDADeviceAPI> inst =
        std::make_shared<CUDADeviceAPI>();
    return inst;
  }

 private:
  static void GPUCopy(
      const void* from, void* to, size_t size, cudaMemcpyKind kind,
      cudaStream_t stream) {
    CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, stream));
    if (stream == 0 && kind == cudaMemcpyDeviceToHost) {
      // only wait for the copy, when it's on the default stream, and it's to
      // host memory
      CUDA_CALL(cudaStreamSynchronize(stream));
    }
  }

  bool is_available_ = true;
};

typedef dmlc::ThreadLocalStore<CUDAThreadEntry> CUDAThreadStore;

CUDAThreadEntry::CUDAThreadEntry() : pool(kDGLCUDA, CUDADeviceAPI::Global()) {}

CUDAThreadEntry* CUDAThreadEntry::ThreadLocal() {
  return CUDAThreadStore::Get();
}

cudaStream_t getCurrentCUDAStream() {
  TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
  if (tensor_dispatcher->IsAvailable())
    return tensor_dispatcher->CUDAGetCurrentStream();
  else  // return the default stream when TA is not available
    return nullptr;
}

DGL_REGISTER_GLOBAL("device_api.cuda")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      DeviceAPI* ptr = CUDADeviceAPI::Global().get();
      *rv = static_cast<void*>(ptr);
    });

}  // namespace runtime
}  // namespace dgl
