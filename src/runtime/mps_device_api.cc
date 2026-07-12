/**
 *  MPS Device API
 */
#include <dgl/runtime/device_api.h>
#include <cstring>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/tensordispatch.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>

#include "workspace_pool.h"

namespace dgl {
namespace runtime {
class MPSDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(DGLContext ctx) final {}
  void GetAttr(DGLContext ctx, DeviceAttrKind kind, DGLRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }
  void* AllocDataSpace(
      DGLContext ctx, size_t nbytes, size_t alignment,
      DGLDataType type_hint) final {
    TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
    if (tensor_dispatcher->IsAvailable()) {
      return tensor_dispatcher->MPSAllocWorkspace(nbytes);
    }
    LOG(FATAL) << "TensorDispatcher is not available for MPSAllocWorkspace.";
    return nullptr;
  }

  void FreeDataSpace(DGLContext ctx, void* ptr) final {
    TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
    if (tensor_dispatcher->IsAvailable()) {
      tensor_dispatcher->MPSFreeWorkspace(ptr);
      return;
    }
    LOG(FATAL) << "TensorDispatcher is not available for MPSFreeWorkspace.";
  }

  void CopyDataFromTo(
      const void* from, size_t from_offset, void* to, size_t to_offset,
      size_t size, DGLContext ctx_from, DGLContext ctx_to,
      DGLDataType type_hint) final {
    std::memcpy(
        static_cast<char*>(to) + to_offset,
        static_cast<const char*>(from) + from_offset,
        size);
  }

  void RecordedCopyDataFromTo(
      void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
      DGLContext ctx_from, DGLContext ctx_to, DGLDataType type_hint,
      void* pytorch_ctx) final {
    LOG(FATAL) << "This piece of code should not be reached.";
  }

  DGLStreamHandle CreateStream(DGLContext) final { return nullptr; }

  void StreamSync(DGLContext ctx, DGLStreamHandle stream) final {}

  void* AllocWorkspace(
      DGLContext ctx, size_t size, DGLDataType type_hint) final {
    return AllocDataSpace(ctx, size, 256, type_hint);
  }
  void FreeWorkspace(DGLContext ctx, void* data) final {
    FreeDataSpace(ctx, data);
  }

  static const std::shared_ptr<MPSDeviceAPI>& Global() {
    static std::shared_ptr<MPSDeviceAPI> inst =
        std::make_shared<MPSDeviceAPI>();
    return inst;
  }
};

DGL_REGISTER_GLOBAL("device_api.metal")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      DeviceAPI* ptr = MPSDeviceAPI::Global().get();
      *rv = static_cast<void*>(ptr);
    });
}  // namespace runtime
}  // namespace dgl
