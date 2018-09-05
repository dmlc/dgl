/*!
 *  Copyright (c) 2016 by Contributors
 * \file cpu_device_api.cc
 */
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/device_api.h>
#include <cstdlib>
#include <cstring>
#include "workspace_pool.h"

namespace tvm {
namespace runtime {
class CPUDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(TVMContext ctx) final {}
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }
  void* AllocDataSpace(TVMContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       TVMType type_hint) final {
    void* ptr;
#if _MSC_VER
    ptr = _aligned_malloc(nbytes, alignment);
    if (ptr == nullptr) throw std::bad_alloc();
#elif defined(_LIBCPP_SGX_CONFIG)
    ptr = memalign(alignment, nbytes);
    if (ptr == nullptr) throw std::bad_alloc();
#else
    int ret = posix_memalign(&ptr, alignment, nbytes);
    if (ret != 0) throw std::bad_alloc();
#endif
    return ptr;
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
#if _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMType type_hint,
                      TVMStreamHandle stream) final {
    memcpy(static_cast<char*>(to) + to_offset,
           static_cast<const char*>(from) + from_offset,
           size);
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
  }

  void* AllocWorkspace(TVMContext ctx, size_t size, TVMType type_hint) final;
  void FreeWorkspace(TVMContext ctx, void* data) final;

  static const std::shared_ptr<CPUDeviceAPI>& Global() {
    static std::shared_ptr<CPUDeviceAPI> inst =
        std::make_shared<CPUDeviceAPI>();
    return inst;
  }
};

struct CPUWorkspacePool : public WorkspacePool {
  CPUWorkspacePool() :
      WorkspacePool(kDLCPU, CPUDeviceAPI::Global()) {}
};

void* CPUDeviceAPI::AllocWorkspace(TVMContext ctx,
                                   size_t size,
                                   TVMType type_hint) {
  return dmlc::ThreadLocalStore<CPUWorkspacePool>::Get()
      ->AllocWorkspace(ctx, size);
}

void CPUDeviceAPI::FreeWorkspace(TVMContext ctx, void* data) {
  dmlc::ThreadLocalStore<CPUWorkspacePool>::Get()->FreeWorkspace(ctx, data);
}

TVM_REGISTER_GLOBAL("device_api.cpu")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = CPUDeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
  });
}  // namespace runtime
}  // namespace tvm
