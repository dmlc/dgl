/**
 *  Copyright (c) 2024 by Contributors
 * @file ascend_device_api.cc
 * @brief Huawei Ascend NPU specific API
 */
#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
#endif

#include <dgl/runtime/device_api.h>
#include <dgl/runtime/registry.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>

#include <cstring>

#include "../workspace_pool.h"

namespace dgl {
namespace runtime {

#ifdef DGL_USE_ASCEND
#define ASCEND_CALL(func)                                                \
  {                                                                      \
    aclError e = (func);                                                 \
    CHECK(e == ACL_SUCCESS) << "Ascend Error: " << aclGetRecentErrMsg(); \
  }
#endif

// Forward declaration
class AscendDeviceAPI;

#ifdef DGL_USE_ASCEND
/** @brief Thread local workspace for Ascend */
class AscendThreadEntry {
 public:
  /** @brief thread local pool*/
  WorkspacePool pool;
  /** @brief constructor */
  AscendThreadEntry();
  // get the threadlocal workspace
  static AscendThreadEntry* ThreadLocal() {
    static thread_local AscendThreadEntry entry;
    return &entry;
  }
};
#endif

class AscendDeviceAPI final : public DeviceAPI {
 public:
  AscendDeviceAPI() {
#ifdef DGL_USE_ASCEND
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
      is_available_ = false;
      return;
    }
    uint32_t count = 0;
    ret = aclrtGetDeviceCount(&count);
    is_available_ = (ret == ACL_SUCCESS && count > 0);
#else
    is_available_ = false;
#endif
  }

  ~AscendDeviceAPI() {
#ifdef DGL_USE_ASCEND
    // Don't call aclFinalize here as it may cause issues
    // if other parts of the program are still using ACL
#endif
  }

  bool IsAvailable() final { return is_available_; }

  void SetDevice(DGLContext ctx) final {
#ifdef DGL_USE_ASCEND
    ASCEND_CALL(aclrtSetDevice(ctx.device_id));
#else
    LOG(FATAL) << "Ascend runtime is not enabled. Please compile with -DUSE_ASCEND=ON";
#endif
  }

  void GetAttr(DGLContext ctx, DeviceAttrKind kind, DGLRetValue* rv) final {
#ifdef DGL_USE_ASCEND
    switch (kind) {
      case kExist: {
        uint32_t count = 0;
        aclError ret = aclrtGetDeviceCount(&count);
        *rv = (ret == ACL_SUCCESS && static_cast<uint32_t>(ctx.device_id) < count) ? 1 : 0;
        break;
      }
      case kDeviceName: {
        const char* name = aclrtGetSocName();
        *rv = std::string(name ? name : "Ascend NPU");
        break;
      }
      case kMaxThreadsPerBlock:
      case kWarpSize:
      case kMaxSharedMemoryPerBlock:
      case kComputeVersion:
      case kMaxClockRate:
      case kMultiProcessorCount:
      case kMaxThreadDimensions:
        // Ascend doesn't have direct equivalent concepts
        *rv = 0;
        break;
    }
#else
    if (kind == kExist) {
      *rv = 0;
    } else {
      LOG(FATAL) << "Ascend runtime is not enabled.";
    }
#endif
  }

  void* AllocDataSpace(
      DGLContext ctx, size_t nbytes, size_t alignment,
      DGLDataType type_hint) final {
#ifdef DGL_USE_ASCEND
    SetDevice(ctx);
    // Ascend 910B3: avoid stale L2 cache line by offsetting data ptr +64.
    void* base = nullptr;
    ASCEND_CALL(aclrtMalloc(&base, nbytes + 64, ACL_MEM_MALLOC_NORMAL_ONLY));
    // Only zero the first 64 bytes (sacrificial cache line), not the whole buffer.
    ASCEND_CALL(aclrtMemset(base, 64, 0, 64));
    void* ret = static_cast<char*>(base) + 64;
    return ret;
#else
    LOG(FATAL) << "Ascend runtime is not enabled.";
    return nullptr;
#endif
  }

  void FreeDataSpace(DGLContext ctx, void* ptr) final {
#ifdef DGL_USE_ASCEND
    SetDevice(ctx);
    // AllocDataSpace returns ptr+64; recover the base pointer for free.
    ASCEND_CALL(aclrtFree(static_cast<char*>(ptr) - 64));
#else
    LOG(FATAL) << "Ascend runtime is not enabled.";
#endif
  }

  void CopyDataFromTo(
      const void* from, size_t from_offset, void* to, size_t to_offset,
      size_t size, DGLContext ctx_from, DGLContext ctx_to,
      DGLDataType type_hint) final {
#ifdef DGL_USE_ASCEND
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;

    if (size == 0) return;

    if (ctx_from.device_type == kDGLAscend && ctx_to.device_type == kDGLAscend) {
      ASCEND_CALL(aclrtSetDevice(ctx_from.device_id));
      if (ctx_from.device_id == ctx_to.device_id) {
        ASCEND_CALL(aclrtMemcpy(
            to, size, from, size, ACL_MEMCPY_DEVICE_TO_DEVICE));
      } else {
        void* temp = malloc(size);
        ASCEND_CALL(aclrtMemcpy(temp, size, from, size, ACL_MEMCPY_DEVICE_TO_HOST));
        ASCEND_CALL(aclrtSetDevice(ctx_to.device_id));
        ASCEND_CALL(aclrtMemcpy(to, size, temp, size, ACL_MEMCPY_HOST_TO_DEVICE));
        free(temp);
      }
    } else if (ctx_from.device_type == kDGLAscend && ctx_to.device_type == kDGLCPU) {
      // Device to Host
      ASCEND_CALL(aclrtSetDevice(ctx_from.device_id));
      ASCEND_CALL(aclrtSynchronizeStream(static_cast<aclrtStream>(current_stream_)));
      ASCEND_CALL(aclrtMemcpy(
          to, size, from, size, ACL_MEMCPY_DEVICE_TO_HOST));
    } else if (ctx_from.device_type == kDGLCPU && ctx_to.device_type == kDGLAscend) {
      // Host to Device
      ASCEND_CALL(aclrtSetDevice(ctx_to.device_id));
      ASCEND_CALL(aclrtMemcpy(
          to, size, from, size, ACL_MEMCPY_HOST_TO_DEVICE));
    } else {
      LOG(FATAL) << "Expect copy from/to Ascend or between Ascend devices";
    }
#else
    LOG(FATAL) << "Ascend runtime is not enabled.";
#endif
  }

  void RecordedCopyDataFromTo(
      void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
      DGLContext ctx_from, DGLContext ctx_to, DGLDataType type_hint,
      void* pytorch_ctx) final {
    // For now, just do a regular copy
    CopyDataFromTo(from, from_offset, to, to_offset, size, ctx_from, ctx_to,
                   type_hint);
  }

  DGLStreamHandle CreateStream(DGLContext ctx) final {
#ifdef DGL_USE_ASCEND
    ASCEND_CALL(aclrtSetDevice(ctx.device_id));
    aclrtStream stream;
    ASCEND_CALL(aclrtCreateStream(&stream));
    return static_cast<DGLStreamHandle>(stream);
#else
    LOG(FATAL) << "Ascend runtime is not enabled.";
    return nullptr;
#endif
  }

  void FreeStream(DGLContext ctx, DGLStreamHandle stream) final {
#ifdef DGL_USE_ASCEND
    ASCEND_CALL(aclrtSetDevice(ctx.device_id));
    ASCEND_CALL(aclrtDestroyStream(static_cast<aclrtStream>(stream)));
#else
    LOG(FATAL) << "Ascend runtime is not enabled.";
#endif
  }

  void StreamSync(DGLContext ctx, DGLStreamHandle stream) final {
#ifdef DGL_USE_ASCEND
    ASCEND_CALL(aclrtSetDevice(ctx.device_id));
    ASCEND_CALL(aclrtSynchronizeStream(static_cast<aclrtStream>(stream)));
#else
    LOG(FATAL) << "Ascend runtime is not enabled.";
#endif
  }

  void SetStream(DGLContext ctx, DGLStreamHandle stream) final {
#ifdef DGL_USE_ASCEND
    current_stream_ = static_cast<aclrtStream>(stream);
#endif
  }

  DGLStreamHandle GetStream() const final {
#ifdef DGL_USE_ASCEND
    return static_cast<DGLStreamHandle>(current_stream_);
#else
    return nullptr;
#endif
  }

  void* AllocWorkspace(
      DGLContext ctx, size_t size, DGLDataType type_hint) final {
#ifdef DGL_USE_ASCEND
    SetDevice(ctx);
    return AscendThreadEntry::ThreadLocal()->pool.AllocWorkspace(ctx, size);
#else
    LOG(FATAL) << "Ascend runtime is not enabled.";
    return nullptr;
#endif
  }

  void FreeWorkspace(DGLContext ctx, void* data) final {
#ifdef DGL_USE_ASCEND
    SetDevice(ctx);
    AscendThreadEntry::ThreadLocal()->pool.FreeWorkspace(ctx, data);
#else
    LOG(FATAL) << "Ascend runtime is not enabled.";
#endif
  }

  static const std::shared_ptr<AscendDeviceAPI>& Global() {
    static std::shared_ptr<AscendDeviceAPI> inst =
        std::make_shared<AscendDeviceAPI>();
    return inst;
  }

 private:
  bool is_available_ = false;
#ifdef DGL_USE_ASCEND
  aclrtStream current_stream_ = nullptr;
#endif
};

#ifdef DGL_USE_ASCEND
// Define the constructor after AscendDeviceAPI is fully defined
AscendThreadEntry::AscendThreadEntry()
    : pool(kDGLAscend, AscendDeviceAPI::Global()) {}
#endif

DGL_REGISTER_GLOBAL("device_api.ascend")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      DeviceAPI* ptr = AscendDeviceAPI::Global().get();
      *rv = static_cast<void*>(ptr);
    });

#ifdef DGL_USE_ASCEND
static thread_local aclrtStream current_ascend_stream = nullptr;

aclrtStream getCurrentAscendStream() {
  return current_ascend_stream;
}

void setCurrentAscendStream(aclrtStream stream) {
  current_ascend_stream = stream;
}
#endif

}  // namespace runtime
}  // namespace dgl

