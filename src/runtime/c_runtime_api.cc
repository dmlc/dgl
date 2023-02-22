/**
 *  Copyright (c) 2016-2022 by Contributors
 * @file c_runtime_api.cc
 * @brief Runtime API implementation
 */
#include <dgl/runtime/c_backend_api.h>
#include <dgl/runtime/c_runtime_api.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/module.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/tensordispatch.h>
#include <dmlc/thread_local.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>

#include "runtime_base.h"

namespace dgl {
namespace runtime {

/**
 * @brief The name of Device API factory.
 * @param type The device type.
 */
inline std::string DeviceName(int type) {
  switch (type) {
    case kDGLCPU:
      return "cpu";
    case kDGLCUDA:
      return "cuda";
    // add more device here once supported
    default:
      LOG(FATAL) << "unknown type =" << type;
      return "Unknown";
  }
}

class DeviceAPIManager {
 public:
  static const int kMaxDeviceAPI = 32;
  // Get API
  static DeviceAPI* Get(const DGLContext& ctx) { return Get(ctx.device_type); }
  static DeviceAPI* Get(int dev_type, bool allow_missing = false) {
    return Global()->GetAPI(dev_type, allow_missing);
  }

 private:
  std::array<DeviceAPI*, kMaxDeviceAPI> api_;
  DeviceAPI* rpc_api_{nullptr};
  std::mutex mutex_;
  // constructor
  DeviceAPIManager() { std::fill(api_.begin(), api_.end(), nullptr); }
  // Global static variable.
  static DeviceAPIManager* Global() {
    static DeviceAPIManager inst;
    return &inst;
  }
  // Get or initialize API.
  DeviceAPI* GetAPI(int type, bool allow_missing) {
    if (type < kRPCSessMask) {
      if (api_[type] != nullptr) return api_[type];
      std::lock_guard<std::mutex> lock(mutex_);
      if (api_[type] != nullptr) return api_[type];
      api_[type] = GetAPI(DeviceName(type), allow_missing);
      return api_[type];
    } else {
      if (rpc_api_ != nullptr) return rpc_api_;
      std::lock_guard<std::mutex> lock(mutex_);
      if (rpc_api_ != nullptr) return rpc_api_;
      rpc_api_ = GetAPI("rpc", allow_missing);
      return rpc_api_;
    }
  }
  DeviceAPI* GetAPI(const std::string name, bool allow_missing) {
    std::string factory = "device_api." + name;
    auto* f = Registry::Get(factory);
    if (f == nullptr) {
      CHECK(allow_missing)
          << "Device API " << name
          << " is not enabled. Please install the cuda version of dgl.";
      return nullptr;
    }
    void* ptr = (*f)();
    return static_cast<DeviceAPI*>(ptr);
  }
};

DeviceAPI* DeviceAPI::Get(DGLContext ctx, bool allow_missing) {
  return DeviceAPIManager::Get(
      static_cast<int>(ctx.device_type), allow_missing);
}

DeviceAPI* DeviceAPI::Get(DGLDeviceType dev_type, bool allow_missing) {
  return DeviceAPIManager::Get(static_cast<int>(dev_type), allow_missing);
}

void* DeviceAPI::AllocWorkspace(
    DGLContext ctx, size_t size, DGLDataType type_hint) {
  return AllocDataSpace(ctx, size, kTempAllocaAlignment, type_hint);
}

void DeviceAPI::FreeWorkspace(DGLContext ctx, void* ptr) {
  FreeDataSpace(ctx, ptr);
}

DGLStreamHandle DeviceAPI::CreateStream(DGLContext ctx) {
  LOG(FATAL) << "Device does not support stream api.";
  return 0;
}

void DeviceAPI::FreeStream(DGLContext ctx, DGLStreamHandle stream) {
  LOG(FATAL) << "Device does not support stream api.";
}

void DeviceAPI::SyncStreamFromTo(
    DGLContext ctx, DGLStreamHandle event_src, DGLStreamHandle event_dst) {
  LOG(FATAL) << "Device does not support stream api.";
}

bool DeviceAPI::PinData(void* ptr, size_t nbytes) {
  LOG(FATAL) << "Device does not support cudaHostRegister api.";
  return false;
}

void* DeviceAPI::AllocPinnedDataSpace(
    size_t nbytes, void** ctx, void** deleter) {
  LOG(FATAL) << "Device does not support cudaHostAlloc api.";
  return nullptr;
}

void DeviceAPI::FreePinnedDataSpace(void** deleter) {
  LOG(FATAL) << "Device does not support cudaHostFree api.";
}

void DeviceAPI::UnpinData(void* ptr) {
  LOG(FATAL) << "Device does not support cudaHostUnregister api.";
}
}  // namespace runtime
}  // namespace dgl

using namespace dgl::runtime;

struct DGLRuntimeEntry {
  std::string ret_str;
  std::string last_error;
  DGLByteArray ret_bytes;
};

typedef dmlc::ThreadLocalStore<DGLRuntimeEntry> DGLAPIRuntimeStore;

const char* DGLGetLastError() {
  return DGLAPIRuntimeStore::Get()->last_error.c_str();
}

void DGLAPISetLastError(const char* msg) {
#ifndef _LIBCPP_SGX_CONFIG
  DGLAPIRuntimeStore::Get()->last_error = msg;
#else
  sgx::OCallPackedFunc("__sgx_set_last_error__", msg);
#endif
}

int DGLModLoadFromFile(
    const char* file_name, const char* format, DGLModuleHandle* out) {
  API_BEGIN();
  Module m = Module::LoadFromFile(file_name, format);
  *out = new Module(m);
  API_END();
}

int DGLModImport(DGLModuleHandle mod, DGLModuleHandle dep) {
  API_BEGIN();
  static_cast<Module*>(mod)->Import(*static_cast<Module*>(dep));
  API_END();
}

int DGLModGetFunction(
    DGLModuleHandle mod, const char* func_name, int query_imports,
    DGLFunctionHandle* func) {
  API_BEGIN();
  PackedFunc pf =
      static_cast<Module*>(mod)->GetFunction(func_name, query_imports != 0);
  if (pf != nullptr) {
    *func = new PackedFunc(pf);
  } else {
    *func = nullptr;
  }
  API_END();
}

int DGLModFree(DGLModuleHandle mod) {
  API_BEGIN();
  delete static_cast<Module*>(mod);
  API_END();
}

int DGLBackendGetFuncFromEnv(
    void* mod_node, const char* func_name, DGLFunctionHandle* func) {
  API_BEGIN();
  *func =
      (DGLFunctionHandle)(static_cast<ModuleNode*>(mod_node)->GetFuncFromEnv(
          func_name));
  API_END();
}

void* DGLBackendAllocWorkspace(
    int device_type, int device_id, uint64_t size, int dtype_code_hint,
    int dtype_bits_hint) {
  DGLContext ctx;
  ctx.device_type = static_cast<DGLDeviceType>(device_type);
  ctx.device_id = device_id;

  DGLDataType type_hint;
  type_hint.code = static_cast<decltype(type_hint.code)>(dtype_code_hint);
  type_hint.bits = static_cast<decltype(type_hint.bits)>(dtype_bits_hint);
  type_hint.lanes = 1;

  return DeviceAPIManager::Get(ctx)->AllocWorkspace(
      ctx, static_cast<size_t>(size), type_hint);
}

int DGLBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  DGLContext ctx;
  ctx.device_type = static_cast<DGLDeviceType>(device_type);
  ctx.device_id = device_id;
  DeviceAPIManager::Get(ctx)->FreeWorkspace(ctx, ptr);
  return 0;
}

int DGLBackendRunOnce(void** handle, int (*f)(void*), void* cdata, int nbytes) {
  if (*handle == nullptr) {
    *handle = reinterpret_cast<void*>(1);
    return (*f)(cdata);
  }
  return 0;
}

int DGLFuncFree(DGLFunctionHandle func) {
  API_BEGIN();
  delete static_cast<PackedFunc*>(func);
  API_END();
}

int DGLFuncCall(
    DGLFunctionHandle func, DGLValue* args, int* arg_type_codes, int num_args,
    DGLValue* ret_val, int* ret_type_code) {
  API_BEGIN();
  DGLRetValue rv;
  (*static_cast<const PackedFunc*>(func))
      .CallPacked(DGLArgs(args, arg_type_codes, num_args), &rv);
  // handle return string.
  if (rv.type_code() == kStr || rv.type_code() == kDGLDataType ||
      rv.type_code() == kBytes) {
    DGLRuntimeEntry* e = DGLAPIRuntimeStore::Get();
    if (rv.type_code() != kDGLDataType) {
      e->ret_str = *rv.ptr<std::string>();
    } else {
      e->ret_str = rv.operator std::string();
    }
    if (rv.type_code() == kBytes) {
      e->ret_bytes.data = e->ret_str.c_str();
      e->ret_bytes.size = e->ret_str.length();
      *ret_type_code = kBytes;
      ret_val->v_handle = &(e->ret_bytes);
    } else {
      *ret_type_code = kStr;
      ret_val->v_str = e->ret_str.c_str();
    }
  } else {
    rv.MoveToCHost(ret_val, ret_type_code);
  }
  API_END();
}

int DGLCFuncSetReturn(
    DGLRetValueHandle ret, DGLValue* value, int* type_code, int num_ret) {
  API_BEGIN();
  CHECK_EQ(num_ret, 1);
  DGLRetValue* rv = static_cast<DGLRetValue*>(ret);
  *rv = DGLArgValue(value[0], type_code[0]);
  API_END();
}

int DGLFuncCreateFromCFunc(
    DGLPackedCFunc func, void* resource_handle, DGLPackedCFuncFinalizer fin,
    DGLFunctionHandle* out) {
  API_BEGIN();
  if (fin == nullptr) {
    *out =
        new PackedFunc([func, resource_handle](DGLArgs args, DGLRetValue* rv) {
          int ret = func(
              (DGLValue*)args.values, (int*)args.type_codes,  // NOLINT(*)
              args.num_args, rv, resource_handle);
          if (ret != 0) {
            std::string err = "DGLCall CFunc Error:\n";
            err += DGLGetLastError();
            throw dmlc::Error(err);
          }
        });
  } else {
    // wrap it in a shared_ptr, with fin as deleter.
    // so fin will be called when the lambda went out of scope.
    std::shared_ptr<void> rpack(resource_handle, fin);
    *out = new PackedFunc([func, rpack](DGLArgs args, DGLRetValue* rv) {
      int ret = func(
          (DGLValue*)args.values, (int*)args.type_codes,  // NOLINT(*)
          args.num_args, rv, rpack.get());
      if (ret != 0) {
        std::string err = "DGLCall CFunc Error:\n";
        err += DGLGetLastError();
        throw dmlc::Error(err);
      }
    });
  }
  API_END();
}

int DGLStreamCreate(int device_type, int device_id, DGLStreamHandle* out) {
  API_BEGIN();
  DGLContext ctx;
  ctx.device_type = static_cast<DGLDeviceType>(device_type);
  ctx.device_id = device_id;
  *out = DeviceAPIManager::Get(ctx)->CreateStream(ctx);
  API_END();
}

int DGLStreamFree(int device_type, int device_id, DGLStreamHandle stream) {
  API_BEGIN();
  DGLContext ctx;
  ctx.device_type = static_cast<DGLDeviceType>(device_type);
  ctx.device_id = device_id;
  DeviceAPIManager::Get(ctx)->FreeStream(ctx, stream);
  API_END();
}

int DGLSetStream(int device_type, int device_id, DGLStreamHandle stream) {
  API_BEGIN();
  DGLContext ctx;
  ctx.device_type = static_cast<DGLDeviceType>(device_type);
  ctx.device_id = device_id;
  DeviceAPIManager::Get(ctx)->SetStream(ctx, stream);
  API_END();
}

int DGLGetStream(int device_type, int device_id, DGLStreamHandle* stream) {
  API_BEGIN();
  DGLContext ctx;
  ctx.device_type = static_cast<DGLDeviceType>(device_type);
  ctx.device_id = device_id;
  *stream = DeviceAPIManager::Get(ctx)->GetStream();
  API_END();
}

int DGLSynchronize(int device_type, int device_id, DGLStreamHandle stream) {
  API_BEGIN();
  DGLContext ctx;
  ctx.device_type = static_cast<DGLDeviceType>(device_type);
  ctx.device_id = device_id;
  DeviceAPIManager::Get(ctx)->StreamSync(ctx, stream);
  API_END();
}

int DGLStreamStreamSynchronize(
    int device_type, int device_id, DGLStreamHandle src, DGLStreamHandle dst) {
  API_BEGIN();
  DGLContext ctx;
  ctx.device_type = static_cast<DGLDeviceType>(device_type);
  ctx.device_id = device_id;
  DeviceAPIManager::Get(ctx)->SyncStreamFromTo(ctx, src, dst);
  API_END();
}

int DGLCbArgToReturn(DGLValue* value, int code) {
  API_BEGIN();
  dgl::runtime::DGLRetValue rv;
  rv = dgl::runtime::DGLArgValue(*value, code);
  int tcode;
  rv.MoveToCHost(value, &tcode);
  CHECK_EQ(tcode, code);
  API_END();
}

int DGLLoadTensorAdapter(const char* path) {
  return TensorDispatcher::Global()->Load(path) ? 0 : -1;
}

// set device api
DGL_REGISTER_GLOBAL(dgl::runtime::symbol::dgl_set_device)
    .set_body([](DGLArgs args, DGLRetValue* ret) {
      DGLContext ctx;
      ctx.device_type = static_cast<DGLDeviceType>(args[0].operator int());
      ctx.device_id = args[1];
      DeviceAPIManager::Get(ctx)->SetDevice(ctx);
    });

// set device api
DGL_REGISTER_GLOBAL("_GetDeviceAttr")
    .set_body([](DGLArgs args, DGLRetValue* ret) {
      DGLContext ctx;
      ctx.device_type = static_cast<DGLDeviceType>(args[0].operator int());
      ctx.device_id = args[1];

      DeviceAttrKind kind = static_cast<DeviceAttrKind>(args[2].operator int());
      if (kind == kExist) {
        DeviceAPI* api = DeviceAPIManager::Get(ctx.device_type, true);
        if (api != nullptr) {
          api->GetAttr(ctx, kind, ret);
        } else {
          *ret = 0;
        }
      } else {
        DeviceAPIManager::Get(ctx)->GetAttr(ctx, kind, ret);
      }
    });
