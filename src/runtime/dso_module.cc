/**
 *  Copyright (c) 2017 by Contributors
 * @file dso_dll_module.cc
 * @brief Module to load from dynamic shared library.
 */
#include <dgl/runtime/module.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>

#include "module_util.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace dgl {
namespace runtime {

// Module to load from dynamic shared libary.
// This is the default module DGL used for host-side AOT
class DSOModuleNode final : public ModuleNode {
 public:
  ~DSOModuleNode() {
    if (lib_handle_) Unload();
  }

  const char* type_key() const final { return "dso"; }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    BackendPackedCFunc faddr;
    if (name == runtime::symbol::dgl_module_main) {
      const char* entry_name = reinterpret_cast<const char*>(
          GetSymbol(runtime::symbol::dgl_module_main));
      CHECK(entry_name != nullptr)
          << "Symbol " << runtime::symbol::dgl_module_main
          << " is not presented";
      faddr = reinterpret_cast<BackendPackedCFunc>(GetSymbol(entry_name));
    } else {
      faddr = reinterpret_cast<BackendPackedCFunc>(GetSymbol(name.c_str()));
    }
    if (faddr == nullptr) return PackedFunc();
    return WrapPackedFunc(faddr, sptr_to_self);
  }

  void Init(const std::string& name) {
    Load(name);
    if (auto* ctx_addr = reinterpret_cast<void**>(
            GetSymbol(runtime::symbol::dgl_module_ctx))) {
      *ctx_addr = this;
    }
    InitContextFunctions(
        [this](const char* fname) { return GetSymbol(fname); });
    // Load the imported modules
    const char* dev_mblob = reinterpret_cast<const char*>(
        GetSymbol(runtime::symbol::dgl_dev_mblob));
    if (dev_mblob != nullptr) {
      ImportModuleBlob(dev_mblob, &imports_);
    }
  }

 private:
  // Platform dependent handling.
#if defined(_WIN32)
  // library handle
  HMODULE lib_handle_{nullptr};
  // Load the library
  void Load(const std::string& name) {
    // use wstring version that is needed by LLVM.
    std::wstring wname(name.begin(), name.end());
    lib_handle_ = LoadLibraryW(wname.c_str());
    CHECK(lib_handle_ != nullptr)
        << "Failed to load dynamic shared library " << name;
  }
  void* GetSymbol(const char* name) {
    return reinterpret_cast<void*>(
        GetProcAddress(lib_handle_, (LPCSTR)name));  // NOLINT(*)
  }
  void Unload() { FreeLibrary(lib_handle_); }
#else
  // Library handle
  void* lib_handle_{nullptr};
  // load the library
  void Load(const std::string& name) {
    lib_handle_ = dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
    CHECK(lib_handle_ != nullptr)
        << "Failed to load dynamic shared library " << name << " " << dlerror();
  }
  void* GetSymbol(const char* name) { return dlsym(lib_handle_, name); }
  void Unload() { dlclose(lib_handle_); }
#endif
};

DGL_REGISTER_GLOBAL("module.loadfile_so")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      std::shared_ptr<DSOModuleNode> n = std::make_shared<DSOModuleNode>();
      n->Init(args[0]);
      *rv = runtime::Module(n);
    });
}  // namespace runtime
}  // namespace dgl
