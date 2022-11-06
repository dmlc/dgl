/**
 *  Copyright (c) 2017 by Contributors
 * @file system_lib_module.cc
 * @brief SystemLib module.
 */
#include <dgl/runtime/c_backend_api.h>
#include <dgl/runtime/registry.h>

#include <mutex>

#include "module_util.h"

namespace dgl {
namespace runtime {

class SystemLibModuleNode : public ModuleNode {
 public:
  SystemLibModuleNode() = default;

  const char* type_key() const final { return "system_lib"; }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    std::lock_guard<std::mutex> lock(mutex_);

    if (module_blob_ != nullptr) {
      // If we previously recorded submodules, load them now.
      ImportModuleBlob(reinterpret_cast<const char*>(module_blob_), &imports_);
      module_blob_ = nullptr;
    }

    auto it = tbl_.find(name);
    if (it != tbl_.end()) {
      return WrapPackedFunc(
          reinterpret_cast<BackendPackedCFunc>(it->second), sptr_to_self);
    } else {
      return PackedFunc();
    }
  }

  void RegisterSymbol(const std::string& name, void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (name == symbol::dgl_module_ctx) {
      void** ctx_addr = reinterpret_cast<void**>(ptr);
      *ctx_addr = this;
    } else if (name == symbol::dgl_dev_mblob) {
      // Record pointer to content of submodules to be loaded.
      // We defer loading submodules to the first call to GetFunction().
      // The reason is that RegisterSymbol() gets called when initializing the
      // syslib (i.e. library loading time), and the registeries aren't ready
      // yet. Therefore, we might not have the functionality to load submodules
      // now.
      CHECK(module_blob_ == nullptr) << "Resetting mobule blob?";
      module_blob_ = ptr;
    } else {
      auto it = tbl_.find(name);
      if (it != tbl_.end() && ptr != it->second) {
        LOG(WARNING) << "SystemLib symbol " << name
                     << " get overriden to a different address " << ptr << "->"
                     << it->second;
      }
      tbl_[name] = ptr;
    }
  }

  static const std::shared_ptr<SystemLibModuleNode>& Global() {
    static std::shared_ptr<SystemLibModuleNode> inst =
        std::make_shared<SystemLibModuleNode>();
    return inst;
  }

 private:
  // Internal mutex
  std::mutex mutex_;
  // Internal symbol table
  std::unordered_map<std::string, void*> tbl_;
  // Module blob to be imported
  void* module_blob_{nullptr};
};

DGL_REGISTER_GLOBAL("module._GetSystemLib")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      *rv = runtime::Module(SystemLibModuleNode::Global());
    });
}  // namespace runtime
}  // namespace dgl

int DGLBackendRegisterSystemLibSymbol(const char* name, void* ptr) {
  dgl::runtime::SystemLibModuleNode::Global()->RegisterSymbol(name, ptr);
  return 0;
}
