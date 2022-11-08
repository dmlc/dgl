/**
 *  Copyright (c) 2020 by Contributors
 * @file featgraph/src/featgraph.cc
 * @brief FeatGraph kernels.
 */
#include <dmlc/logging.h>
#include <featgraph.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

namespace dgl {
namespace featgraph {

/* @brief Singleton that loads the featgraph module. */
class FeatGraphModule {
 public:
  static FeatGraphModule* Global() {
    static FeatGraphModule inst;
    return &inst;
  }

  void Load(const std::string& path) {
    mod = tvm::runtime::Module::LoadFromFile(path);
  }

  inline tvm::runtime::ModuleNode* Get() {
    auto ret = mod.operator->();
    if (!ret) {
      LOG(FATAL) << "FeatGraph module have not been loaded. "
                 << "Please set path of featgraph shared library.";
    }
    return ret;
  }

 private:
  tvm::runtime::Module mod;
  FeatGraphModule() {}
};

/* @brief Load Featgraph module from given path. */
void LoadFeatGraphModule(const std::string& path) {
  FeatGraphModule::Global()->Load(path);
}

/* @brief Convert DLDataType to string. */
inline std::string DTypeAsStr(const DLDataType& t) {
  switch (t.code) {
    case 0U:
      return "int" + std::to_string(t.bits);
    case 1U:
      return "uint" + std::to_string(t.bits);
    case 2U:
      return "float" + std::to_string(t.bits);
    case 3U:
      return "bfloat" + std::to_string(t.bits);
    default:
      LOG(FATAL) << "Type code " << t.code << " not recognized";
  }
}

/* @brief Get operator filename. */
inline std::string GetOperatorName(
    const std::string& base_name, const DLDataType& dtype,
    const DLDataType& idtype) {
  return base_name + "_" + DTypeAsStr(dtype) + "_" + DTypeAsStr(idtype);
}

/* @brief Call FeatGraph's SDDMM kernel. */
void SDDMMTreeReduction(
    DLManagedTensor* row, DLManagedTensor* col, DLManagedTensor* lhs,
    DLManagedTensor* rhs, DLManagedTensor* out) {
  tvm::runtime::ModuleNode* mod = FeatGraphModule::Global()->Get();
  std::string f_name = GetOperatorName(
      "SDDMMTreeReduction", (row->dl_tensor).dtype, (lhs->dl_tensor).dtype);
  tvm::runtime::PackedFunc f = mod->GetFunction(f_name);
  if (f != nullptr) f(row, col, lhs, rhs, out);
}

}  // namespace featgraph
}  // namespace dgl
