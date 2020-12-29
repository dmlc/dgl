#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <dmlc/logging.h>
#include <sstream>

#include <featgraph.h>

namespace dgl {
namespace featgraph {

/* \brief Singleton that loads the featgraph module.
 */
class FeatGraphModule {
public:
  tvm::runtime::ModuleNode* mod;

  static FeatGraphModule* Global() {
    static FeatGraphModule inst;
    return &inst;
  }

  void SetPath(const std::string& path) {
    mod = const_cast<tvm::runtime::ModuleNode*>(
      tvm::runtime::Module::LoadFromFile(path).operator->());
  }
private:
  FeatGraphModule() {}
};

inline std::string DTypeAsStr(const DLDataType& t) {
  switch(t.code) {
    case 0U: return "int" + std::to_string(t.bits);
    case 1U: return "uint" + std::to_string(t.bits);
    case 2U: return "float" + std::to_string(t.bits);
    case 3U: return "bfloat" + std::to_string(t.bits);
    default: LOG(FATAL) << "Type code " << t.code << " not recognized";
  }
}

inline std::string GetOperatorName(
    const std::string& base_name,
    const DLDataType& dtype,
    const DLDataType& idtype) {
  return base_name + "_" + DTypeAsStr(dtype) + "_" + DTypeAsStr(idtype);
}


void SDDMMTreeReduction(DLManagedTensor* row, DLManagedTensor* col, 
                        DLManagedTensor* lhs, DLManagedTensor* rhs, 
                        DLManagedTensor* out) {
  tvm::runtime::ModuleNode* mod = FeatGraphModule::Global()->mod;
  std::string f_name = GetOperatorName("SDDMMTreeReduction",
                                       (row->dl_tensor).dtype,
                                       (lhs->dl_tensor).dtype);
  tvm::runtime::PackedFunc f = mod->GetFunction(f_name);
  if (f != nullptr)
    f(row, col, lhs, rhs, out);
}

}  // namespace featgraph
}  // namespace dgl
