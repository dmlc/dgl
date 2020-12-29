#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <dmlc/logging.h>
#include <sstream>

#include <featgraph.h>

namespace dgl {
namespace featgraph {

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
  return base_name + "_" + DTypeAsStr(dtype) + "_" +DTypeAsStr(idtype);
}

/* \brief Load FeatGraph kernels.
 */
tvm::runtime::Module LoadFeatGraph() {
  static tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile("build/libfeatgraph_kernels.so");
  return mod;
}

void SDDMMTreeReduction(DLManagedTensor* row, DLManagedTensor* col, 
                        DLManagedTensor* lhs, DLManagedTensor* rhs, 
                        DLManagedTensor* out) {
  const static tvm::runtime::ModuleNode* mod = LoadFeatGraph().operator->();
  std::string f_name = GetOperatorName("SDDMMTreeReduction",
                                       (row->dl_tensor).dtype,
                                       (lhs->dl_tensor).dtype);
  tvm::runtime::PackedFunc f = const_cast<tvm::runtime::ModuleNode*>(mod)->GetFunction(f_name);
  if (f != nullptr)
    f(row, col, lhs, rhs, out);
}

}  // namespace featgraph
}  // namespace dgl
