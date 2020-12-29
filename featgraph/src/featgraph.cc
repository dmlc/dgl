#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <sstream>

#include <featgraph.h>

namespace dgl {
namespace featgraph {

tvm::runtime::Module LoadFeatGraph() {
  static tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile("../build/libfeatgraph_kernels.so");
  return mod;
}

void SDDMMTreeReduction(DLManagedTensor* row, DLManagedTensor* col, 
                        DLManagedTensor* lhs, DLManagedTensor* rhs, 
                        DLManagedTensor* out) {
  const static tvm::runtime::ModuleNode* mod = LoadFeatGraph().operator->();
  std::stringstream ss;
  ss << "SDDMMTreeReduction_int" << (int) (row->dl_tensor).dtype.bits;
  ss << "_float" << (int) (lhs->dl_tensor).dtype.bits;
  std::string f_name = ss.str();
  tvm::runtime::PackedFunc f = const_cast<tvm::runtime::ModuleNode*>(mod)->GetFunction(f_name);
  if (f != nullptr)
    f(row, col, lhs, rhs, out);
}

}  // namespace featgraph
}  // namespace dgl
