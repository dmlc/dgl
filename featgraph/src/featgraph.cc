#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <sstream>

#include <featgraph.h>

namespace dgl {
namespace featgraph {

static tvm::runtime::Module featgraph = tvm::runtime::Module::LoadFromFile("/home/ubuntu/dgl/build/libfeatgraph_kernels.so");

void SDDMMTreeReduction(DLManagedTensor* row, DLManagedTensor* col, 
                        DLManagedTensor* lhs, DLManagedTensor* rhs, 
                        DLManagedTensor* out) {
  std::stringstream ss;
  ss << "SDDMMTreeReduction_int" << (int) (row->dl_tensor).dtype.bits;
  ss << "_float" << (int) (lhs->dl_tensor).dtype.bits;
  std::string f_name = ss.str();
  tvm::runtime::PackedFunc f = featgraph.GetFunction(f_name);
  if (f != nullptr)
    f(row, col, lhs, rhs, out);
}

}  // namespace featgraph
}  // namespace dgl
