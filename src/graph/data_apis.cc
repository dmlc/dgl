#include <string.h>
#include <dgl/graph.h>
#include "../c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::PackedFunc;
using dgl::runtime::NDArray;

namespace dgl {

DGL_REGISTER_GLOBAL("contrib.graph_store._CAPI_DGLCreateSharedMem")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string mem_name = args[0];
    int64_t num_nodes = args[1];
    int64_t num_feats = args[2];
    int dtype = args[3];
    std::string fill = args[4];
    bool is_create = args[5];
    // TODO use the dtype correctly.
    NDArray arr = NDArray::EmptyShared(mem_name, {num_nodes, num_feats},
                                       DLDataType{kDLFloat, 32, 1}, DLContext{kDLCPU, 0},
                                       is_create);
    *rv = arr;
    if (fill == "zero" && is_create)
      memset(arr->data, 0, arr.GetSize());
  });

DGL_REGISTER_GLOBAL("contrib.graph_store._CAPI_DGLCreateSharedMemWithData")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string mem_name = args[0];
    NDArray data = args[1];
    std::vector<int64_t> shape(data->shape, data->shape + data->ndim);
    size_t mem_size = 1;
    for (auto s : shape)
      mem_size *= s;
    // TODO set this correctly.
    mem_size *= 4;
    // TODO use the dtype correctly.
    NDArray arr = NDArray::EmptyShared(mem_name, shape,
                                       DLDataType{kDLFloat, 32, 1}, DLContext{kDLCPU, 0},
                                       true);
    memcpy(arr->data, data->data, mem_size);
    *rv = arr;
  });

}  // namespace dgl
