/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/graph.cc
 * \brief DGL graph index APIs
 */

#include <string.h>
#include <dgl/graph.h>
#include "../c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::PackedFunc;
using dgl::runtime::NDArray;

namespace dgl {

std::unordered_map<std::string, DLDataType> dtype_map = {
  {"float32", DLDataType{kDLFloat, 32, 1}},
  {"int32", DLDataType{kDLInt, 32, 1}},
};

DGL_REGISTER_GLOBAL("contrib.graph_store._CAPI_DGLCreateSharedMem")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string mem_name = args[0];
    dgl::runtime::NDArray shape_arr = args[1];
    std::string dtype_str = args[2];
    std::string fill = args[3];
    bool is_create = args[4];
    auto it = dtype_map.find(dtype_str);
    CHECK(it != dtype_map.end()) << "Unsupported dtype " << dtype_str;
    auto dtype = it->second;
    std::vector<int64_t> shape(shape_arr->shape[0]);
    auto *shape_data = static_cast<int64_t *>(shape_arr->data);
    for (size_t i = 0; i < shape.size(); i++)
      shape[i] = shape_data[i];
    NDArray arr = NDArray::EmptyShared(mem_name, shape, dtype,
                                       DLContext{kDLCPU, 0}, is_create);
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
    mem_size *= data->dtype.bits / 8;
    NDArray arr = NDArray::EmptyShared(mem_name, shape, data->dtype,
                                       DLContext{kDLCPU, 0}, true);
    memcpy(arr->data, data->data, mem_size);
    *rv = arr;
  });

}  // namespace dgl
