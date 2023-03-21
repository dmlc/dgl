/**
 *  Copyright (c) 2018 by Contributors
 * @file c_api_common.cc
 * @brief DGL C API common implementations
 */
#include "c_api_common.h"

#include <dgl/graph_interface.h>

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::NDArray;
using dgl::runtime::PackedFunc;

namespace dgl {

PackedFunc ConvertNDArrayVectorToPackedFunc(const std::vector<NDArray>& vec) {
  auto body = [vec](DGLArgs args, DGLRetValue* rv) {
    const uint64_t which = args[0];
    if (which >= vec.size()) {
      LOG(FATAL) << "invalid choice";
    } else {
      *rv = std::move(vec[which]);
    }
  };
  return PackedFunc(body);
}

PackedFunc ConvertEdgeArrayToPackedFunc(const EdgeArray& ea) {
  auto body = [ea](DGLArgs args, DGLRetValue* rv) {
    const int which = args[0];
    if (which == 0) {
      *rv = std::move(ea.src);
    } else if (which == 1) {
      *rv = std::move(ea.dst);
    } else if (which == 2) {
      *rv = std::move(ea.id);
    } else {
      LOG(FATAL) << "invalid choice";
    }
  };
  return PackedFunc(body);
}

}  // namespace dgl
