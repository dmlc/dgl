#pragma once

#include <dgl/graph_interface.h>

namespace dgl {

inline IdArray NewIdArray(int64_t length) {
  return IdArray::Empty({length}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
}

}  // namespace dgl
