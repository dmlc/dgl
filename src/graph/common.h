#pragma once

#include <dgl/graph_interface.h>

namespace dgl {

inline IdArray NewIdArray(int64_t length) {
  return IdArray::Empty({length}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
}

inline IdArray VecToIdArray(const std::vector<dgl_id_t>& vec) {
  IdArray ret = NewIdArray(vec.size());
  std::copy(vec.begin(), vec.end(), static_cast<dgl_id_t*>(ret->data));
  return ret;
}

}  // namespace dgl
