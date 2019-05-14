#include <dgl/array.h>

namespace dgl {

IdArray NewIdArray(int64_t length) {
  return IdArray::Empty({length}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
}

IdArray VecToIdArray(const std::vector<dgl_id_t>& vec) {
  IdArray ret = NewIdArray(vec.size());
  std::copy(vec.begin(), vec.end(), static_cast<dgl_id_t*>(ret->data));
  return ret;
}

IdArray Clone(IdArray arr) {
  IdArray ret = NewIdArray(arr->shape[0]);
  return ret;
}

}  // namespace dgl
