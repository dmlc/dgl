#include <dgl/c_api_common.h>

namespace dgl {

DLManagedTensor* CreateTmpDLManagedTensor(const TVMArgValue& arg) {
  const DLTensor* dl_tensor = arg;
  DLManagedTensor* ret = new DLManagedTensor();
  ret->deleter = [] (DLManagedTensor* self) { delete self; };
  ret->manager_ctx = nullptr;
  ret->dl_tensor = *dl_tensor;
  return ret;
}

PackedFunc ConvertNDArrayVectorToPackedFunc(const std::vector<NDArray>& vec) {
    auto body = [vec](TVMArgs args, TVMRetValue* rv) {
        size_t which = args[0];
        if (which >= vec.size()) {
            LOG(FATAL) << "invalid choice";
        } else {
            *rv = std::move(vec[which]);
        }
    };
    return PackedFunc(body);
}

} // namespace dgl

