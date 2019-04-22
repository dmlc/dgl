/*!
 *  Copyright (c) 2018 by Contributors
 * \file c_runtime_api.cc
 * \brief DGL C API common implementations
 */
#include "c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::PackedFunc;
using dgl::runtime::NDArray;

namespace dgl {

DLManagedTensor* CreateTmpDLManagedTensor(const DGLArgValue& arg) {
  const DLTensor* dl_tensor = arg;
  DLManagedTensor* ret = new DLManagedTensor();
  ret->deleter = [] (DLManagedTensor* self) { delete self; };
  ret->manager_ctx = nullptr;
  ret->dl_tensor = *dl_tensor;
  return ret;
}

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

DGL_REGISTER_GLOBAL("_GetVectorWrapperSize")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* ptr = args[0];
    const CAPIVectorWrapper* wrapper = static_cast<const CAPIVectorWrapper*>(ptr);
    *rv = static_cast<int64_t>(wrapper->pointers.size());
  });

DGL_REGISTER_GLOBAL("_GetVectorWrapperData")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* ptr = args[0];
    CAPIVectorWrapper* wrapper = static_cast<CAPIVectorWrapper*>(ptr);
    *rv = static_cast<void*>(wrapper->pointers.data());
  });

DGL_REGISTER_GLOBAL("_FreeVectorWrapper")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* ptr = args[0];
    CAPIVectorWrapper* wrapper = static_cast<CAPIVectorWrapper*>(ptr);
    delete wrapper;
  });

}  // namespace dgl
