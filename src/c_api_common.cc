/*!
 *  Copyright (c) 2018 by Contributors
 * \file c_runtime_api.cc
 * \brief DGL C API common implementations
 */
#include "c_api_common.h"
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::PackedFunc;
using dgl::runtime::NDArray;

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

using namespace runtime;

DGL_REGISTER_GLOBAL("_Test")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CHECK(args[0].IsObjectType<List<Value>>());
    auto l = args[0].AsObjectRef<List<Value>>();
    for (int i = 0; i < l.size(); ++i) {
      LOG(INFO) << "Item#" << i << ": " << l[i]->data.operator int64_t();
    }
  });

DGL_REGISTER_GLOBAL("_Test2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    using ArgType = Map<std::string, List<Value>>;
    CHECK(args[0].IsObjectType<ArgType>());
    auto m = args[0].AsObjectRef<ArgType>();
    for (const auto& kv : m) {
      LOG(INFO) << "Key: " << kv.first;
      for (const auto& ele : kv.second) {
        LOG(INFO) << "\t" << ele->data.operator int64_t();
      }
    }
  });

}  // namespace dgl
