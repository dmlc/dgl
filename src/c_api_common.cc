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

class StuffObject : public Object {
 public:
  std::string color;
  int num;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("color", &color);
    v->Visit("num", &num);
  }

  static constexpr const char* _type_key = "Stuff";
  DGL_DECLARE_OBJECT_TYPE_INFO(StuffObject, Object);
};

class Stuff : public ObjectRef {
 public:
  Stuff() {}
  explicit Stuff(std::shared_ptr<Object> o): ObjectRef(o) {}

  const StuffObject* operator->() const {
    return static_cast<const StuffObject*>(obj_.get());
  }

  using ContainerType = StuffObject;
};

DGL_REGISTER_GLOBAL("_CreateStuff")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string color = args[0];
    int64_t num = args[1];
    auto o = std::make_shared<StuffObject>();
    o->color = color;
    o->num = num;
    *rv = o;
  });

DGL_REGISTER_GLOBAL("_Test3")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CHECK(args[0].IsObjectType<List<Stuff>>());
    auto l = args[0].AsObjectRef<List<Stuff>>();
    for (int i = 0; i < l.size(); ++i) {
      LOG(INFO) << "Stuff#" << i << ": color=" << l[i]->color << " num=" << l[i]->num;
    }
  });

DGL_REGISTER_GLOBAL("_CreateBlueStuffs")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    List<Stuff> list;
    for (int i = 0; i < 10; ++i) {
      auto o = std::make_shared<StuffObject>();
      o->color = "blue";
      o->num = i + 10;
      list.push_back(Stuff(o));
    }
    *rv = list;
  });

}  // namespace dgl
