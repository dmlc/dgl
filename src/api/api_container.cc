/**
 *  Copyright (c) 2019 by Contributors
 * @file api/api_container.cc
 * @brief Runtime container APIs. (reference: tvm/src/api/api_lang.cc)
 */
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/registry.h>

namespace dgl {
namespace runtime {

DGL_REGISTER_GLOBAL("_List").set_body([](DGLArgs args, DGLRetValue* rv) {
  auto ret_obj = std::make_shared<runtime::ListObject>();
  for (int i = 0; i < args.size(); ++i) {
    ret_obj->data.push_back(args[i].obj_sptr());
  }
  *rv = ret_obj;
});

DGL_REGISTER_GLOBAL("_ListGetItem").set_body([](DGLArgs args, DGLRetValue* rv) {
  auto& sptr = args[0].obj_sptr();
  CHECK(sptr->is_type<ListObject>());
  auto* o = static_cast<const ListObject*>(sptr.get());
  int64_t i = args[1];
  CHECK_LT(i, o->data.size()) << "list out of bound";
  *rv = o->data[i];
});

DGL_REGISTER_GLOBAL("_ListSize").set_body([](DGLArgs args, DGLRetValue* rv) {
  auto& sptr = args[0].obj_sptr();
  CHECK(sptr->is_type<ListObject>());
  auto* o = static_cast<const ListObject*>(sptr.get());
  *rv = static_cast<int64_t>(o->data.size());
});

DGL_REGISTER_GLOBAL("_Map").set_body([](DGLArgs args, DGLRetValue* rv) {
  CHECK_EQ(args.size() % 2, 0);
  if (args.size() != 0 && args[0].type_code() == kStr) {
    // StrMap
    StrMapObject::ContainerType data;
    for (int i = 0; i < args.size(); i += 2) {
      CHECK(args[i].type_code() == kStr) << "The key of the map must be string";
      CHECK(args[i + 1].type_code() == kObjectHandle)
          << "The value of the map must be an object type";
      data.emplace(std::make_pair(
          args[i].operator std::string(), args[i + 1].obj_sptr()));
    }
    auto obj = std::make_shared<StrMapObject>();
    obj->data = std::move(data);
    *rv = obj;
  } else {
    // object container
    MapObject::ContainerType data;
    for (int i = 0; i < args.size(); i += 2) {
      CHECK(args[i].type_code() == kObjectHandle)
          << "The key of the map must be an object type";
      CHECK(args[i + 1].type_code() == kObjectHandle)
          << "The value of the map must be an object type";
      data.emplace(std::make_pair(args[i].obj_sptr(), args[i + 1].obj_sptr()));
    }
    auto obj = std::make_shared<MapObject>();
    obj->data = std::move(data);
    *rv = obj;
  }
});

DGL_REGISTER_GLOBAL("_EmptyStrMap").set_body([](DGLArgs args, DGLRetValue* rv) {
  StrMapObject::ContainerType data;
  auto obj = std::make_shared<StrMapObject>();
  obj->data = std::move(data);
  *rv = obj;
});

DGL_REGISTER_GLOBAL("_MapSize").set_body([](DGLArgs args, DGLRetValue* rv) {
  auto& sptr = args[0].obj_sptr();
  if (sptr->is_type<MapObject>()) {
    auto* o = static_cast<const MapObject*>(sptr.get());
    *rv = static_cast<int64_t>(o->data.size());
  } else {
    CHECK(sptr->is_type<StrMapObject>());
    auto* o = static_cast<const StrMapObject*>(sptr.get());
    *rv = static_cast<int64_t>(o->data.size());
  }
});

DGL_REGISTER_GLOBAL("_MapGetItem").set_body([](DGLArgs args, DGLRetValue* rv) {
  auto& sptr = args[0].obj_sptr();
  if (sptr->is_type<MapObject>()) {
    auto* o = static_cast<const MapObject*>(sptr.get());
    auto it = o->data.find(args[1].obj_sptr());
    CHECK(it != o->data.end()) << "cannot find the key in the map";
    *rv = (*it).second;
  } else {
    CHECK(sptr->is_type<StrMapObject>());
    auto* o = static_cast<const StrMapObject*>(sptr.get());
    auto it = o->data.find(args[1].operator std::string());
    CHECK(it != o->data.end()) << "cannot find the key in the map";
    *rv = (*it).second;
  }
});

DGL_REGISTER_GLOBAL("_MapItems").set_body([](DGLArgs args, DGLRetValue* rv) {
  auto& sptr = args[0].obj_sptr();
  if (sptr->is_type<MapObject>()) {
    auto* o = static_cast<const MapObject*>(sptr.get());
    auto rkvs = std::make_shared<ListObject>();
    for (const auto& kv : o->data) {
      rkvs->data.push_back(kv.first);
      rkvs->data.push_back(kv.second);
    }
    *rv = rkvs;
  } else {
    CHECK(sptr->is_type<StrMapObject>());
    auto* o = static_cast<const StrMapObject*>(sptr.get());
    auto rkvs = std::make_shared<ListObject>();
    for (const auto& kv : o->data) {
      rkvs->data.push_back(MakeValue(kv.first));
      rkvs->data.push_back(kv.second);
    }
    *rv = rkvs;
  }
});

DGL_REGISTER_GLOBAL("_MapCount").set_body([](DGLArgs args, DGLRetValue* rv) {
  auto& sptr = args[0].obj_sptr();
  if (sptr->is_type<MapObject>()) {
    auto* o = static_cast<const MapObject*>(sptr.get());
    *rv = static_cast<int64_t>(o->data.count(args[1].obj_sptr()));
  } else {
    CHECK(sptr->is_type<StrMapObject>());
    auto* o = static_cast<const StrMapObject*>(sptr.get());
    *rv = static_cast<int64_t>(o->data.count(args[1].operator std::string()));
  }
});

DGL_REGISTER_GLOBAL("_Value").set_body([](DGLArgs args, DGLRetValue* rv) {
  *rv = MakeValue(args[0]);
});

DGL_REGISTER_GLOBAL("_ValueGet").set_body([](DGLArgs args, DGLRetValue* rv) {
  auto& sptr = args[0].obj_sptr();
  CHECK(sptr->is_type<ValueObject>());
  auto* o = static_cast<const ValueObject*>(sptr.get());
  *rv = o->data;
});

}  // namespace runtime
}  // namespace dgl
