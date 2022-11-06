/**
 *  Copyright (c) 2016 by Contributors
 * Implementation of C API (reference: tvm/src/api/c_api.cc)
 * @file c_api.cc
 */
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/c_object_api.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/object.h>
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>

#include <exception>
#include <string>
#include <vector>

#include "runtime_base.h"

/** @brief entry to to easily hold returning information */
struct DGLAPIThreadLocalEntry {
  /** @brief result holder for returning strings */
  std::vector<std::string> ret_vec_str;
  /** @brief result holder for returning string pointers */
  std::vector<const char*> ret_vec_charp;
  /** @brief result holder for retruning string */
  std::string ret_str;
};

using namespace dgl::runtime;

/** @brief Thread local store that can be used to hold return values. */
typedef dmlc::ThreadLocalStore<DGLAPIThreadLocalEntry> DGLAPIThreadLocalStore;

using DGLAPIObject = std::shared_ptr<Object>;

struct APIAttrGetter : public AttrVisitor {
  std::string skey;
  DGLRetValue* ret;
  bool found_object_ref{false};

  void Visit(const char* key, double* value) final {
    if (skey == key) *ret = value[0];
  }
  void Visit(const char* key, int64_t* value) final {
    if (skey == key) *ret = value[0];
  }
  void Visit(const char* key, uint64_t* value) final {
    CHECK_LE(
        value[0], static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        << "cannot return too big constant";
    if (skey == key) *ret = static_cast<int64_t>(value[0]);
  }
  void Visit(const char* key, int* value) final {
    if (skey == key) *ret = static_cast<int64_t>(value[0]);
  }
  void Visit(const char* key, bool* value) final {
    if (skey == key) *ret = static_cast<int64_t>(value[0]);
  }
  void Visit(const char* key, std::string* value) final {
    if (skey == key) *ret = value[0];
  }
  void Visit(const char* key, ObjectRef* value) final {
    if (skey == key) {
      *ret = value[0];
      found_object_ref = true;
    }
  }
  void Visit(const char* key, NDArray* value) final {
    if (skey == key) *ret = value[0];
  }
};

struct APIAttrDir : public AttrVisitor {
  std::vector<std::string>* names;

  void Visit(const char* key, double* value) final { names->push_back(key); }
  void Visit(const char* key, int64_t* value) final { names->push_back(key); }
  void Visit(const char* key, uint64_t* value) final { names->push_back(key); }
  void Visit(const char* key, bool* value) final { names->push_back(key); }
  void Visit(const char* key, int* value) final { names->push_back(key); }
  void Visit(const char* key, std::string* value) final {
    names->push_back(key);
  }
  void Visit(const char* key, ObjectRef* value) final { names->push_back(key); }
  void Visit(const char* key, NDArray* value) final { names->push_back(key); }
};

int DGLObjectFree(ObjectHandle handle) {
  API_BEGIN();
  delete static_cast<DGLAPIObject*>(handle);
  API_END();
}

int DGLObjectTypeKey2Index(const char* type_key, int* out_index) {
  API_BEGIN();
  *out_index = static_cast<int>(Object::TypeKey2Index(type_key));
  API_END();
}

int DGLObjectGetTypeIndex(ObjectHandle handle, int* out_index) {
  API_BEGIN();
  *out_index =
      static_cast<int>((*static_cast<DGLAPIObject*>(handle))->type_index());
  API_END();
}

int DGLObjectGetAttr(
    ObjectHandle handle, const char* key, DGLValue* ret_val, int* ret_type_code,
    int* ret_success) {
  API_BEGIN();
  DGLRetValue rv;
  APIAttrGetter getter;
  getter.skey = key;
  getter.ret = &rv;
  DGLAPIObject* tobject = static_cast<DGLAPIObject*>(handle);
  if (getter.skey == "type_key") {
    ret_val->v_str = (*tobject)->type_key();
    *ret_type_code = kStr;
    *ret_success = 1;
  } else {
    (*tobject)->VisitAttrs(&getter);
    *ret_success = getter.found_object_ref || rv.type_code() != kNull;
    if (rv.type_code() == kStr || rv.type_code() == kDGLDataType) {
      DGLAPIThreadLocalEntry* e = DGLAPIThreadLocalStore::Get();
      e->ret_str = rv.operator std::string();
      *ret_type_code = kStr;
      ret_val->v_str = e->ret_str.c_str();
    } else {
      rv.MoveToCHost(ret_val, ret_type_code);
    }
  }
  API_END();
}

int DGLObjectListAttrNames(
    ObjectHandle handle, int* out_size, const char*** out_array) {
  DGLAPIThreadLocalEntry* ret = DGLAPIThreadLocalStore::Get();
  API_BEGIN();
  ret->ret_vec_str.clear();
  DGLAPIObject* tobject = static_cast<DGLAPIObject*>(handle);
  APIAttrDir dir;
  dir.names = &(ret->ret_vec_str);
  (*tobject)->VisitAttrs(&dir);
  ret->ret_vec_charp.clear();
  for (size_t i = 0; i < ret->ret_vec_str.size(); ++i) {
    ret->ret_vec_charp.push_back(ret->ret_vec_str[i].c_str());
  }
  *out_array = dmlc::BeginPtr(ret->ret_vec_charp);
  *out_size = static_cast<int>(ret->ret_vec_str.size());
  API_END();
}
