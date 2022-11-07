/**
 *  Copyright (c) 2019 by Contributors
 * @file packed_func_ext.h
 * @brief Extension package to PackedFunc
 *   This enables pass ObjectRef types into/from PackedFunc.
 */
#ifndef DGL_PACKED_FUNC_EXT_H_
#define DGL_PACKED_FUNC_EXT_H_

#include <memory>
#include <sstream>
#include <string>
#include <type_traits>

#include "./runtime/container.h"
#include "./runtime/object.h"
#include "./runtime/packed_func.h"

namespace dgl {
namespace runtime {
/**
 * @brief Runtime type checker for node type.
 * @tparam T the type to be checked.
 */
template <typename T>
struct ObjectTypeChecker {
  static inline bool Check(Object* sptr) {
    // This is the only place in the project where RTTI is used
    // It can be turned off, but will make non strict checking.
    // TODO(tqchen) possibly find alternative to turn of RTTI
    using ContainerType = typename T::ContainerType;
    return sptr->derived_from<ContainerType>();
  }
  static inline void PrintName(std::ostringstream& os) {  // NOLINT(*)
    using ContainerType = typename T::ContainerType;
    os << ContainerType::_type_key;
  }
};

template <typename T>
struct ObjectTypeChecker<List<T> > {
  static inline bool Check(Object* sptr) {
    if (sptr == nullptr) return false;
    if (!sptr->is_type<ListObject>()) return false;
    ListObject* n = static_cast<ListObject*>(sptr);
    for (const auto& p : n->data) {
      if (!ObjectTypeChecker<T>::Check(p.get())) return false;
    }
    return true;
  }
  static inline void PrintName(std::ostringstream& os) {  // NOLINT(*)
    os << "list<";
    ObjectTypeChecker<T>::PrintName(os);
    os << ">";
  }
};

template <typename V>
struct ObjectTypeChecker<Map<std::string, V> > {
  static inline bool Check(Object* sptr) {
    if (sptr == nullptr) return false;
    if (!sptr->is_type<StrMapObject>()) return false;
    StrMapObject* n = static_cast<StrMapObject*>(sptr);
    for (const auto& kv : n->data) {
      if (!ObjectTypeChecker<V>::Check(kv.second.get())) return false;
    }
    return true;
  }
  static inline void PrintName(std::ostringstream& os) {  // NOLINT(*)
    os << "map<string";
    os << ',';
    ObjectTypeChecker<V>::PrintName(os);
    os << '>';
  }
};

template <typename K, typename V>
struct ObjectTypeChecker<Map<K, V> > {
  static inline bool Check(Object* sptr) {
    if (sptr == nullptr) return false;
    if (!sptr->is_type<MapObject>()) return false;
    MapObject* n = static_cast<MapObject*>(sptr);
    for (const auto& kv : n->data) {
      if (!ObjectTypeChecker<K>::Check(kv.first.get())) return false;
      if (!ObjectTypeChecker<V>::Check(kv.second.get())) return false;
    }
    return true;
  }
  static inline void PrintName(std::ostringstream& os) {  // NOLINT(*)
    os << "map<";
    ObjectTypeChecker<K>::PrintName(os);
    os << ',';
    ObjectTypeChecker<V>::PrintName(os);
    os << '>';
  }
};

template <typename T>
inline std::string NodeTypeName() {
  std::ostringstream os;
  ObjectTypeChecker<T>::PrintName(os);
  return os.str();
}

// extensions for DGLArgValue

template <typename TObjectRef>
inline TObjectRef DGLArgValue::AsObjectRef() const {
  static_assert(
      std::is_base_of<ObjectRef, TObjectRef>::value,
      "Conversion only works for ObjectRef derived class");
  if (type_code_ == kNull) return TObjectRef();
  DGL_CHECK_TYPE_CODE(type_code_, kObjectHandle);
  std::shared_ptr<Object>& sptr = *ptr<std::shared_ptr<Object> >();
  CHECK(ObjectTypeChecker<TObjectRef>::Check(sptr.get()))
      << "Expected type " << NodeTypeName<TObjectRef>() << " but get "
      << sptr->type_key();
  return TObjectRef(sptr);
}

inline std::shared_ptr<Object>& DGLArgValue::obj_sptr() {
  DGL_CHECK_TYPE_CODE(type_code_, kObjectHandle);
  return *ptr<std::shared_ptr<Object> >();
}

template <typename TObjectRef, typename>
inline bool DGLArgValue::IsObjectType() const {
  DGL_CHECK_TYPE_CODE(type_code_, kObjectHandle);
  std::shared_ptr<Object>& sptr = *ptr<std::shared_ptr<Object> >();
  return ObjectTypeChecker<TObjectRef>::Check(sptr.get());
}

// extensions for DGLRetValue

inline DGLRetValue& DGLRetValue::operator=(
    const std::shared_ptr<Object>& other) {
  if (other.get() == nullptr) {
    SwitchToPOD(kNull);
  } else {
    SwitchToClass<std::shared_ptr<Object> >(kObjectHandle, other);
  }
  return *this;
}

inline DGLRetValue& DGLRetValue::operator=(const ObjectRef& other) {
  if (!other.defined()) {
    SwitchToPOD(kNull);
  } else {
    SwitchToClass<std::shared_ptr<Object> >(kObjectHandle, other.obj_);
  }
  return *this;
}

template <typename TObjectRef>
inline TObjectRef DGLRetValue::AsObjectRef() const {
  static_assert(
      std::is_base_of<ObjectRef, TObjectRef>::value,
      "Conversion only works for ObjectRef");
  if (type_code_ == kNull) return TObjectRef();
  DGL_CHECK_TYPE_CODE(type_code_, kObjectHandle);
  return TObjectRef(*ptr<std::shared_ptr<Object> >());
}

inline void DGLArgsSetter::operator()(
    size_t i, const ObjectRef& other) const {  // NOLINT(*)
  if (other.defined()) {
    values_[i].v_handle = const_cast<std::shared_ptr<Object>*>(&(other.obj_));
    type_codes_[i] = kObjectHandle;
  } else {
    type_codes_[i] = kNull;
  }
}

}  // namespace runtime
}  // namespace dgl
#endif  // DGL_PACKED_FUNC_EXT_H_
