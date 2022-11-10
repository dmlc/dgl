/**
 *  Copyright (c) 2019 by Contributors
 * @file runtime/object.h
 * @brief Defines the Object data structures.
 */
#ifndef DGL_RUNTIME_OBJECT_H_
#define DGL_RUNTIME_OBJECT_H_

#include <dmlc/logging.h>

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace dgl {
namespace runtime {

// forward declaration
class Object;
class ObjectRef;
class NDArray;

/**
 * @brief Visitor class to each object attribute.
 *  The content is going to be called for each field.
 */
class AttrVisitor {
 public:
  //! \cond Doxygen_Suppress
  virtual void Visit(const char* key, double* value) = 0;
  virtual void Visit(const char* key, int64_t* value) = 0;
  virtual void Visit(const char* key, uint64_t* value) = 0;
  virtual void Visit(const char* key, int* value) = 0;
  virtual void Visit(const char* key, bool* value) = 0;
  virtual void Visit(const char* key, std::string* value) = 0;
  virtual void Visit(const char* key, ObjectRef* value) = 0;
  virtual void Visit(const char* key, NDArray* value) = 0;
  template <
      typename ENum,
      typename = typename std::enable_if<std::is_enum<ENum>::value>::type>
  void Visit(const char* key, ENum* ptr) {
    static_assert(
        std::is_same<int, typename std::underlying_type<ENum>::type>::value,
        "declare enum to be enum int to use visitor");
    this->Visit(key, reinterpret_cast<int*>(ptr));
  }
  //! \endcond
};

/**
 * @brief base class of object container.
 *  All object's internal is stored as std::shared_ptr<Object>
 */
class Object {
 public:
  /** @brief virtual destructor */
  virtual ~Object() {}
  /** @return The unique type key of the object */
  virtual const char* type_key() const = 0;
  /**
   * @brief Apply visitor to each field of the Object
   *  Visitor could mutate the content of the object.
   *  override if Object contains attribute fields.
   * @param visitor The visitor
   */
  virtual void VisitAttrs(AttrVisitor* visitor) {}
  /** @return the type index of the object */
  virtual uint32_t type_index() const = 0;
  /**
   * @brief Whether this object derives from object with type_index=tid.
   *  Implemented by DGL_DECLARE_OBJECT_TYPE_INFO
   *
   * @param tid The type index.
   * @return the check result.
   */
  virtual bool _DerivedFrom(uint32_t tid) const;
  /**
   * @brief get a runtime unique type index given a type key
   * @param type_key Type key of a type.
   * @return the corresponding type index.
   */
  static uint32_t TypeKey2Index(const char* type_key);
  /**
   * @brief get type key from type index.
   * @param index The type index
   * @return the corresponding type key.
   */
  static const char* TypeIndex2Key(uint32_t index);
  /**
   * @return whether the type is derived from
   */
  template <typename T>
  inline bool derived_from() const;
  /**
   * @return whether the object is of type T
   * @tparam The type to be checked.
   */
  template <typename T>
  inline bool is_type() const;
  // object ref can see this
  friend class ObjectRef;
  static constexpr const char* _type_key = "Object";
};

/** @brief base class of all reference object */
class ObjectRef {
 public:
  /** @brief type indicate the container type */
  using ContainerType = Object;
  /**
   * @brief Comparator
   *
   * Compare with the two are referencing to the same object (compare by
   * address).
   *
   * @param other Another object ref.
   * @return the compare result.
   * @sa same_as
   */
  inline bool operator==(const ObjectRef& other) const;
  /**
   * @brief Comparator
   *
   * Compare with the two are referencing to the same object (compare by
   * address).
   *
   * @param other Another object ref.
   * @return the compare result.
   */
  inline bool same_as(const ObjectRef& other) const;
  /**
   * @brief Comparator
   *
   * The operator overload allows ObjectRef be used in std::map.
   *
   * @param other Another object ref.
   * @return the compare result.
   */
  inline bool operator<(const ObjectRef& other) const;
  /**
   * @brief Comparator
   * @param other Another object ref.
   * @return the compare result.
   * @sa same_as
   */
  inline bool operator!=(const ObjectRef& other) const;
  /** @return the hash function for ObjectRef */
  inline size_t hash() const;
  /** @return whether the expression is null */
  inline bool defined() const;
  /** @return the internal type index of Object */
  inline uint32_t type_index() const;
  /** @return the internal object pointer */
  inline const Object* get() const;
  /** @return the internal object pointer */
  inline const Object* operator->() const;
  /**
   * @brief Downcast this object to its actual type.
   * This returns nullptr if the object is not of the requested type.
   * Example usage:
   *
   * if (const Banana *banana = obj->as<Banana>()) {
   *   // This is a Banana!
   * }
   * @tparam T the target type, must be subtype of Object
   */
  template <typename T>
  inline const T* as() const;

  /** @brief default constructor */
  ObjectRef() = default;
  explicit ObjectRef(std::shared_ptr<Object> obj) : obj_(obj) {}

  /** @brief the internal object, do not touch */
  std::shared_ptr<Object> obj_;
};

/**
 * @brief helper macro to declare type information in a base object.
 *
 * This is macro should be used in abstract base class definition
 * because it does not define type_key and type_index.
 */
#define DGL_DECLARE_BASE_OBJECT_INFO(TypeName, Parent)         \
  const bool _DerivedFrom(uint32_t tid) const override {       \
    static uint32_t tidx = TypeKey2Index(TypeName::_type_key); \
    if (tidx == tid) return true;                              \
    return Parent::_DerivedFrom(tid);                          \
  }

/**
 * @brief helper macro to declare type information in a terminal class
 *
 * This is macro should be used in terminal class definition.
 *
 * For example:
 *
 * // This class is an abstract class and cannot create instances
 * class SomeBaseClass : public Object {
 *  public:
 *   static constexpr const char* _type_key = "some_base";
 *   DGL_DECLARE_BASE_OBJECT_INFO(SomeBaseClass, Object);
 * };
 *
 * // Child class that allows instantiation
 * class SomeChildClass : public SomeBaseClass {
 *  public:
 *   static constexpr const char* _type_key = "some_child";
 *   DGL_DECLARE_OBJECT_TYPE_INFO(SomeChildClass, SomeBaseClass);
 * };
 */
#define DGL_DECLARE_OBJECT_TYPE_INFO(TypeName, Parent)               \
  const char* type_key() const final { return TypeName::_type_key; } \
  uint32_t type_index() const final {                                \
    static uint32_t tidx = TypeKey2Index(TypeName::_type_key);       \
    return tidx;                                                     \
  }                                                                  \
  bool _DerivedFrom(uint32_t tid) const final {                      \
    static uint32_t tidx = TypeKey2Index(TypeName::_type_key);       \
    if (tidx == tid) return true;                                    \
    return Parent::_DerivedFrom(tid);                                \
  }

/** @brief Macro to generate common object reference class method definition */
#define DGL_DEFINE_OBJECT_REF_METHODS(TypeName, BaseTypeName, ObjectName)   \
  TypeName() {}                                                             \
  explicit TypeName(std::shared_ptr<runtime::Object> obj)                   \
      : BaseTypeName(obj) {}                                                \
  const ObjectName* operator->() const {                                    \
    return static_cast<const ObjectName*>(obj_.get());                      \
  }                                                                         \
  ObjectName* operator->() { return static_cast<ObjectName*>(obj_.get()); } \
  std::shared_ptr<ObjectName> sptr() const {                                \
    return CHECK_NOTNULL(std::dynamic_pointer_cast<ObjectName>(obj_));      \
  }                                                                         \
  operator bool() const { return this->defined(); }                         \
  using ContainerType = ObjectName

/** @brief Macro to generate object reference class definition */
#define DGL_DEFINE_OBJECT_REF(TypeName, ObjectName)       \
  class TypeName : public ::dgl::runtime::ObjectRef {     \
   public:                                                \
    DGL_DEFINE_OBJECT_REF_METHODS(                        \
        TypeName, ::dgl::runtime::ObjectRef, ObjectName); \
  }

// implementations of inline functions after this
template <typename T>
inline bool Object::is_type() const {
  // use static field so query only happens once.
  static uint32_t type_id = Object::TypeKey2Index(T::_type_key);
  return type_id == this->type_index();
}

template <typename T>
inline bool Object::derived_from() const {
  // use static field so query only happens once.
  static uint32_t type_id = Object::TypeKey2Index(T::_type_key);
  return this->_DerivedFrom(type_id);
}

inline const Object* ObjectRef::get() const { return obj_.get(); }

inline const Object* ObjectRef::operator->() const { return obj_.get(); }

inline bool ObjectRef::defined() const { return obj_.get() != nullptr; }

inline bool ObjectRef::operator==(const ObjectRef& other) const {
  return obj_.get() == other.obj_.get();
}

inline bool ObjectRef::same_as(const ObjectRef& other) const {
  return obj_.get() == other.obj_.get();
}

inline bool ObjectRef::operator<(const ObjectRef& other) const {
  return obj_.get() < other.obj_.get();
}

inline bool ObjectRef::operator!=(const ObjectRef& other) const {
  return obj_.get() != other.obj_.get();
}

inline size_t ObjectRef::hash() const {
  return std::hash<Object*>()(obj_.get());
}

inline uint32_t ObjectRef::type_index() const {
  CHECK(obj_.get() != nullptr) << "null type";
  return get()->type_index();
}

template <typename T>
inline const T* ObjectRef::as() const {
  const Object* ptr = get();
  if (ptr && ptr->is_type<T>()) {
    return static_cast<const T*>(ptr);
  }
  return nullptr;
}

/** @brief The hash function for nodes */
struct ObjectHash {
  size_t operator()(const ObjectRef& a) const { return a.hash(); }
};

/** @brief The equal comparator for nodes */
struct ObjectEqual {
  bool operator()(const ObjectRef& a, const ObjectRef& b) const {
    return a.get() == b.get();
  }
};

}  // namespace runtime
}  // namespace dgl

namespace std {
template <>
struct hash<::dgl::runtime::ObjectRef> {
  std::size_t operator()(const ::dgl::runtime::ObjectRef& k) const {
    return k.hash();
  }
};

}  // namespace std

#endif  // DGL_RUNTIME_OBJECT_H_
