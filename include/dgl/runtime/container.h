/**
 *  Copyright (c) 2019 by Contributors
 * @file runtime/container.h
 * @brief Defines the container object data structures.
 */
#ifndef DGL_RUNTIME_CONTAINER_H_
#define DGL_RUNTIME_CONTAINER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "object.h"
#include "packed_func.h"

namespace dgl {
namespace runtime {

/**
 * @brief value object.
 *
 * It is typically used to wrap a non-Object type to Object type.
 * Any type that is supported by DGLRetValue is supported by this.
 */
class ValueObject : public Object {
 public:
  /** @brief the value data */
  DGLRetValue data;

  static constexpr const char* _type_key = "Value";
  DGL_DECLARE_OBJECT_TYPE_INFO(ValueObject, Object);
};

/** @brief Construct a value object. */
template <typename T>
inline std::shared_ptr<ValueObject> MakeValue(T&& val) {
  auto obj = std::make_shared<ValueObject>();
  obj->data = val;
  return obj;
}

/** @brief Vallue reference type */
class Value : public ObjectRef {
 public:
  Value() {}
  explicit Value(std::shared_ptr<Object> o) : ObjectRef(o) {}

  const ValueObject* operator->() const {
    return static_cast<const ValueObject*>(obj_.get());
  }

  using ContainerType = ValueObject;
};

/** @brief list obj content in list */
class ListObject : public Object {
 public:
  /** @brief the data content */
  std::vector<std::shared_ptr<Object> > data;

  void VisitAttrs(AttrVisitor* visitor) final {
    // Visitor to list have no effect.
  }

  static constexpr const char* _type_key = "List";
  DGL_DECLARE_OBJECT_TYPE_INFO(ListObject, Object);
};

/** @brief map obj content */
class MapObject : public Object {
 public:
  void VisitAttrs(AttrVisitor* visitor) final {
    // Visitor to map have no effect.
  }
  // hash function
  struct Hash {
    size_t operator()(const std::shared_ptr<Object>& n) const {
      return std::hash<Object*>()(n.get());
    }
  };
  // comparator
  struct Equal {
    bool operator()(
        const std::shared_ptr<Object>& a,
        const std::shared_ptr<Object>& b) const {
      return a.get() == b.get();
    }
  };

  /** @brief The corresponding conatiner type */
  using ContainerType = std::unordered_map<
      std::shared_ptr<Object>, std::shared_ptr<Object>, Hash, Equal>;

  /** @brief the data content */
  ContainerType data;

  static constexpr const char* _type_key = "Map";
  DGL_DECLARE_OBJECT_TYPE_INFO(MapObject, Object);
};

/** @brief specialized map obj with string as key */
class StrMapObject : public Object {
 public:
  void VisitAttrs(AttrVisitor* visitor) final {
    // Visitor to map have no effect.
  }
  /** @brief The corresponding conatiner type */
  using ContainerType =
      std::unordered_map<std::string, std::shared_ptr<Object> >;

  /** @brief the data content */
  ContainerType data;

  static constexpr const char* _type_key = "StrMap";
  DGL_DECLARE_OBJECT_TYPE_INFO(StrMapObject, Object);
};

/**
 * @brief iterator adapter that adapts TIter to return another type.
 * @tparam Converter a struct that contains converting function
 * @tparam TIter the content iterator type.
 */
template <typename Converter, typename TIter>
class IterAdapter {
 public:
  explicit IterAdapter(TIter iter) : iter_(iter) {}
  inline IterAdapter& operator++() {  // NOLINT(*)
    ++iter_;
    return *this;
  }
  inline IterAdapter& operator++(int) {  // NOLINT(*)
    ++iter_;
    return *this;
  }
  inline IterAdapter operator+(int offset) const {  // NOLINT(*)
    return IterAdapter(iter_ + offset);
  }
  inline bool operator==(IterAdapter other) const {
    return iter_ == other.iter_;
  }
  inline bool operator!=(IterAdapter other) const { return !(*this == other); }
  inline const typename Converter::ResultType operator*() const {
    return Converter::convert(*iter_);
  }

 private:
  TIter iter_;
};

/**
 * @brief List container of ObjectRef.
 *
 * List implements copy on write semantics, which means list is mutable
 * but copy will happen when list is referenced in more than two places.
 *
 * That is said when using this container for runtime arguments or return
 * values, try use the constructor to create the list at once (for example
 * from an existing vector).
 *
 * operator[] only provide const access, use Set to mutate the content.
 *
 * @tparam T The content ObjectRef type.
 *
 * @note The element type must subclass \c ObjectRef.  Otherwise, the
 * compiler would throw an error:
 *
 * <code>
 *      error: no type named 'type' in 'struct std::enable_if<false, void>'
 * </code>
 *
 * Example:
 *
 * <code>
 *     // List<int> list;          // fails
 *     // List<NDArray> list2;     // fails
 *     List<Value> list;           // works
 *     list.push_back(Value(MakeValue(1)));  // works
 *     list.push_back(Value(MakeValue(NDArray::Empty(shape, dtype, ctx))));  //
 * works
 * </code>
 */
template <
    typename T,
    typename =
        typename std::enable_if<std::is_base_of<ObjectRef, T>::value>::type>
class List : public ObjectRef {
 public:
  /**
   * @brief default constructor
   */
  List() { obj_ = std::make_shared<ListObject>(); }
  /**
   * @brief move constructor
   * @param other source
   */
  List(List<T>&& other) {  // NOLINT(*)
    obj_ = std::move(other.obj_);
  }
  /**
   * @brief copy constructor
   * @param other source
   */
  List(const List<T>& other) : ObjectRef(other.obj_) {  // NOLINT(*)
  }
  /**
   * @brief constructor from pointer
   * @param n the container pointer
   */
  explicit List(std::shared_ptr<Object> n) : ObjectRef(n) {}
  /**
   * @brief constructor from iterator
   * @param begin begin of iterator
   * @param end end of iterator
   * @tparam IterType The type of iterator
   */
  template <typename IterType>
  List(IterType begin, IterType end) {
    assign(begin, end);
  }
  /**
   * @brief constructor from initializer list
   * @param init The initalizer list
   */
  List(std::initializer_list<T> init) {  // NOLINT(*)
    assign(init.begin(), init.end());
  }
  /**
   * @brief constructor from vector
   * @param init The vector
   */
  List(const std::vector<T>& init) {  // NOLINT(*)
    assign(init.begin(), init.end());
  }
  /**
   * @brief Constructs a container with n elements. Each element is a copy of
   * val
   * @param n The size of the container
   * @param val The init value
   */
  explicit List(size_t n, const T& val) {
    auto tmp_obj = std::make_shared<ListObject>();
    for (size_t i = 0; i < n; ++i) {
      tmp_obj->data.push_back(val.obj_);
    }
    obj_ = std::move(tmp_obj);
  }
  /**
   * @brief move assign operator
   * @param other The source of assignment
   * @return reference to self.
   */
  List<T>& operator=(List<T>&& other) {
    obj_ = std::move(other.obj_);
    return *this;
  }
  /**
   * @brief copy assign operator
   * @param other The source of assignment
   * @return reference to self.
   */
  List<T>& operator=(const List<T>& other) {
    obj_ = other.obj_;
    return *this;
  }
  /**
   * @brief reset the list to content from iterator.
   * @param begin begin of iterator
   * @param end end of iterator
   * @tparam IterType The type of iterator
   */
  template <typename IterType>
  void assign(IterType begin, IterType end) {
    auto n = std::make_shared<ListObject>();
    for (IterType it = begin; it != end; ++it) {
      n->data.push_back((*it).obj_);
    }
    obj_ = std::move(n);
  }
  /**
   * @brief Read i-th element from list.
   * @param i The index
   * @return the i-th element.
   */
  inline const T operator[](size_t i) const {
    return T(static_cast<const ListObject*>(obj_.get())->data[i]);
  }
  /** @return The size of the list */
  inline size_t size() const {
    if (obj_.get() == nullptr) return 0;
    return static_cast<const ListObject*>(obj_.get())->data.size();
  }
  /**
   * @brief copy on write semantics
   *  Do nothing if current handle is the unique copy of the list.
   *  Otherwise make a new copy of the list to ensure the current handle
   *  hold a unique copy.
   *
   * @return Handle to the internal obj container(which ganrantees to be unique)
   */
  inline ListObject* CopyOnWrite() {
    if (obj_.get() == nullptr || !obj_.unique()) {
      obj_ = std::make_shared<ListObject>(
          *static_cast<const ListObject*>(obj_.get()));
    }
    return static_cast<ListObject*>(obj_.get());
  }
  /**
   * @brief push a new item to the back of the list
   * @param item The item to be pushed.
   */
  inline void push_back(const T& item) {
    ListObject* n = this->CopyOnWrite();
    n->data.push_back(item.obj_);
  }
  /**
   * @brief set i-th element of the list.
   * @param i The index
   * @param value The value to be setted.
   */
  inline void Set(size_t i, const T& value) {
    ListObject* n = this->CopyOnWrite();
    n->data[i] = value.obj_;
  }
  /** @return whether list is empty */
  inline bool empty() const { return size() == 0; }
  /** @brief Copy the content to a vector */
  inline std::vector<T> ToVector() const {
    return std::vector<T>(begin(), end());
  }
  /** @brief specify container obj */
  using ContainerType = ListObject;

  struct Ptr2ObjectRef {
    using ResultType = T;
    static inline T convert(const std::shared_ptr<Object>& n) { return T(n); }
  };
  using iterator = IterAdapter<
      Ptr2ObjectRef, std::vector<std::shared_ptr<Object> >::const_iterator>;

  using reverse_iterator = IterAdapter<
      Ptr2ObjectRef,
      std::vector<std::shared_ptr<Object> >::const_reverse_iterator>;

  /** @return begin iterator */
  inline iterator begin() const {
    return iterator(static_cast<const ListObject*>(obj_.get())->data.begin());
  }
  /** @return end iterator */
  inline iterator end() const {
    return iterator(static_cast<const ListObject*>(obj_.get())->data.end());
  }
  /** @return rbegin iterator */
  inline reverse_iterator rbegin() const {
    return reverse_iterator(
        static_cast<const ListObject*>(obj_.get())->data.rbegin());
  }
  /** @return rend iterator */
  inline reverse_iterator rend() const {
    return reverse_iterator(
        static_cast<const ListObject*>(obj_.get())->data.rend());
  }
};

/**
 * @brief Map container of ObjectRef->ObjectRef.
 *
 * Map implements copy on write semantics, which means map is mutable
 * but copy will happen when list is referenced in more than two places.
 *
 * That is said when using this container for runtime arguments or return
 * values, try use the constructor to create it at once (for example
 * from an existing std::map).
 *
 * operator[] only provide const acces, use Set to mutate the content.
 *
 * @tparam K The key ObjectRef type.
 * @tparam V The value ObjectRef type.
 *
 * @note The element type must subclass \c ObjectRef.  Otherwise, the
 * compiler would throw an error:
 *
 * <code>
 *      error: no type named 'type' in 'struct std::enable_if<false, void>'
 * </code>
 *
 * Example:
 *
 * <code>
 *     // Map<std::string, int> map;          // fails
 *     // Map<std::string, NDArray> map2;     // fails
 *     Map<std::string, Value> map;           // works
 *     map.Set("key1", Value(MakeValue(1)));  // works
 *     map.Set("key2", Value(MakeValue(NDArray::Empty(shape, dtype, ctx))));  //
 * works
 * </code>
 */
template <
    typename K, typename V,
    typename = typename std::enable_if<
        std::is_base_of<ObjectRef, K>::value ||
        std::is_base_of<std::string, K>::value>::type,
    typename =
        typename std::enable_if<std::is_base_of<ObjectRef, V>::value>::type>
class Map : public ObjectRef {
 public:
  /**
   * @brief default constructor
   */
  Map() { obj_ = std::make_shared<MapObject>(); }
  /**
   * @brief move constructor
   * @param other source
   */
  Map(Map<K, V>&& other) {  // NOLINT(*)
    obj_ = std::move(other.obj_);
  }
  /**
   * @brief copy constructor
   * @param other source
   */
  Map(const Map<K, V>& other) : ObjectRef(other.obj_) {  // NOLINT(*)
  }
  /**
   * @brief constructor from pointer
   * @param n the container pointer
   */
  explicit Map(std::shared_ptr<Object> n) : ObjectRef(n) {}
  /**
   * @brief constructor from iterator
   * @param begin begin of iterator
   * @param end end of iterator
   * @tparam IterType The type of iterator
   */
  template <typename IterType>
  Map(IterType begin, IterType end) {
    assign(begin, end);
  }
  /**
   * @brief constructor from initializer list
   * @param init The initalizer list
   */
  Map(std::initializer_list<std::pair<K, V> > init) {  // NOLINT(*)
    assign(init.begin(), init.end());
  }
  /**
   * @brief constructor from vector
   * @param init The vector
   */
  template <typename Hash, typename Equal>
  Map(const std::unordered_map<K, V, Hash, Equal>& init) {  // NOLINT(*)
    assign(init.begin(), init.end());
  }
  /**
   * @brief move assign operator
   * @param other The source of assignment
   * @return reference to self.
   */
  Map<K, V>& operator=(Map<K, V>&& other) {
    obj_ = std::move(other.obj_);
    return *this;
  }
  /**
   * @brief copy assign operator
   * @param other The source of assignment
   * @return reference to self.
   */
  Map<K, V>& operator=(const Map<K, V>& other) {
    obj_ = other.obj_;
    return *this;
  }
  /**
   * @brief reset the list to content from iterator.
   * @param begin begin of iterator
   * @param end end of iterator
   * @tparam IterType The type of iterator
   */
  template <typename IterType>
  void assign(IterType begin, IterType end) {
    auto n = std::shared_ptr<MapObject>();
    for (IterType i = begin; i != end; ++i) {
      n->data.emplace(std::make_pair(i->first.obj_, i->second.obj_));
    }
    obj_ = std::move(n);
  }
  /**
   * @brief Read element from map.
   * @param key The key
   * @return the corresonding element.
   */
  inline const V operator[](const K& key) const {
    return V(static_cast<const MapObject*>(obj_.get())->data.at(key.obj_));
  }
  /**
   * @brief Read element from map.
   * @param key The key
   * @return the corresonding element.
   */
  inline const V at(const K& key) const {
    return V(static_cast<const MapObject*>(obj_.get())->data.at(key.obj_));
  }
  /** @return The size of the list */
  inline size_t size() const {
    if (obj_.get() == nullptr) return 0;
    return static_cast<const MapObject*>(obj_.get())->data.size();
  }
  /** @return The size of the list */
  inline size_t count(const K& key) const {
    if (obj_.get() == nullptr) return 0;
    return static_cast<const MapObject*>(obj_.get())->data.count(key.obj_);
  }
  /**
   * @brief copy on write semantics
   *  Do nothing if current handle is the unique copy of the list.
   *  Otherwise make a new copy of the list to ensure the current handle
   *  hold a unique copy.
   *
   * @return Handle to the internal obj container(which ganrantees to be unique)
   */
  inline MapObject* CopyOnWrite() {
    if (obj_.get() == nullptr || !obj_.unique()) {
      obj_ = std::make_shared<MapObject>(
          *static_cast<const MapObject*>(obj_.get()));
    }
    return static_cast<MapObject*>(obj_.get());
  }
  /**
   * @brief set the Map.
   * @param key The index key.
   * @param value The value to be setted.
   */
  inline void Set(const K& key, const V& value) {
    MapObject* n = this->CopyOnWrite();
    n->data[key.obj_] = value.obj_;
  }

  /** @return whether list is empty */
  inline bool empty() const { return size() == 0; }
  /** @brief specify container obj */
  using ContainerType = MapObject;

  struct Ptr2ObjectRef {
    using ResultType = std::pair<K, V>;
    static inline ResultType convert(
        const std::pair<std::shared_ptr<Object>, std::shared_ptr<Object> >& n) {
      return std::make_pair(K(n.first), V(n.second));
    }
  };

  using iterator =
      IterAdapter<Ptr2ObjectRef, MapObject::ContainerType::const_iterator>;

  /** @return begin iterator */
  inline iterator begin() const {
    return iterator(static_cast<const MapObject*>(obj_.get())->data.begin());
  }
  /** @return end iterator */
  inline iterator end() const {
    return iterator(static_cast<const MapObject*>(obj_.get())->data.end());
  }
  /** @return begin iterator */
  inline iterator find(const K& key) const {
    return iterator(
        static_cast<const MapObject*>(obj_.get())->data.find(key.obj_));
  }
};

// specialize of string map
template <typename V, typename T1, typename T2>
class Map<std::string, V, T1, T2> : public ObjectRef {
 public:
  // for code reuse
  Map() { obj_ = std::make_shared<StrMapObject>(); }
  Map(Map<std::string, V>&& other) {  // NOLINT(*)
    obj_ = std::move(other.obj_);
  }
  Map(const Map<std::string, V>& other) : ObjectRef(other.obj_) {  // NOLINT(*)
  }
  explicit Map(std::shared_ptr<Object> n) : ObjectRef(n) {}
  template <typename IterType>
  Map(IterType begin, IterType end) {
    assign(begin, end);
  }
  Map(std::initializer_list<std::pair<std::string, V> > init) {  // NOLINT(*)
    assign(init.begin(), init.end());
  }

  template <typename Hash, typename Equal>
  Map(const std::unordered_map<std::string, V, Hash, Equal>&
          init) {  // NOLINT(*)
    assign(init.begin(), init.end());
  }
  Map<std::string, V>& operator=(Map<std::string, V>&& other) {
    obj_ = std::move(other.obj_);
    return *this;
  }
  Map<std::string, V>& operator=(const Map<std::string, V>& other) {
    obj_ = other.obj_;
    return *this;
  }
  template <typename IterType>
  void assign(IterType begin, IterType end) {
    auto n = std::make_shared<StrMapObject>();
    for (IterType i = begin; i != end; ++i) {
      n->data.emplace(std::make_pair(i->first, i->second.obj_));
    }
    obj_ = std::move(n);
  }
  inline const V operator[](const std::string& key) const {
    return V(static_cast<const StrMapObject*>(obj_.get())->data.at(key));
  }
  inline const V at(const std::string& key) const {
    return V(static_cast<const StrMapObject*>(obj_.get())->data.at(key));
  }
  inline size_t size() const {
    if (obj_.get() == nullptr) return 0;
    return static_cast<const StrMapObject*>(obj_.get())->data.size();
  }
  inline size_t count(const std::string& key) const {
    if (obj_.get() == nullptr) return 0;
    return static_cast<const StrMapObject*>(obj_.get())->data.count(key);
  }
  inline StrMapObject* CopyOnWrite() {
    if (obj_.get() == nullptr || !obj_.unique()) {
      obj_ = std::make_shared<MapObject>(
          *static_cast<const MapObject*>(obj_.get()));
    }
    return static_cast<StrMapObject*>(obj_.get());
  }
  inline void Set(const std::string& key, const V& value) {
    StrMapObject* n = this->CopyOnWrite();
    n->data[key] = value.obj_;
  }
  inline bool empty() const { return size() == 0; }
  using ContainerType = StrMapObject;

  struct Ptr2ObjectRef {
    using ResultType = std::pair<std::string, V>;
    static inline ResultType convert(
        const std::pair<std::string, std::shared_ptr<Object> >& n) {
      return std::make_pair(n.first, V(n.second));
    }
  };

  using iterator =
      IterAdapter<Ptr2ObjectRef, StrMapObject::ContainerType::const_iterator>;

  /** @return begin iterator */
  inline iterator begin() const {
    return iterator(static_cast<const StrMapObject*>(obj_.get())->data.begin());
  }
  /** @return end iterator */
  inline iterator end() const {
    return iterator(static_cast<const StrMapObject*>(obj_.get())->data.end());
  }
  /** @return begin iterator */
  inline iterator find(const std::string& key) const {
    return iterator(
        static_cast<const StrMapObject*>(obj_.get())->data.find(key));
  }
};

/**
 * @brief Helper function to convert a List<Value> object to a vector.
 * @tparam T element type
 * @param list Input list object.
 * @return std vector
 */
template <typename T>
inline std::vector<T> ListValueToVector(const List<Value>& list) {
  std::vector<T> ret;
  ret.reserve(list.size());
  for (Value val : list)
    // (BarclayII) apparently MSVC 2017 CL 19.10 had trouble parsing
    //     ret.push_back(val->data)
    // So I kindly tell it how to properly parse it.
    ret.push_back(val->data.operator T());
  return ret;
}

}  // namespace runtime
}  // namespace dgl

#endif  // DGL_RUNTIME_CONTAINER_H_
