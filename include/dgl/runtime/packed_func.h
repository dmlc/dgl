/**
 *  Copyright (c) 2017 by Contributors
 * @file dgl/runtime/packed_func.h
 * @brief Type-erased function used across DGL API.
 */
#ifndef DGL_RUNTIME_PACKED_FUNC_H_
#define DGL_RUNTIME_PACKED_FUNC_H_

#include <dmlc/logging.h>

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "c_runtime_api.h"
#include "module.h"
#include "ndarray.h"

// Whether use DGL runtime in header only mode.
#ifndef DGL_RUNTIME_HEADER_ONLY
#define DGL_RUNTIME_HEADER_ONLY 0
#endif

namespace dgl {
namespace runtime {

// Forward declare ObjectRef and Object for extensions.
// This header works fine without depend on ObjectRef
// as long as it is not used.
class Object;
class ObjectRef;

// forward declarations
class DGLArgs;
class DGLArgValue;
class DGLRetValue;
class DGLArgsSetter;

/**
 * @brief Packed function is a type-erased function.
 *  The arguments are passed by packed format.
 *
 *  This is an useful unified interface to call generated functions,
 *  It is the unified function function type of DGL.
 *  It corresponds to DGLFunctionHandle in C runtime API.
 */
class PackedFunc {
 public:
  /**
   * @brief The internal std::function
   * @param args The arguments to the function.
   * @param rv The return value.
   *
   * @code
   *   // Example code on how to implemented FType
   *   void MyPackedFunc(DGLArgs args, DGLRetValue* rv) {
   *     // automatically convert arguments to desired type.
   *     int a0 = args[0];
   *     float a1 = args[1];
   *     ...
   *     // automatically assign values to rv
   *     std::string my_return_value = "x";
   *     *rv = my_return_value;
   *   }
   * @endcode
   */
  using FType = std::function<void(DGLArgs args, DGLRetValue* rv)>;
  /** @brief default constructor */
  PackedFunc() {}
  /**
   * @brief constructing a packed function from a std::function.
   * @param body the internal container of packed function.
   */
  explicit PackedFunc(FType body) : body_(body) {}
  /**
   * @brief Call packed function by directly passing in unpacked format.
   * @param args Arguments to be passed.
   * @tparam Args arguments to be passed.
   *
   * @code
   *   // Example code on how to call packed function
   *   void CallPacked(PackedFunc f) {
   *     // call like normal functions by pass in arguments
   *     // return value is automatically converted back
   *     int rvalue = f(1, 2.0);
   *   }
   * @endcode
   */
  template <typename... Args>
  inline DGLRetValue operator()(Args&&... args) const;
  /**
   * @brief Call the function in packed format.
   * @param args The arguments
   * @param rv The return value.
   */
  inline void CallPacked(DGLArgs args, DGLRetValue* rv) const;
  /** @return the internal body function */
  inline FType body() const;
  /** @return Whether the packed function is nullptr */
  bool operator==(std::nullptr_t null) const { return body_ == nullptr; }
  /** @return Whether the packed function is not nullptr */
  bool operator!=(std::nullptr_t null) const { return body_ != nullptr; }

 private:
  /** @brief internal container of packed function */
  FType body_;
};

/**
 * @brief Please refer to \ref TypedPackedFuncAnchor
 * "TypedPackedFunc<R(Args..)>"
 */
template <typename FType>
class TypedPackedFunc;

/**
 * @anchor TypedPackedFuncAnchor
 * @brief A PackedFunc wrapper to provide typed function signature.
 * It is backed by a PackedFunc internally.
 *
 * TypedPackedFunc enables compile time type checking.
 * TypedPackedFunc works with the runtime system:
 * - It can be passed as an argument of PackedFunc.
 * - It can be assigned to DGLRetValue.
 * - It can be directly converted to a type-erased PackedFunc.
 *
 * Developers should prefer TypedPackedFunc over PackedFunc in C++ code
 * as it enables compile time checking.
 * We can construct a TypedPackedFunc from a lambda function
 * with the same signature.
 *
 * @code
 *  // user defined lambda function.
 *  auto addone = [](int x)->int {
 *    return x + 1;
 *  };
 *  // We can directly convert
 *  // lambda function to TypedPackedFunc
 *  TypedPackedFunc<int(int)> ftyped(addone);
 *  // invoke the function.
 *  int y = ftyped(1);
 *  // Can be directly converted to PackedFunc
 *  PackedFunc packed = ftype;
 * @endcode
 * @tparam R The return value of the function.
 * @tparam Args The argument signature of the function.
 */
template <typename R, typename... Args>
class TypedPackedFunc<R(Args...)> {
 public:
  /** @brief short hand for this function type */
  using TSelf = TypedPackedFunc<R(Args...)>;
  /** @brief default constructor */
  TypedPackedFunc() {}
  /**
   * @brief construct by wrap a PackedFunc
   *
   * Example usage:
   * @code
   * PackedFunc packed([](DGLArgs args, DGLRetValue *rv) {
   *   int x = args[0];
   *   *rv = x + 1;
   *  });
   * // construct from packed function
   * TypedPackedFunc<int(int)> ftyped(packed);
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * @endcode
   *
   * @param packed The packed function
   */
  inline explicit TypedPackedFunc(PackedFunc packed);
  /**
   * @brief construct from a lambda function with the same signature.
   *
   * Example usage:
   * @code
   * auto typed_lambda = [](int x)->int { return x + 1; }
   * // construct from packed function
   * TypedPackedFunc<int(int)> ftyped(typed_lambda);
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * @endcode
   *
   * @param typed_lambda typed lambda function.
   * @tparam FLambda the type of the lambda function.
   */
  template <
      typename FLambda, typename = typename std::enable_if<std::is_convertible<
                            FLambda, std::function<R(Args...)> >::value>::type>
  explicit TypedPackedFunc(const FLambda& typed_lambda) {
    this->AssignTypedLambda(typed_lambda);
  }
  /**
   * @brief copy assignment operator from typed lambda
   *
   * Example usage:
   * @code
   * // construct from packed function
   * TypedPackedFunc<int(int)> ftyped;
   * ftyped = [](int x) { return x + 1; }
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * @endcode
   *
   * @param typed_lambda typed lambda function.
   * @tparam FLambda the type of the lambda function.
   * @returns reference to self.
   */
  template <
      typename FLambda, typename = typename std::enable_if<std::is_convertible<
                            FLambda,
                            std::function<R(Args...)> >::value>::type>
  TSelf& operator=(FLambda typed_lambda) {  // NOLINT(*)
    this->AssignTypedLambda(typed_lambda);
    return *this;
  }
  /**
   * @brief copy assignment operator from PackedFunc.
   * @param packed The packed function.
   * @returns reference to self.
   */
  TSelf& operator=(PackedFunc packed) {
    packed_ = packed;
    return *this;
  }
  /**
   * @brief Invoke the operator.
   * @param args The arguments
   * @returns The return value.
   */
  inline R operator()(Args... args) const;
  /**
   * @brief convert to PackedFunc
   * @return the internal PackedFunc
   */
  operator PackedFunc() const { return packed(); }
  /**
   * @return reference the internal PackedFunc
   */
  const PackedFunc& packed() const { return packed_; }

 private:
  friend class DGLRetValue;
  /** @brief The internal packed function */
  PackedFunc packed_;
  /**
   * @brief Assign the packed field using a typed lambda function.
   *
   * @param flambda The lambda function.
   * @tparam FLambda The lambda function type.
   * @note We capture the lambda when possible for maximum efficiency.
   */
  template <typename FLambda>
  inline void AssignTypedLambda(FLambda flambda);
};

/** @brief Arguments into DGL functions. */
class DGLArgs {
 public:
  const DGLValue* values;
  const int* type_codes;
  int num_args;
  /**
   * @brief constructor
   * @param values The argument values
   * @param type_codes The argument type codes
   * @param num_args number of arguments.
   */
  DGLArgs(const DGLValue* values, const int* type_codes, int num_args)
      : values(values), type_codes(type_codes), num_args(num_args) {}
  /** @return size of the arguments */
  inline int size() const;
  /**
   * @brief Get i-th argument
   * @param i the index.
   * @return the ith argument.
   */
  inline DGLArgValue operator[](int i) const;
};

/**
 * @brief Type traits to mark if a class is dgl extension type.
 *
 * To enable extension type in C++ must be register () ed via marco.
 * DGL_REGISTER_EXT_TYPE(TypeName) after defining this with this traits.
 *
 * Extension class can be passed and returned via PackedFunc in all dgl runtime.
 * Internally extension class is stored as T*.
 *
 * @tparam T the typename
 */
template <typename T>
struct extension_class_info {
  static const int code = 0;
};

/**
 * @brief Runtime function table about extension type.
 */
class ExtTypeVTable {
 public:
  /** @brief function to be called to delete a handle */
  void (*destroy)(void* handle);
  /** @brief function to be called when clone a handle */
  void* (*clone)(void* handle);
  /**
   * @brief Register type
   * @tparam T The type to be register.
   * @return The registered vtable.
   */
  template <typename T>
  static inline ExtTypeVTable* Register_();
  /**
   * @brief Get a vtable based on type code.
   * @param type_code The type code
   * @return The registered vtable.
   */
  DGL_DLL static ExtTypeVTable* Get(int type_code);

 private:
  // Internal registration function.
  DGL_DLL static ExtTypeVTable* RegisterInternal(
      int type_code, const ExtTypeVTable& vt);
};

/**
 * @brief Internal base class to
 *  handle conversion to POD values.
 */
class DGLPODValue_ {
 public:
  operator double() const {
    // Allow automatic conversion from int to float
    // This avoids errors when user pass in int from
    // the frontend while the API expects a float.
    if (type_code_ == kDGLInt) {
      return static_cast<double>(value_.v_int64);
    }
    DGL_CHECK_TYPE_CODE(type_code_, kDGLFloat);
    return value_.v_float64;
  }
  operator int64_t() const {
    DGL_CHECK_TYPE_CODE(type_code_, kDGLInt);
    return value_.v_int64;
  }
  operator uint64_t() const {
    DGL_CHECK_TYPE_CODE(type_code_, kDGLInt);
    return value_.v_int64;
  }
  operator int() const {
    DGL_CHECK_TYPE_CODE(type_code_, kDGLInt);
    CHECK_LE(value_.v_int64, std::numeric_limits<int>::max());
    return static_cast<int>(value_.v_int64);
  }
  operator bool() const {
    DGL_CHECK_TYPE_CODE(type_code_, kDGLInt);
    return value_.v_int64 != 0;
  }
  operator void*() const {
    if (type_code_ == kNull) return nullptr;
    if (type_code_ == kArrayHandle) return value_.v_handle;
    DGL_CHECK_TYPE_CODE(type_code_, kHandle);
    return value_.v_handle;
  }
  operator DGLArray*() const {
    if (type_code_ == kArrayHandle || type_code_ == kNDArrayContainer) {
      return static_cast<DGLArray*>(value_.v_handle);
    } else {
      if (type_code_ == kNull) return nullptr;
      LOG(FATAL) << "Expected "
                 << "DGLArray* or NDArray but get " << TypeCode2Str(type_code_);
      return nullptr;
    }
  }
  operator NDArray() const {
    if (type_code_ == kNull) return NDArray();
    DGL_CHECK_TYPE_CODE(type_code_, kNDArrayContainer);
    return NDArray(static_cast<NDArray::Container*>(value_.v_handle));
  }
  operator DGLContext() const {
    DGL_CHECK_TYPE_CODE(type_code_, kDGLContext);
    return value_.v_ctx;
  }
  template <typename TExtension>
  const TExtension& AsExtension() const {
    CHECK_LT(type_code_, kExtEnd);
    return static_cast<TExtension*>(value_.v_handle)[0];
  }
  int type_code() const { return type_code_; }
  /**
   * @brief return handle as specific pointer type.
   * @tparam T the data type.
   * @return The pointer type.
   */
  template <typename T>
  T* ptr() const {
    return static_cast<T*>(value_.v_handle);
  }

 protected:
  friend class DGLArgsSetter;
  friend class DGLRetValue;
  DGLPODValue_() : type_code_(kNull) {}
  DGLPODValue_(DGLValue value, int type_code)
      : value_(value), type_code_(type_code) {}

  /** @brief The value */
  DGLValue value_;
  /** @brief the type code */
  int type_code_;
};

/**
 * @brief A single argument value to PackedFunc.
 *  Containing both type_code and DGLValue
 *
 *  Provides utilities to do type cast into other types.
 */
class DGLArgValue : public DGLPODValue_ {
 public:
  /** @brief default constructor */
  DGLArgValue() {}
  /**
   * @brief constructor
   * @param value of the function
   * @param type_code The type code.
   */
  DGLArgValue(DGLValue value, int type_code) : DGLPODValue_(value, type_code) {}
  // reuse converter from parent
  using DGLPODValue_::operator double;
  using DGLPODValue_::operator int64_t;
  using DGLPODValue_::operator uint64_t;
  using DGLPODValue_::operator int;
  using DGLPODValue_::operator bool;
  using DGLPODValue_::operator void*;
  using DGLPODValue_::operator DGLArray*;
  using DGLPODValue_::operator NDArray;
  using DGLPODValue_::operator DGLContext;

  // conversion operator.
  operator std::string() const {
    if (type_code_ == kDGLDataType) {
      return DGLDataType2String(operator DGLDataType());
    } else if (type_code_ == kBytes) {
      DGLByteArray* arr = static_cast<DGLByteArray*>(value_.v_handle);
      return std::string(arr->data, arr->size);
    } else {
      DGL_CHECK_TYPE_CODE(type_code_, kStr);
      return std::string(value_.v_str);
    }
  }
  operator DGLDataType() const {
    if (type_code_ == kStr) {
      return String2DGLDataType(operator std::string());
    }
    DGL_CHECK_TYPE_CODE(type_code_, kDGLDataType);
    return value_.v_type;
  }
  operator PackedFunc() const {
    if (type_code_ == kNull) return PackedFunc();
    DGL_CHECK_TYPE_CODE(type_code_, kFuncHandle);
    return *ptr<PackedFunc>();
  }
  template <typename FType>
  operator TypedPackedFunc<FType>() const {
    return TypedPackedFunc<FType>(operator PackedFunc());
  }
  operator Module() const {
    DGL_CHECK_TYPE_CODE(type_code_, kModuleHandle);
    return *ptr<Module>();
  }
  const DGLValue& value() const { return value_; }

  // Deferred extension handler.
  template <typename TObjectRef>
  inline TObjectRef AsObjectRef() const;

  // Convert this value to arbitrary class type
  template <
      typename T,
      typename = typename std::enable_if<std::is_class<T>::value>::type>
  inline operator T() const;

  // Return true if the value is of TObjectRef type
  template <
      typename TObjectRef, typename = typename std::enable_if<
                               std::is_class<TObjectRef>::value>::type>
  inline bool IsObjectType() const;

  // get internal node ptr, if it is node
  inline std::shared_ptr<Object>& obj_sptr();
};

/**
 * @brief Return Value container,
 *  Unlike DGLArgValue, which only holds reference and do not delete
 *  the underlying container during destruction.
 *
 *  DGLRetValue holds value and will manage the underlying containers
 *  when it stores a complicated data type.
 */
class DGLRetValue : public DGLPODValue_ {
 public:
  /** @brief default constructor */
  DGLRetValue() {}
  /**
   * @brief move constructor from anoter return value.
   * @param other The other return value.
   */
  DGLRetValue(DGLRetValue&& other)
      : DGLPODValue_(other.value_, other.type_code_) {
    other.value_.v_handle = nullptr;
    other.type_code_ = kNull;
  }
  /** @brief destructor */
  ~DGLRetValue() { this->Clear(); }
  // reuse converter from parent
  using DGLPODValue_::operator double;
  using DGLPODValue_::operator int64_t;
  using DGLPODValue_::operator uint64_t;
  using DGLPODValue_::operator int;
  using DGLPODValue_::operator bool;
  using DGLPODValue_::operator void*;
  using DGLPODValue_::operator DGLArray*;
  using DGLPODValue_::operator DGLContext;
  using DGLPODValue_::operator NDArray;
  // Disable copy and assign from another value, but allow move.
  DGLRetValue(const DGLRetValue& other) { this->Assign(other); }
  // conversion operators
  operator std::string() const {
    if (type_code_ == kDGLDataType) {
      return DGLDataType2String(operator DGLDataType());
    } else if (type_code_ == kBytes) {
      return *ptr<std::string>();
    }
    DGL_CHECK_TYPE_CODE(type_code_, kStr);
    return *ptr<std::string>();
  }
  operator DGLDataType() const {
    if (type_code_ == kStr) {
      return String2DGLDataType(operator std::string());
    }
    DGL_CHECK_TYPE_CODE(type_code_, kDGLDataType);
    return value_.v_type;
  }
  operator PackedFunc() const {
    if (type_code_ == kNull) return PackedFunc();
    DGL_CHECK_TYPE_CODE(type_code_, kFuncHandle);
    return *ptr<PackedFunc>();
  }
  template <typename FType>
  operator TypedPackedFunc<FType>() const {
    return TypedPackedFunc<FType>(operator PackedFunc());
  }
  operator Module() const {
    DGL_CHECK_TYPE_CODE(type_code_, kModuleHandle);
    return *ptr<Module>();
  }
  // Assign operators
  DGLRetValue& operator=(DGLRetValue&& other) {
    this->Clear();
    value_ = other.value_;
    type_code_ = other.type_code_;
    other.type_code_ = kNull;
    return *this;
  }
  DGLRetValue& operator=(double value) {
    this->SwitchToPOD(kDGLFloat);
    value_.v_float64 = value;
    return *this;
  }
  DGLRetValue& operator=(std::nullptr_t value) {
    this->SwitchToPOD(kNull);
    value_.v_handle = value;
    return *this;
  }
  DGLRetValue& operator=(void* value) {
    this->SwitchToPOD(kHandle);
    value_.v_handle = value;
    return *this;
  }
  DGLRetValue& operator=(int64_t value) {
    this->SwitchToPOD(kDGLInt);
    value_.v_int64 = value;
    return *this;
  }
  DGLRetValue& operator=(int value) {
    this->SwitchToPOD(kDGLInt);
    value_.v_int64 = value;
    return *this;
  }
  DGLRetValue& operator=(DGLDataType t) {
    this->SwitchToPOD(kDGLDataType);
    value_.v_type = t;
    return *this;
  }
  DGLRetValue& operator=(DGLContext ctx) {
    this->SwitchToPOD(kDGLContext);
    value_.v_ctx = ctx;
    return *this;
  }
  DGLRetValue& operator=(bool value) {
    this->SwitchToPOD(kDGLInt);
    value_.v_int64 = value;
    return *this;
  }
  DGLRetValue& operator=(std::string value) {
    this->SwitchToClass(kStr, value);
    return *this;
  }
  DGLRetValue& operator=(DGLByteArray value) {
    this->SwitchToClass(kBytes, std::string(value.data, value.size));
    return *this;
  }
  DGLRetValue& operator=(NDArray other) {
    this->Clear();
    type_code_ = kNDArrayContainer;
    value_.v_handle = other.data_;
    other.data_ = nullptr;
    return *this;
  }
  DGLRetValue& operator=(PackedFunc f) {
    this->SwitchToClass(kFuncHandle, f);
    return *this;
  }
  template <typename FType>
  DGLRetValue& operator=(const TypedPackedFunc<FType>& f) {
    return operator=(f.packed());
  }
  DGLRetValue& operator=(Module m) {
    this->SwitchToClass(kModuleHandle, m);
    return *this;
  }
  DGLRetValue& operator=(const DGLRetValue& other) {  // NOLINT(*0
    this->Assign(other);
    return *this;
  }
  DGLRetValue& operator=(const DGLArgValue& other) {
    this->Assign(other);
    return *this;
  }
  template <
      typename T, typename = typename std::enable_if<
                      extension_class_info<T>::code != 0>::type>
  DGLRetValue& operator=(const T& other) {
    this->SwitchToClass<T>(extension_class_info<T>::code, other);
    return *this;
  }
  /**
   * @brief Move the value back to front-end via C API.
   *  This marks the current container as null.
   *  The managed resources is moved to front-end and
   *  the front end should take charge in managing them.
   *
   * @param ret_value The return value.
   * @param ret_type_code The return type code.
   */
  void MoveToCHost(DGLValue* ret_value, int* ret_type_code) {
    // cannot move str; need specially handle.
    CHECK(type_code_ != kStr && type_code_ != kBytes);
    *ret_value = value_;
    *ret_type_code = type_code_;
    type_code_ = kNull;
  }
  /** @return The value field, if the data is POD */
  const DGLValue& value() const {
    CHECK(
        type_code_ != kObjectHandle && type_code_ != kFuncHandle &&
        type_code_ != kModuleHandle && type_code_ != kStr)
        << "DGLRetValue.value can only be used for POD data";
    return value_;
  }
  // ObjectRef related extenstions: in dgl/packed_func_ext.h
  template <
      typename T,
      typename = typename std::enable_if<std::is_class<T>::value>::type>
  inline operator T() const;
  template <typename TObjectRef>
  inline TObjectRef AsObjectRef() const;
  inline DGLRetValue& operator=(const ObjectRef& other);
  inline DGLRetValue& operator=(const std::shared_ptr<Object>& other);

 private:
  template <typename T>
  void Assign(const T& other) {
    switch (other.type_code()) {
      case kStr: {
        SwitchToClass<std::string>(kStr, other);
        break;
      }
      case kBytes: {
        SwitchToClass<std::string>(kBytes, other);
        break;
      }
      case kFuncHandle: {
        SwitchToClass<PackedFunc>(kFuncHandle, other);
        break;
      }
      case kModuleHandle: {
        SwitchToClass<Module>(kModuleHandle, other);
        break;
      }
      case kNDArrayContainer: {
        *this = other.operator NDArray();
        break;
      }
      case kObjectHandle: {
        SwitchToClass<std::shared_ptr<Object> >(
            kObjectHandle, *other.template ptr<std::shared_ptr<Object> >());
        break;
      }
      default: {
        if (other.type_code() < kExtBegin) {
          SwitchToPOD(other.type_code());
          value_ = other.value_;
        } else {
#if DGL_RUNTIME_HEADER_ONLY
          LOG(FATAL) << "Header only mode do not support ext type";
#else
          this->Clear();
          type_code_ = other.type_code();
          value_.v_handle = (*(ExtTypeVTable::Get(other.type_code())->clone))(
              other.value().v_handle);
#endif
        }
        break;
      }
    }
  }
  // get the internal container.
  void SwitchToPOD(int type_code) {
    if (type_code_ != type_code) {
      this->Clear();
      type_code_ = type_code;
    }
  }
  template <typename T>
  void SwitchToClass(int type_code, T v) {
    if (type_code_ != type_code) {
      this->Clear();
      type_code_ = type_code;
      value_.v_handle = new T(v);
    } else {
      *static_cast<T*>(value_.v_handle) = v;
    }
  }
  void Clear() {
    if (type_code_ == kNull) return;
    switch (type_code_) {
      case kStr:
      case kBytes:
        delete ptr<std::string>();
        break;
      case kFuncHandle:
        delete ptr<PackedFunc>();
        break;
      case kModuleHandle:
        delete ptr<Module>();
        break;
      case kObjectHandle:
        delete ptr<std::shared_ptr<Object> >();
        break;
      case kNDArrayContainer: {
        static_cast<NDArray::Container*>(value_.v_handle)->DecRef();
        break;
      }
    }
    if (type_code_ > kExtBegin) {
#if DGL_RUNTIME_HEADER_ONLY
      LOG(FATAL) << "Header only mode do not support ext type";
#else
      (*(ExtTypeVTable::Get(type_code_)->destroy))(value_.v_handle);
#endif
    }
    type_code_ = kNull;
  }
};

// implementation details
inline DGLArgValue DGLArgs::operator[](int i) const {
  CHECK_LT(i, num_args) << "not enough argument passed, " << num_args
                        << " passed"
                        << " but request arg[" << i << "].";
  return DGLArgValue(values[i], type_codes[i]);
}

inline int DGLArgs::size() const { return num_args; }

inline void PackedFunc::CallPacked(DGLArgs args, DGLRetValue* rv) const {
  body_(args, rv);
}

inline PackedFunc::FType PackedFunc::body() const { return body_; }

// internal namespace
namespace detail {

template <bool stop, std::size_t I, typename F>
struct for_each_dispatcher {
  template <typename T, typename... Args>
  static void run(const F& f, T&& value, Args&&... args) {  // NOLINT(*)
    f(I, std::forward<T>(value));
    for_each_dispatcher<sizeof...(Args) == 0, (I + 1), F>::run(
        f, std::forward<Args>(args)...);
  }
};

template <std::size_t I, typename F>
struct for_each_dispatcher<true, I, F> {
  static void run(const F& f) {}  // NOLINT(*)
};

template <typename F, typename... Args>
inline void for_each(const F& f, Args&&... args) {  // NOLINT(*)
  for_each_dispatcher<sizeof...(Args) == 0, 0, F>::run(
      f, std::forward<Args>(args)...);
}
}  // namespace detail

/* @brief argument settter to PackedFunc */
class DGLArgsSetter {
 public:
  DGLArgsSetter(DGLValue* values, int* type_codes)
      : values_(values), type_codes_(type_codes) {}
  // setters for POD types
  template <
      typename T,
      typename = typename std::enable_if<std::is_integral<T>::value>::type>
  void operator()(size_t i, T value) const {
    values_[i].v_int64 = static_cast<int64_t>(value);
    type_codes_[i] = kDGLInt;
  }
  void operator()(size_t i, uint64_t value) const {
    values_[i].v_int64 = static_cast<int64_t>(value);
    CHECK_LE(value, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    type_codes_[i] = kDGLInt;
  }
  void operator()(size_t i, double value) const {
    values_[i].v_float64 = value;
    type_codes_[i] = kDGLFloat;
  }
  void operator()(size_t i, std::nullptr_t value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kNull;
  }
  void operator()(size_t i, const DGLArgValue& value) const {
    values_[i] = value.value_;
    type_codes_[i] = value.type_code_;
  }
  void operator()(size_t i, void* value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kHandle;
  }
  void operator()(size_t i, DGLArray* value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kArrayHandle;
  }
  void operator()(size_t i, DGLContext value) const {
    values_[i].v_ctx = value;
    type_codes_[i] = kDGLContext;
  }
  void operator()(size_t i, DGLDataType value) const {
    values_[i].v_type = value;
    type_codes_[i] = kDGLDataType;
  }
  void operator()(size_t i, const char* value) const {
    values_[i].v_str = value;
    type_codes_[i] = kStr;
  }
  // setters for container type
  // They must be reference(instead of const ref)
  // to make sure they are alive in the tuple(instead of getting converted)
  void operator()(size_t i, const std::string& value) const {  // NOLINT(*)
    values_[i].v_str = value.c_str();
    type_codes_[i] = kStr;
  }
  void operator()(size_t i, const DGLByteArray& value) const {  // NOLINT(*)
    values_[i].v_handle = const_cast<DGLByteArray*>(&value);
    type_codes_[i] = kBytes;
  }
  void operator()(size_t i, const PackedFunc& value) const {  // NOLINT(*)
    values_[i].v_handle = const_cast<PackedFunc*>(&value);
    type_codes_[i] = kFuncHandle;
  }
  template <typename FType>
  void operator()(
      size_t i, const TypedPackedFunc<FType>& value) const {  // NOLINT(*)
    operator()(i, value.packed());
  }
  void operator()(size_t i, const Module& value) const {  // NOLINT(*)
    values_[i].v_handle = const_cast<Module*>(&value);
    type_codes_[i] = kModuleHandle;
  }
  void operator()(size_t i, const NDArray& value) const {  // NOLINT(*)
    values_[i].v_handle = value.data_;
    type_codes_[i] = kNDArrayContainer;
  }
  void operator()(size_t i, const DGLRetValue& value) const {  // NOLINT(*)
    if (value.type_code() == kStr) {
      values_[i].v_str = value.ptr<std::string>()->c_str();
      type_codes_[i] = kStr;
    } else {
      CHECK_NE(value.type_code(), kBytes) << "not handled.";
      values_[i] = value.value_;
      type_codes_[i] = value.type_code();
    }
  }
  // extension
  template <
      typename T, typename = typename std::enable_if<
                      extension_class_info<T>::code != 0>::type>
  inline void operator()(size_t i, const T& value) const;
  // ObjectRef related extenstions: in dgl/packed_func_ext.h
  inline void operator()(size_t i, const ObjectRef& other) const;  // NOLINT(*)

 private:
  /** @brief The values fields */
  DGLValue* values_;
  /** @brief The type code fields */
  int* type_codes_;
};

template <typename... Args>
inline DGLRetValue PackedFunc::operator()(Args&&... args) const {
  const int kNumArgs = sizeof...(Args);
  const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
  DGLValue values[kArraySize];
  int type_codes[kArraySize];
  detail::for_each(
      DGLArgsSetter(values, type_codes), std::forward<Args>(args)...);
  DGLRetValue rv;
  body_(DGLArgs(values, type_codes, kNumArgs), &rv);
  return rv;
}

namespace detail {
template <typename R, int nleft, int index, typename F>
struct unpack_call_dispatcher {
  template <typename... Args>
  static void run(
      const F& f, const DGLArgs& args_pack, DGLRetValue* rv,
      Args&&... unpacked_args) {
    unpack_call_dispatcher<R, nleft - 1, index + 1, F>::run(
        f, args_pack, rv, std::forward<Args>(unpacked_args)...,
        args_pack[index]);
  }
};

template <typename R, int index, typename F>
struct unpack_call_dispatcher<R, 0, index, F> {
  template <typename... Args>
  static void run(
      const F& f, const DGLArgs& args_pack, DGLRetValue* rv,
      Args&&... unpacked_args) {
    *rv = R(f(std::forward<Args>(unpacked_args)...));
  }
};

template <int index, typename F>
struct unpack_call_dispatcher<void, 0, index, F> {
  template <typename... Args>
  static void run(
      const F& f, const DGLArgs& args_pack, DGLRetValue* rv,
      Args&&... unpacked_args) {
    f(std::forward<Args>(unpacked_args)...);
  }
};

template <typename R, int nargs, typename F>
inline void unpack_call(const F& f, const DGLArgs& args, DGLRetValue* rv) {
  unpack_call_dispatcher<R, nargs, 0, F>::run(f, args, rv);
}

template <typename R, typename... Args>
inline R call_packed(const PackedFunc& pf, Args&&... args) {
  return R(pf(std::forward<Args>(args)...));
}

template <typename R>
struct typed_packed_call_dispatcher {
  template <typename... Args>
  static inline R run(const PackedFunc& pf, Args&&... args) {
    return pf(std::forward<Args>(args)...);
  }
};

template <>
struct typed_packed_call_dispatcher<void> {
  template <typename... Args>
  static inline void run(const PackedFunc& pf, Args&&... args) {
    pf(std::forward<Args>(args)...);
  }
};
}  // namespace detail

template <typename R, typename... Args>
TypedPackedFunc<R(Args...)>::TypedPackedFunc(PackedFunc packed)
    : packed_(packed) {}

template <typename R, typename... Args>
template <typename FType>
inline void TypedPackedFunc<R(Args...)>::AssignTypedLambda(FType flambda) {
  packed_ = PackedFunc([flambda](const DGLArgs& args, DGLRetValue* rv) {
    detail::unpack_call<R, sizeof...(Args)>(flambda, args, rv);
  });
}

template <typename R, typename... Args>
inline R TypedPackedFunc<R(Args...)>::operator()(Args... args) const {
  return detail::typed_packed_call_dispatcher<R>::run(
      packed_, std::forward<Args>(args)...);
}

// extension and node type handling
namespace detail {
template <typename T, typename TSrc, bool is_ext>
struct DGLValueCast {
  static T Apply(const TSrc* self) { return self->template AsObjectRef<T>(); }
};

template <typename T, typename TSrc>
struct DGLValueCast<T, TSrc, true> {
  static T Apply(const TSrc* self) { return self->template AsExtension<T>(); }
};
}  // namespace detail

template <typename T, typename>
inline DGLArgValue::operator T() const {
  return detail::DGLValueCast<
      T, DGLArgValue, extension_class_info<T>::code != 0>::Apply(this);
}

template <typename T, typename>
inline DGLRetValue::operator T() const {
  return detail::DGLValueCast<
      T, DGLRetValue, extension_class_info<T>::code != 0>::Apply(this);
}

template <typename T, typename>
inline void DGLArgsSetter::operator()(size_t i, const T& value) const {
  static_assert(
      extension_class_info<T>::code != 0, "Need to have extesion code");
  type_codes_[i] = extension_class_info<T>::code;
  values_[i].v_handle = const_cast<T*>(&value);
}

// extension type handling
template <typename T>
struct ExtTypeInfo {
  static void destroy(void* handle) { delete static_cast<T*>(handle); }
  static void* clone(void* handle) { return new T(*static_cast<T*>(handle)); }
};

template <typename T>
inline ExtTypeVTable* ExtTypeVTable::Register_() {
  const int code = extension_class_info<T>::code;
  static_assert(
      code != 0,
      "require extension_class_info traits to be declared with non-zero code");
  ExtTypeVTable vt;
  vt.clone = ExtTypeInfo<T>::clone;
  vt.destroy = ExtTypeInfo<T>::destroy;
  return ExtTypeVTable::RegisterInternal(code, vt);
}

// Implement Module::GetFunction
// Put implementation in this file so we have seen the PackedFunc
inline PackedFunc Module::GetFunction(
    const std::string& name, bool query_imports) {
  PackedFunc pf = node_->GetFunction(name, node_);
  if (pf != nullptr) return pf;
  if (query_imports) {
    for (const Module& m : node_->imports_) {
      pf = m.node_->GetFunction(name, m.node_);
      if (pf != nullptr) return pf;
    }
  }
  return pf;
}
}  // namespace runtime
}  // namespace dgl
#endif  // DGL_RUNTIME_PACKED_FUNC_H_
