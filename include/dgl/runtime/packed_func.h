/*!
 *  Copyright (c) 2017 by Contributors
 * \file tvm/runtime/packed_func.h
 * \brief Type-erased function used across TVM API.
 */
#ifndef TVM_RUNTIME_PACKED_FUNC_H_
#define TVM_RUNTIME_PACKED_FUNC_H_

#include <dmlc/logging.h>
#include <functional>
#include <tuple>
#include <vector>
#include <string>
#include <limits>
#include <memory>
#include <type_traits>
#include "c_runtime_api.h"
#include "module.h"
#include "ndarray.h"

namespace HalideIR {
// Forward declare type for extensions
// The header works fine without depending on this.
struct Type;
struct Expr;
}

// Whether use TVM runtime in header only mode.
#ifndef TVM_RUNTIME_HEADER_ONLY
#define TVM_RUNTIME_HEADER_ONLY 0
#endif

namespace tvm {
// Forward declare NodeRef and Node for extensions.
// This header works fine without depend on NodeRef
// as long as it is not used.
class Node;
class NodeRef;

namespace runtime {
// forward declarations
class TVMArgs;
class TVMArgValue;
class TVMRetValue;
class TVMArgsSetter;

/*!
 * \brief Packed function is a type-erased function.
 *  The arguments are passed by packed format.
 *
 *  This is an useful unified interface to call generated functions,
 *  It is the unified function function type of TVM.
 *  It corresponds to TVMFunctionHandle in C runtime API.
 */
class PackedFunc {
 public:
  /*!
   * \brief The internal std::function
   * \param args The arguments to the function.
   * \param rv The return value.
   *
   * \code
   *   // Example code on how to implemented FType
   *   void MyPackedFunc(TVMArgs args, TVMRetValue* rv) {
   *     // automatically convert arguments to desired type.
   *     int a0 = args[0];
   *     float a1 = args[1];
   *     ...
   *     // automatically assign values to rv
   *     std::string my_return_value = "x";
   *     *rv = my_return_value;
   *   }
   * \endcode
   */
  using FType = std::function<void (TVMArgs args, TVMRetValue* rv)>;
  /*! \brief default constructor */
  PackedFunc() {}
  /*!
   * \brief constructing a packed function from a std::function.
   * \param body the internal container of packed function.
   */
  explicit PackedFunc(FType body) : body_(body) {}
  /*!
   * \brief Call packed function by directly passing in unpacked format.
   * \param args Arguments to be passed.
   * \tparam Args arguments to be passed.
   *
   * \code
   *   // Example code on how to call packed function
   *   void CallPacked(PackedFunc f) {
   *     // call like normal functions by pass in arguments
   *     // return value is automatically converted back
   *     int rvalue = f(1, 2.0);
   *   }
   * \endcode
   */
  template<typename... Args>
  inline TVMRetValue operator()(Args&& ...args) const;
  /*!
   * \brief Call the function in packed format.
   * \param args The arguments
   * \param rv The return value.
   */
  inline void CallPacked(TVMArgs args, TVMRetValue* rv) const;
  /*! \return the internal body function */
  inline FType body() const;
  /*! \return Whether the packed function is nullptr */
  bool operator==(std::nullptr_t null) const {
    return body_ == nullptr;
  }
  /*! \return Whether the packed function is not nullptr */
  bool operator!=(std::nullptr_t null) const {
    return body_ != nullptr;
  }

 private:
  /*! \brief internal container of packed function */
  FType body_;
};

/*!
 * \brief Please refer to \ref TypedPackedFuncAnchor "TypedPackedFunc<R(Args..)>"
 */
template<typename FType>
class TypedPackedFunc;

/*!
 * \anchor TypedPackedFuncAnchor
 * \brief A PackedFunc wrapper to provide typed function signature.
 * It is backed by a PackedFunc internally.
 *
 * TypedPackedFunc enables compile time type checking.
 * TypedPackedFunc works with the runtime system:
 * - It can be passed as an argument of PackedFunc.
 * - It can be assigned to TVMRetValue.
 * - It can be directly converted to a type-erased PackedFunc.
 *
 * Developers should prefer TypedPackedFunc over PackedFunc in C++ code
 * as it enables compile time checking.
 * We can construct a TypedPackedFunc from a lambda function
 * with the same signature.
 *
 * \code
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
 * \endcode
 * \tparam R The return value of the function.
 * \tparam Args The argument signature of the function.
 */
template<typename R, typename ...Args>
class TypedPackedFunc<R(Args...)> {
 public:
  /*! \brief short hand for this function type */
  using TSelf = TypedPackedFunc<R(Args...)>;
  /*! \brief default constructor */
  TypedPackedFunc() {}
  /*!
   * \brief construct by wrap a PackedFunc
   *
   * Example usage:
   * \code
   * PackedFunc packed([](TVMArgs args, TVMRetValue *rv) {
   *   int x = args[0];
   *   *rv = x + 1;
   *  });
   * // construct from packed function
   * TypedPackedFunc<int(int)> ftyped(packed);
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param packed The packed function
   */
  inline explicit TypedPackedFunc(PackedFunc packed);
  /*!
   * \brief construct from a lambda function with the same signature.
   *
   * Example usage:
   * \code
   * auto typed_lambda = [](int x)->int { return x + 1; }
   * // construct from packed function
   * TypedPackedFunc<int(int)> ftyped(typed_lambda);
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param typed_lambda typed lambda function.
   * \tparam FLambda the type of the lambda function.
   */
  template<typename FLambda,
           typename = typename std::enable_if<
             std::is_convertible<FLambda,
                                 std::function<R(Args...)>
                                 >::value>::type>
  explicit TypedPackedFunc(const FLambda& typed_lambda) {
    this->AssignTypedLambda(typed_lambda);
  }
  /*!
   * \brief copy assignment operator from typed lambda
   *
   * Example usage:
   * \code
   * // construct from packed function
   * TypedPackedFunc<int(int)> ftyped;
   * ftyped = [](int x) { return x + 1; }
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param typed_lambda typed lambda function.
   * \tparam FLambda the type of the lambda function.
   * \returns reference to self.
   */
  template<typename FLambda,
           typename = typename std::enable_if<
             std::is_convertible<FLambda,
                                 std::function<R(Args...)>
                                 >::value>::type>
  TSelf& operator=(FLambda typed_lambda) {  // NOLINT(*)
    this->AssignTypedLambda(typed_lambda);
    return *this;
  }
  /*!
   * \brief copy assignment operator from PackedFunc.
   * \param packed The packed function.
   * \returns reference to self.
   */
  TSelf& operator=(PackedFunc packed) {
    packed_ = packed;
    return *this;
  }
  /*!
   * \brief Invoke the operator.
   * \param args The arguments
   * \returns The return value.
   */
  inline R operator()(Args ...args) const;
  /*!
   * \brief convert to PackedFunc
   * \return the internal PackedFunc
   */
  operator PackedFunc() const {
    return packed();
  }
  /*!
   * \return reference the internal PackedFunc
   */
  const PackedFunc& packed() const {
    return packed_;
  }

 private:
  friend class TVMRetValue;
  /*! \brief The internal packed function */
  PackedFunc packed_;
  /*!
   * \brief Assign the packed field using a typed lambda function.
   *
   * \param flambda The lambda function.
   * \tparam FLambda The lambda function type.
   * \note We capture the lambda when possible for maximum efficiency.
   */
  template<typename FLambda>
  inline void AssignTypedLambda(FLambda flambda);
};

/*! \brief Arguments into TVM functions. */
class TVMArgs {
 public:
  const TVMValue* values;
  const int* type_codes;
  int num_args;
  /*!
   * \brief constructor
   * \param values The argument values
   * \param type_codes The argument type codes
   * \param num_args number of arguments.
   */
  TVMArgs(const TVMValue* values,
          const int* type_codes,
          int num_args)
      : values(values),
        type_codes(type_codes),
        num_args(num_args) { }
  /*! \return size of the arguments */
  inline int size() const;
  /*!
   * \brief Get i-th argument
   * \param i the index.
   * \return the ith argument.
   */
  inline TVMArgValue operator[](int i) const;
};

/*!
 * \brief Convert type code to its name
 * \param type_code The type code .
 * \return The name of type code.
 */
inline const char* TypeCode2Str(int type_code);

/*!
 * \brief convert a string to TVM type.
 * \param s The string to be converted.
 * \return The corresponding tvm type.
 */
inline TVMType String2TVMType(std::string s);

/*!
 * \brief convert a TVM type to string.
 * \param t The type to be converted.
 * \return The corresponding tvm type in string.
 */
inline std::string TVMType2String(TVMType t);

// macro to check type code.
#define TVM_CHECK_TYPE_CODE(CODE, T)                           \
  CHECK_EQ(CODE, T) << " expected "                            \
  << TypeCode2Str(T) << " but get " << TypeCode2Str(CODE)      \

/*!
 * \brief Type traits to mark if a class is tvm extension type.
 *
 * To enable extension type in C++ must be register () ed via marco.
 * TVM_REGISTER_EXT_TYPE(TypeName) after defining this with this traits.
 *
 * Extension class can be passed and returned via PackedFunc in all tvm runtime.
 * Internally extension class is stored as T*.
 *
 * \tparam T the typename
 */
template<typename T>
struct extension_class_info {
  static const int code = 0;
};

/*!
 * \brief Runtime function table about extension type.
 */
class ExtTypeVTable {
 public:
  /*! \brief function to be called to delete a handle */
  void (*destroy)(void* handle);
  /*! \brief function to be called when clone a handle */
  void* (*clone)(void* handle);
  /*!
   * \brief Register type
   * \tparam T The type to be register.
   * \return The registered vtable.
   */
  template <typename T>
  static inline ExtTypeVTable* Register_();
  /*!
   * \brief Get a vtable based on type code.
   * \param type_code The type code
   * \return The registered vtable.
   */
  TVM_DLL static ExtTypeVTable* Get(int type_code);

 private:
  // Internal registration function.
  TVM_DLL static ExtTypeVTable* RegisterInternal(int type_code, const ExtTypeVTable& vt);
};

/*!
 * \brief Internal base class to
 *  handle conversion to POD values.
 */
class TVMPODValue_ {
 public:
  operator double() const {
    // Allow automatic conversion from int to float
    // This avoids errors when user pass in int from
    // the frontend while the API expects a float.
    if (type_code_ == kDLInt) {
      return static_cast<double>(value_.v_int64);
    }
    TVM_CHECK_TYPE_CODE(type_code_, kDLFloat);
    return value_.v_float64;
  }
  operator int64_t() const {
    TVM_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64;
  }
  operator uint64_t() const {
    TVM_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64;
  }
  operator int() const {
    TVM_CHECK_TYPE_CODE(type_code_, kDLInt);
    CHECK_LE(value_.v_int64,
             std::numeric_limits<int>::max());
    return static_cast<int>(value_.v_int64);
  }
  operator bool() const {
    TVM_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64 != 0;
  }
  operator void*() const {
    if (type_code_ == kNull) return nullptr;
    if (type_code_ == kArrayHandle) return value_.v_handle;
    TVM_CHECK_TYPE_CODE(type_code_, kHandle);
    return value_.v_handle;
  }
  operator DLTensor*() const {
    if (type_code_ == kArrayHandle ||
        type_code_ == kNDArrayContainer) {
      return static_cast<DLTensor*>(value_.v_handle);
    } else {
      if (type_code_ == kNull) return nullptr;
      LOG(FATAL) << "Expected "
                 << "DLTensor* or NDArray but get "
                 << TypeCode2Str(type_code_);
      return nullptr;
    }
  }
  operator NDArray() const {
    if (type_code_ == kNull) return NDArray();
    TVM_CHECK_TYPE_CODE(type_code_, kNDArrayContainer);
    return NDArray(static_cast<NDArray::Container*>(value_.v_handle));
  }
  operator TVMContext() const {
    TVM_CHECK_TYPE_CODE(type_code_, kTVMContext);
    return value_.v_ctx;
  }
  template<typename TExtension>
  const TExtension& AsExtension() const {
    CHECK_LT(type_code_, kExtEnd);
    return static_cast<TExtension*>(value_.v_handle)[0];
  }
  int type_code() const {
    return type_code_;
  }
  /*!
   * \brief return handle as specific pointer type.
   * \tparam T the data type.
   * \return The pointer type.
   */
  template<typename T>
  T* ptr() const {
    return static_cast<T*>(value_.v_handle);
  }

 protected:
  friend class TVMArgsSetter;
  friend class TVMRetValue;
  TVMPODValue_() : type_code_(kNull) {}
  TVMPODValue_(TVMValue value, int type_code)
      : value_(value), type_code_(type_code) {}

  /*! \brief The value */
  TVMValue value_;
  /*! \brief the type code */
  int type_code_;
};

/*!
 * \brief A single argument value to PackedFunc.
 *  Containing both type_code and TVMValue
 *
 *  Provides utilities to do type cast into other types.
 */
class TVMArgValue : public TVMPODValue_ {
 public:
  /*! \brief default constructor */
  TVMArgValue() {}
  /*!
   * \brief constructor
   * \param value of the function
   * \param type_code The type code.
   */
  TVMArgValue(TVMValue value, int type_code)
      : TVMPODValue_(value, type_code) {
  }
  // reuse converter from parent
  using TVMPODValue_::operator double;
  using TVMPODValue_::operator int64_t;
  using TVMPODValue_::operator uint64_t;
  using TVMPODValue_::operator int;
  using TVMPODValue_::operator bool;
  using TVMPODValue_::operator void*;
  using TVMPODValue_::operator DLTensor*;
  using TVMPODValue_::operator NDArray;
  using TVMPODValue_::operator TVMContext;

  // conversion operator.
  operator std::string() const {
    if (type_code_ == kTVMType) {
      return TVMType2String(operator TVMType());
    } else if (type_code_ == kBytes) {
      TVMByteArray* arr = static_cast<TVMByteArray*>(value_.v_handle);
      return std::string(arr->data, arr->size);
    } else {
      TVM_CHECK_TYPE_CODE(type_code_, kStr);
      return std::string(value_.v_str);
    }
  }
  operator TVMType() const {
    if (type_code_ == kStr) {
      return String2TVMType(operator std::string());
    }
    TVM_CHECK_TYPE_CODE(type_code_, kTVMType);
    return value_.v_type;
  }
  operator PackedFunc() const {
    if (type_code_ == kNull) return PackedFunc();
    TVM_CHECK_TYPE_CODE(type_code_, kFuncHandle);
    return *ptr<PackedFunc>();
  }
  template<typename FType>
  operator TypedPackedFunc<FType>() const {
    return TypedPackedFunc<FType>(operator PackedFunc());
  }
  operator Module() const {
    TVM_CHECK_TYPE_CODE(type_code_, kModuleHandle);
    return *ptr<Module>();
  }
  const TVMValue& value() const {
    return value_;
  }
  // Deferred extension handler.
  template<typename TNodeRef>
  inline TNodeRef AsNodeRef() const;
  template<typename T,
           typename = typename std::enable_if<
             std::is_class<T>::value>::type>
  inline operator T() const;
  template<typename TNodeRef,
           typename = typename std::enable_if<
             std::is_class<TNodeRef>::value>::type>
  inline bool IsNodeType() const;
  inline operator HalideIR::Type() const;
  inline operator HalideIR::Expr() const;
  // get internal node ptr, if it is node
  inline std::shared_ptr<Node>& node_sptr();
};

/*!
 * \brief Return Value container,
 *  Unlike TVMArgValue, which only holds reference and do not delete
 *  the underlying container during destruction.
 *
 *  TVMRetValue holds value and will manage the underlying containers
 *  when it stores a complicated data type.
 */
class TVMRetValue : public TVMPODValue_ {
 public:
  /*! \brief default constructor */
  TVMRetValue() {}
  /*!
   * \brief move constructor from anoter return value.
   * \param other The other return value.
   */
  TVMRetValue(TVMRetValue&& other)
      : TVMPODValue_(other.value_, other.type_code_) {
    other.value_.v_handle = nullptr;
    other.type_code_ = kNull;
  }
  /*! \brief destructor */
  ~TVMRetValue() {
    this->Clear();
  }
  // reuse converter from parent
  using TVMPODValue_::operator double;
  using TVMPODValue_::operator int64_t;
  using TVMPODValue_::operator uint64_t;
  using TVMPODValue_::operator int;
  using TVMPODValue_::operator bool;
  using TVMPODValue_::operator void*;
  using TVMPODValue_::operator DLTensor*;
  using TVMPODValue_::operator TVMContext;
  using TVMPODValue_::operator NDArray;
  // Disable copy and assign from another value, but allow move.
  TVMRetValue(const TVMRetValue& other) {
    this->Assign(other);
  }
  // conversion operators
  operator std::string() const {
    if (type_code_ == kTVMType) {
      return TVMType2String(operator TVMType());
    } else if (type_code_ == kBytes) {
      return *ptr<std::string>();
    }
    TVM_CHECK_TYPE_CODE(type_code_, kStr);
    return *ptr<std::string>();
  }
  operator TVMType() const {
    if (type_code_ == kStr) {
      return String2TVMType(operator std::string());
    }
    TVM_CHECK_TYPE_CODE(type_code_, kTVMType);
    return value_.v_type;
  }
  operator PackedFunc() const {
    if (type_code_ == kNull) return PackedFunc();
    TVM_CHECK_TYPE_CODE(type_code_, kFuncHandle);
    return *ptr<PackedFunc>();
  }
  template<typename FType>
  operator TypedPackedFunc<FType>() const {
    return TypedPackedFunc<FType>(operator PackedFunc());
  }
  operator Module() const {
    TVM_CHECK_TYPE_CODE(type_code_, kModuleHandle);
    return *ptr<Module>();
  }
  // Assign operators
  TVMRetValue& operator=(TVMRetValue&& other) {
    this->Clear();
    value_ = other.value_;
    type_code_ = other.type_code_;
    other.type_code_ = kNull;
    return *this;
  }
  TVMRetValue& operator=(double value) {
    this->SwitchToPOD(kDLFloat);
    value_.v_float64 = value;
    return *this;
  }
  TVMRetValue& operator=(std::nullptr_t value) {
    this->SwitchToPOD(kNull);
    value_.v_handle = value;
    return *this;
  }
  TVMRetValue& operator=(void* value) {
    this->SwitchToPOD(kHandle);
    value_.v_handle = value;
    return *this;
  }
  TVMRetValue& operator=(int64_t value) {
    this->SwitchToPOD(kDLInt);
    value_.v_int64 = value;
    return *this;
  }
  TVMRetValue& operator=(int value) {
    this->SwitchToPOD(kDLInt);
    value_.v_int64 = value;
    return *this;
  }
  TVMRetValue& operator=(TVMType t) {
    this->SwitchToPOD(kTVMType);
    value_.v_type = t;
    return *this;
  }
  TVMRetValue& operator=(bool value) {
    this->SwitchToPOD(kDLInt);
    value_.v_int64 = value;
    return *this;
  }
  TVMRetValue& operator=(std::string value) {
    this->SwitchToClass(kStr, value);
    return *this;
  }
  TVMRetValue& operator=(TVMByteArray value) {
    this->SwitchToClass(kBytes, std::string(value.data, value.size));
    return *this;
  }
  TVMRetValue& operator=(NDArray other) {
    this->Clear();
    type_code_ = kNDArrayContainer;
    value_.v_handle = other.data_;
    other.data_ = nullptr;
    return *this;
  }
  TVMRetValue& operator=(PackedFunc f) {
    this->SwitchToClass(kFuncHandle, f);
    return *this;
  }
  template<typename FType>
  TVMRetValue& operator=(const TypedPackedFunc<FType>& f) {
    return operator=(f.packed());
  }
  TVMRetValue& operator=(Module m) {
    this->SwitchToClass(kModuleHandle, m);
    return *this;
  }
  TVMRetValue& operator=(const TVMRetValue& other) {  // NOLINT(*0
    this->Assign(other);
    return *this;
  }
  TVMRetValue& operator=(const TVMArgValue& other) {
    this->Assign(other);
    return *this;
  }
  template<typename T,
           typename = typename std::enable_if<
             extension_class_info<T>::code != 0>::type>
  TVMRetValue& operator=(const T& other) {
    this->SwitchToClass<T>(
        extension_class_info<T>::code, other);
    return *this;
  }
  /*!
   * \brief Move the value back to front-end via C API.
   *  This marks the current container as null.
   *  The managed resources is moved to front-end and
   *  the front end should take charge in managing them.
   *
   * \param ret_value The return value.
   * \param ret_type_code The return type code.
   */
  void MoveToCHost(TVMValue* ret_value,
                   int* ret_type_code) {
    // cannot move str; need specially handle.
    CHECK(type_code_ != kStr && type_code_ != kBytes);
    *ret_value = value_;
    *ret_type_code = type_code_;
    type_code_ = kNull;
  }
  /*! \return The value field, if the data is POD */
  const TVMValue& value() const {
    CHECK(type_code_ != kNodeHandle &&
          type_code_ != kFuncHandle &&
          type_code_ != kModuleHandle &&
          type_code_ != kStr) << "TVMRetValue.value can only be used for POD data";
    return value_;
  }
  // NodeRef related extenstions: in tvm/packed_func_ext.h
  template<typename T,
           typename = typename std::enable_if<
             std::is_class<T>::value>::type>
  inline operator T() const;
  template<typename TNodeRef>
  inline TNodeRef AsNodeRef() const;
  inline TVMRetValue& operator=(const NodeRef& other);
  inline TVMRetValue& operator=(const std::shared_ptr<Node>& other);
  // type related
  inline operator HalideIR::Type() const;
  inline TVMRetValue& operator=(const HalideIR::Type& other);

 private:
  template<typename T>
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
      case kNodeHandle: {
        SwitchToClass<std::shared_ptr<Node> >(
            kNodeHandle, *other.template ptr<std::shared_ptr<Node> >());
        break;
      }
      default: {
        if (other.type_code() < kExtBegin) {
          SwitchToPOD(other.type_code());
          value_ = other.value_;
        } else {
#if TVM_RUNTIME_HEADER_ONLY
          LOG(FATAL) << "Header only mode do not support ext type";
#else
          this->Clear();
          type_code_ = other.type_code();
          value_.v_handle =
              (*(ExtTypeVTable::Get(other.type_code())->clone))(
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
  template<typename T>
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
      case kStr: delete ptr<std::string>(); break;
      case kFuncHandle: delete ptr<PackedFunc>(); break;
      case kModuleHandle: delete ptr<Module>(); break;
      case kNodeHandle: delete ptr<std::shared_ptr<Node> >(); break;
      case kNDArrayContainer: {
        static_cast<NDArray::Container*>(value_.v_handle)->DecRef();
        break;
      }
    }
    if (type_code_ > kExtBegin) {
#if TVM_RUNTIME_HEADER_ONLY
          LOG(FATAL) << "Header only mode do not support ext type";
#else
      (*(ExtTypeVTable::Get(type_code_)->destroy))(value_.v_handle);
#endif
    }
    type_code_ = kNull;
  }
};

// implementation details
inline const char* TypeCode2Str(int type_code) {
  switch (type_code) {
    case kDLInt: return "int";
    case kDLUInt: return "uint";
    case kDLFloat: return "float";
    case kStr: return "str";
    case kBytes: return "bytes";
    case kHandle: return "handle";
    case kNull: return "NULL";
    case kNodeHandle: return "NodeHandle";
    case kArrayHandle: return "ArrayHandle";
    case kTVMType: return "TVMType";
    case kTVMContext: return "TVMContext";
    case kFuncHandle: return "FunctionHandle";
    case kModuleHandle: return "ModuleHandle";
    case kNDArrayContainer: return "NDArrayContainer";
    default: LOG(FATAL) << "unknown type_code="
                        << static_cast<int>(type_code); return "";
  }
}

#ifndef _LIBCPP_SGX_NO_IOSTREAMS
inline std::ostream& operator<<(std::ostream& os, TVMType t) {  // NOLINT(*)
  os << TypeCode2Str(t.code);
  if (t.code == kHandle) return os;
  os << static_cast<int>(t.bits);
  if (t.lanes != 1) {
    os << 'x' << static_cast<int>(t.lanes);
  }
  return os;
}
#endif

inline std::string TVMType2String(TVMType t) {
#ifndef _LIBCPP_SGX_NO_IOSTREAMS
  std::ostringstream os;
  os << t;
  return os.str();
#else
  std::string repr = "";
  repr += TypeCode2Str(t.code);
  if (t.code == kHandle) return repr;
  repr += std::to_string(static_cast<int>(t.bits));
  if (t.lanes != 1) {
    repr += "x" + std::to_string(static_cast<int>(t.lanes));
  }
  return repr;
#endif
}

inline TVMType String2TVMType(std::string s) {
  TVMType t;
  t.bits = 32; t.lanes = 1;
  const char* scan;
  if (s.substr(0, 3) == "int") {
    t.code = kDLInt;  scan = s.c_str() + 3;
  } else if (s.substr(0, 4) == "uint") {
    t.code = kDLUInt; scan = s.c_str() + 4;
  } else if (s.substr(0, 5) == "float") {
    t.code = kDLFloat; scan = s.c_str() + 5;
  } else if (s.substr(0, 6) == "handle") {
    t.code = kHandle;
    t.bits = 64;  // handle uses 64 bit by default.
    scan = s.c_str() + 6;
  } else {
    scan = s.c_str();
    LOG(FATAL) << "unknown type " << s;
  }
  char* xdelim;  // emulate sscanf("%ux%u", bits, lanes)
  uint8_t bits = static_cast<uint8_t>(strtoul(scan, &xdelim, 10));
  if (bits != 0) t.bits = bits;
  if (*xdelim == 'x') {
    t.lanes = static_cast<uint16_t>(strtoul(xdelim + 1, nullptr, 10));
  }
  return t;
}

inline TVMArgValue TVMArgs::operator[](int i) const {
  CHECK_LT(i, num_args)
      << "not enough argument passed, "
      << num_args << " passed"
      << " but request arg[" << i << "].";
  return TVMArgValue(values[i], type_codes[i]);
}

inline int TVMArgs::size() const {
  return num_args;
}

inline void PackedFunc::CallPacked(TVMArgs args, TVMRetValue* rv) const {
  body_(args, rv);
}

inline PackedFunc::FType PackedFunc::body() const {
  return body_;
}



// internal namespace
namespace detail {

template<bool stop, std::size_t I, typename F>
struct for_each_dispatcher {
  template<typename T, typename ...Args>
  static void run(const F& f, T&& value, Args&&... args) {  // NOLINT(*)
    f(I, std::forward<T>(value));
    for_each_dispatcher<sizeof...(Args) == 0, (I+1), F>
        ::run(f, std::forward<Args>(args)...);
  }
};

template<std::size_t I, typename F>
struct for_each_dispatcher<true, I, F>  {
  static void run(const F& f) {}  // NOLINT(*)
};

template<typename F, typename ...Args>
inline void for_each(const F& f, Args&&... args) {  // NOLINT(*)
  for_each_dispatcher<sizeof...(Args) == 0, 0, F>
      ::run(f, std::forward<Args>(args)...);
}
}  // namespace detail

/* \brief argument settter to PackedFunc */
class TVMArgsSetter {
 public:
  TVMArgsSetter(TVMValue* values, int* type_codes)
      : values_(values), type_codes_(type_codes) {}
  // setters for POD types
  template<typename T,
           typename = typename std::enable_if<
             std::is_integral<T>::value>::type>
  void operator()(size_t i, T value) const {
    values_[i].v_int64 = static_cast<int64_t>(value);
    type_codes_[i] = kDLInt;
  }
  void operator()(size_t i, uint64_t value) const {
    values_[i].v_int64 = static_cast<int64_t>(value);
    CHECK_LE(value,
             static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    type_codes_[i] = kDLInt;
  }
  void operator()(size_t i, double value) const {
    values_[i].v_float64 = value;
    type_codes_[i] = kDLFloat;
  }
  void operator()(size_t i, std::nullptr_t value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kNull;
  }
  void operator()(size_t i, const TVMArgValue& value) const {
    values_[i] = value.value_;
    type_codes_[i] = value.type_code_;
  }
  void operator()(size_t i, void* value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kHandle;
  }
  void operator()(size_t i, DLTensor* value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kArrayHandle;
  }
  void operator()(size_t i, TVMContext value) const {
    values_[i].v_ctx = value;
    type_codes_[i] = kTVMContext;
  }
  void operator()(size_t i, TVMType value) const {
    values_[i].v_type = value;
    type_codes_[i] = kTVMType;
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
  void operator()(size_t i, const TVMByteArray& value) const {  // NOLINT(*)
    values_[i].v_handle = const_cast<TVMByteArray*>(&value);
    type_codes_[i] = kBytes;
  }
  void operator()(size_t i, const PackedFunc& value) const {  // NOLINT(*)
    values_[i].v_handle = const_cast<PackedFunc*>(&value);
    type_codes_[i] = kFuncHandle;
  }
  template<typename FType>
  void operator()(size_t i, const TypedPackedFunc<FType>& value) const {  // NOLINT(*)
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
  void operator()(size_t i, const TVMRetValue& value) const {  // NOLINT(*)
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
  template<typename T,
           typename = typename std::enable_if<
             extension_class_info<T>::code != 0>::type>
  inline void operator()(size_t i, const T& value) const;
  // NodeRef related extenstions: in tvm/packed_func_ext.h
  inline void operator()(size_t i, const NodeRef& other) const;  // NOLINT(*)
  inline void operator()(size_t i, const HalideIR::Type& t) const;

 private:
  /*! \brief The values fields */
  TVMValue* values_;
  /*! \brief The type code fields */
  int* type_codes_;
};

template<typename... Args>
inline TVMRetValue PackedFunc::operator()(Args&& ...args) const {
  const int kNumArgs = sizeof...(Args);
  const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
  TVMValue values[kArraySize];
  int type_codes[kArraySize];
  detail::for_each(TVMArgsSetter(values, type_codes),
                   std::forward<Args>(args)...);
  TVMRetValue rv;
  body_(TVMArgs(values, type_codes, kNumArgs), &rv);
  return rv;
}

namespace detail {
template<typename R, int nleft, int index, typename F>
struct unpack_call_dispatcher {
  template<typename ...Args>
  static void run(const F& f,
                  const TVMArgs& args_pack,
                  TVMRetValue* rv,
                  Args&&... unpacked_args) {
    unpack_call_dispatcher<R, nleft - 1, index + 1, F>
        ::run(f, args_pack, rv,
              std::forward<Args>(unpacked_args)...,
              args_pack[index]);
  }
};

template<typename R, int index, typename F>
struct unpack_call_dispatcher<R, 0, index, F> {
  template<typename ...Args>
  static void run(const F& f,
                  const TVMArgs& args_pack,
                  TVMRetValue* rv,
                  Args&&... unpacked_args) {
    *rv = R(f(std::forward<Args>(unpacked_args)...));
  }
};

template<int index, typename F>
struct unpack_call_dispatcher<void, 0, index, F> {
  template<typename ...Args>
  static void run(const F& f,
                  const TVMArgs& args_pack,
                  TVMRetValue* rv,
                  Args&&... unpacked_args) {
    f(std::forward<Args>(unpacked_args)...);
  }
};

template<typename R, int nargs, typename F>
inline void unpack_call(const F& f, const TVMArgs& args, TVMRetValue* rv) {
  unpack_call_dispatcher<R, nargs, 0, F>::run(f, args, rv);
}

template<typename R, typename ...Args>
inline R call_packed(const PackedFunc& pf, Args&& ...args) {
  return R(pf(std::forward<Args>(args)...));
}

template<typename R>
struct typed_packed_call_dispatcher {
  template<typename ...Args>
  static inline R run(const PackedFunc& pf, Args&& ...args) {
    return pf(std::forward<Args>(args)...);
  }
};

template<>
struct typed_packed_call_dispatcher<void> {
  template<typename ...Args>
  static inline void run(const PackedFunc& pf, Args&& ...args) {
    pf(std::forward<Args>(args)...);
  }
};
}  // namespace detail

template<typename R, typename ...Args>
TypedPackedFunc<R(Args...)>::TypedPackedFunc(PackedFunc packed)
  : packed_(packed) {}

template<typename R, typename ...Args>
template<typename FType>
inline void TypedPackedFunc<R(Args...)>::AssignTypedLambda(FType flambda) {
  packed_ = PackedFunc([flambda](const TVMArgs& args, TVMRetValue* rv) {
      detail::unpack_call<R, sizeof...(Args)>(flambda, args, rv);
    });
}

template<typename R, typename ...Args>
inline R TypedPackedFunc<R(Args...)>::operator()(Args... args) const {
  return detail::typed_packed_call_dispatcher<R>
      ::run(packed_, std::forward<Args>(args)...);
}

// extension and node type handling
namespace detail {
template<typename T, typename TSrc, bool is_ext>
struct TVMValueCast {
  static T Apply(const TSrc* self) {
    return self->template AsNodeRef<T>();
  }
};

template<typename T, typename TSrc>
struct TVMValueCast<T, TSrc, true> {
  static T Apply(const TSrc* self) {
    return self->template AsExtension<T>();
  }
};
}  // namespace detail

template<typename T, typename>
inline TVMArgValue::operator T() const {
  return detail::
      TVMValueCast<T, TVMArgValue, extension_class_info<T>::code != 0>
      ::Apply(this);
}

template<typename T, typename>
inline TVMRetValue::operator T() const {
  return detail::
      TVMValueCast<T, TVMRetValue, extension_class_info<T>::code != 0>
      ::Apply(this);
}

template<typename T, typename>
inline void TVMArgsSetter::operator()(size_t i, const T& value) const {
  static_assert(extension_class_info<T>::code != 0,
                "Need to have extesion code");
  type_codes_[i] = extension_class_info<T>::code;
  values_[i].v_handle = const_cast<T*>(&value);
}

// extension type handling
template<typename T>
struct ExtTypeInfo {
  static void destroy(void* handle) {
    delete static_cast<T*>(handle);
  }
  static void* clone(void* handle) {
    return new T(*static_cast<T*>(handle));
  }
};

template<typename T>
inline ExtTypeVTable* ExtTypeVTable::Register_() {
  const int code = extension_class_info<T>::code;
  static_assert(code != 0,
                "require extension_class_info traits to be declared with non-zero code");
  ExtTypeVTable vt;
  vt.clone = ExtTypeInfo<T>::clone;
  vt.destroy = ExtTypeInfo<T>::destroy;
  return ExtTypeVTable::RegisterInternal(code, vt);
}

// Implement Module::GetFunction
// Put implementation in this file so we have seen the PackedFunc
inline PackedFunc Module::GetFunction(const std::string& name, bool query_imports) {
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
}  // namespace tvm
#endif  // TVM_RUNTIME_PACKED_FUNC_H_
