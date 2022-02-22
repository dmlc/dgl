/*!
 *  Copyright (c) 2017 by Contributors
 * \file dgl/runtime/ndarray.h
 * \brief Abstract device memory management API
 */
#ifndef DGL_RUNTIME_NDARRAY_H_
#define DGL_RUNTIME_NDARRAY_H_


#include <atomic>
#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "c_runtime_api.h"
#include "dlpack/dlpack.h"
#include "serializer.h"
#include "shared_mem.h"

// forward declaration
inline std::ostream& operator << (std::ostream& os, DGLType t);

namespace dgl {

/*!
 * \brief Type traits that converts a C type to a DLDataType.
 *
 * Usage:
 * DLDataTypeTraits<int>::dtype == dtype
 */
template<typename T>
struct DLDataTypeTraits {
  static constexpr DLDataType dtype{0, 0, 0};   // dummy
};
#define GEN_DLDATATYPETRAITS_FOR(T, code, bits) \
  template<> \
  struct DLDataTypeTraits<T> { \
    static constexpr DLDataType dtype{code, bits, 1}; \
  }
GEN_DLDATATYPETRAITS_FOR(int8_t, kDLInt, 8);
GEN_DLDATATYPETRAITS_FOR(int16_t, kDLInt, 16);
GEN_DLDATATYPETRAITS_FOR(int32_t, kDLInt, 32);
GEN_DLDATATYPETRAITS_FOR(int64_t, kDLInt, 64);
// XXX(BarclayII) most DL frameworks do not support unsigned int and long arrays, so I'm just
// converting uints to signed DTypes.
GEN_DLDATATYPETRAITS_FOR(uint32_t, kDLInt, 32);
GEN_DLDATATYPETRAITS_FOR(uint64_t, kDLInt, 64);
GEN_DLDATATYPETRAITS_FOR(float, kDLFloat, 32);
GEN_DLDATATYPETRAITS_FOR(double, kDLFloat, 64);
#undef GEN_DLDATATYPETRAITS_FOR

namespace runtime {

/*!
 * \brief Managed NDArray.
 *  The array is backed by reference counted blocks.
 */
class NDArray {
 public:
  // internal container type
  struct Container;
  /*! \brief default constructor */
  NDArray() {}
  /*!
   * \brief cosntruct a NDArray that refers to data
   * \param data The data this NDArray refers to
   */
  explicit inline NDArray(Container* data);
  /*!
   * \brief copy constructor
   * \param other The value to be copied
   */
  inline NDArray(const NDArray& other);  // NOLINT(*)
  /*!
   * \brief move constructor
   * \param other The value to be moved
   */
  NDArray(NDArray&& other) // NOLINT(*)
      : data_(other.data_) {
    other.data_ = nullptr;
  }
  /*! \brief destructor */
  ~NDArray() {
    this->reset();
  }
  /*!
   * \brief Swap this array with another NDArray
   * \param other The other NDArray
   */
  void swap(NDArray& other) {  // NOLINT(*)
    std::swap(data_, other.data_);
  }
  /*!
   * \brief copy assignmemt
   * \param other The value to be assigned.
   * \return reference to self.
   */
  NDArray& operator=(const NDArray& other) {  // NOLINT(*)
    // copy-and-swap idiom
    NDArray(other).swap(*this);  // NOLINT(*)
    return *this;
  }
  /*!
   * \brief move assignmemt
   * \param other The value to be assigned.
   * \return reference to self.
   */
  NDArray& operator=(NDArray&& other) {  // NOLINT(*)
    // copy-and-swap idiom
    NDArray(std::move(other)).swap(*this); // NOLINT(*)
    return *this;
  }
  /*! \return If NDArray is defined */
  bool defined() const {
    return data_ != nullptr;
  }
  /*! \return If both NDArray reference the same container */
  bool same_as(const NDArray& other) const {
    return data_ == other.data_;
  }
  /*! \brief reset the content of NDArray to be nullptr */
  inline void reset();
  /*!
   * \return the reference counter
   * \note this number is approximate in multi-threaded setting.
   */
  inline int use_count() const;
  /*! \return Pointer to content of DLTensor */
  inline const DLTensor* operator->() const;
  /*! \return True if the ndarray is contiguous. */
  bool IsContiguous() const;
  /*! \return the data pointer with type. */
  template <typename T>
  inline T* Ptr() const {
    if (!defined())
      return nullptr;
    else
      return static_cast<T*>(operator->()->data);
  }
  /*!
   * \brief Copy data content from another array.
   * \param other The source array to be copied from.
   * \param stream The stream to perform the copy on if it involves a GPU
   *        context, otherwise this parameter is ignored.
   * \note The copy may happen asynchrously if it involves a GPU context.
   *       DGLSynchronize is necessary.
   */
  inline void CopyFrom(DLTensor* other,
      DGLStreamHandle stream = nullptr);
  inline void CopyFrom(const NDArray& other,
      DGLStreamHandle stream = nullptr);
  /*!
   * \brief Copy data content into another array.
   * \param other The source array to be copied from.
   * \note The copy may happen asynchrously if it involves a GPU context.
   *       DGLSynchronize is necessary.
   */
  inline void CopyTo(DLTensor *other,
                     const DGLStreamHandle &stream = nullptr) const;
  inline void CopyTo(const NDArray &other,
                     const DGLStreamHandle &stream = nullptr) const;
  /*!
   * \brief Copy the data to another context.
   * \param ctx The target context.
   * \return The array under another context.
   */
  inline NDArray CopyTo(const DLContext &ctx,
                        const DGLStreamHandle &stream = nullptr) const;
  /*!
   * \brief Return a new array with a copy of the content.
   */
  inline NDArray Clone(const DGLStreamHandle &stream = nullptr) const;
  /*!
   * \brief In-place method to pin the current array by calling PinData
   *        on the underlying DLTensor.
   * \note This is an in-place method. Behavior depends on the current context,
   *       kDLCPU: will be pinned;
   *       IsPinned: directly return;
   *       kDLGPU: invalid, will throw an error.
   */
  inline void PinMemory_();
  /*!
   * \brief In-place method to unpin the current array by calling UnpinData
   *        on the underlying DLTensor.
   * \note This is an in-place method. Behavior depends on the current context,
   *       IsPinned: will be unpinned;
   *       others: directly return.
   */
  inline void UnpinMemory_();
  /*!
   * \brief Check if the array is pinned.
   */
  inline bool IsPinned() const;
  /*!
   * \brief Load NDArray from stream
   * \param stream The input data stream
   * \return Whether load is successful
   */
  bool Load(dmlc::Stream* stream);
  /*!
   * \brief Save NDArray to stream
   * \param stream The output data stream
   */
  void Save(dmlc::Stream* stream) const;
  /*!
   * \brief Create a NDArray that shares the data memory with the current one.
   * \param shape The shape of the new array.
   * \param dtype The data type of the new array.
   * \param offset The offset (in bytes) of the starting pointer.
   * \note The memory size of new array must be smaller than the current one.
   */
  DGL_DLL NDArray CreateView(
      std::vector<int64_t> shape, DLDataType dtype, int64_t offset = 0);
  /*!
   * \brief Create a reference view of NDArray that
   *  represents as DLManagedTensor.
   * \return A DLManagedTensor
   */
  DGL_DLL DLManagedTensor* ToDLPack() const;
  /*!
   * \brief Create an empty NDArray.
   * \param shape The shape of the new array.
   * \param dtype The data type of the new array.
   * \param ctx The context of the Array.
   * \return The created Array
   */
  DGL_DLL static NDArray Empty(std::vector<int64_t> shape,
                               DLDataType dtype,
                               DLContext ctx);
  /*!
   * \brief Create an empty NDArray with shared memory.
   * \param name The name of shared memory.
   * \param shape The shape of the new array.
   * \param dtype The data type of the new array.
   * \param ctx The context of the Array.
   * \param is_create whether to create shared memory.
   * \return The created Array
   */
  DGL_DLL static NDArray EmptyShared(const std::string &name,
                                     std::vector<int64_t> shape,
                                     DLDataType dtype,
                                     DLContext ctx,
                                     bool is_create);
  /*!
   * \brief Get the size of the array in the number of bytes.
   */
  size_t GetSize() const;

  /*!
   * \brief Get the number of elements in this array.
   */
  int64_t NumElements() const;

  /*!
   * \brief Create a NDArray backed by a dlpack tensor.
   *
   * This allows us to create a NDArray using the memory
   * allocated by an external deep learning framework
   * that is DLPack compatible.
   *
   * The memory is retained until the NDArray went out of scope.
   * \param tensor The DLPack tensor to copy from.
   * \return The created NDArray view.
   */
  DGL_DLL static NDArray FromDLPack(DLManagedTensor* tensor);

  /*!
   * \brief Create a NDArray by copying from std::vector.
   * \tparam T Type of vector data.  Determines the dtype of returned array.
   */
  template<typename T>
  DGL_DLL static NDArray FromVector(
      const std::vector<T>& vec, DLContext ctx = DLContext{kDLCPU, 0});

  /*!
   * \brief Create a std::vector from a 1D NDArray.
   * \tparam T Type of vector data.
   * \note Type casting is NOT performed.  The caller has to make sure that the vector
   *       type matches the dtype of NDArray.
   */
  template<typename T>
  std::vector<T> ToVector() const;

  std::shared_ptr<SharedMemory> GetSharedMem() const;

  /*!
   * \brief Function to copy data from one array to another.
   * \param from The source array.
   * \param to The target array.
   * \param stream The stream used in copy.
   */
  DGL_DLL static void CopyFromTo(
      DLTensor* from, DLTensor* to, DGLStreamHandle stream = nullptr);

  /*!
   * \brief Function to pin the data of a DLTensor.
   * \param tensor The array to be pinned.
   * \note Data of the given array will be pinned inplace.
   *       Behavior depends on the current context,
   *       kDLCPU: will be pinned;
   *       IsPinned: directly return;
   *       kDLGPU: invalid, will throw an error.
   */
  DGL_DLL static void PinData(DLTensor* tensor);

  /*!
   * \brief Function to unpin the data of a DLTensor.
   * \param tensor The array to be unpinned.
   * \note Data of the given array will be unpinned inplace.
   *       Behavior depends on the current context,
   *       IsPinned: will be unpinned;
   *       others: directly return.
   */
  DGL_DLL static void UnpinData(DLTensor* tensor);

  /*!
   * \brief Function check if the data of a DLTensor is pinned.
   * \param tensor The array to be checked.
   * \return true if pinned.
   */
  DGL_DLL static bool IsDataPinned(DLTensor* tensor);

  // internal namespace
  struct Internal;
 private:
  /*! \brief Internal Data content */
  Container* data_{nullptr};
  // enable internal functions
  friend struct Internal;
  friend class DGLRetValue;
  friend class DGLArgsSetter;
};

/*!
 * \brief Save a DLTensor to stream
 * \param strm The outpu stream
 * \param tensor The tensor to be saved.
 */
inline bool SaveDLTensor(dmlc::Stream* strm, const DLTensor* tensor);


/*!
 * \brief Reference counted Container object used to back NDArray.
 *
 *  This object is DLTensor compatible:
 *    the pointer to the NDArrayContainer can be directly
 *    interpreted as a DLTensor*
 *
 * \note: do not use this function directly, use NDArray.
 */
struct NDArray::Container {
 public:
  // NOTE: the first part of this structure is the same as
  // DLManagedTensor, note that, however, the deleter
  // is only called when the reference counter goes to 0
  /*!
   * \brief The corresponding dl_tensor field.
   * \note it is important that the first field is DLTensor
   *  So that this data structure is DLTensor compatible.
   *  The head ptr of this struct can be viewed as DLTensor*.
   */
  DLTensor dl_tensor;

  std::shared_ptr<SharedMemory> mem;
  /*!
   * \brief addtional context, reserved for recycling
   * \note We can attach additional content here
   *  which the current container depend on
   *  (e.g. reference to original memory when creating views).
   */
  void* manager_ctx{nullptr};
  /*!
   * \brief Customized deleter
   *
   * \note The customized deleter is helpful to enable
   *  different ways of memory allocator that are not
   *  currently defined by the system.
   */
  void (*deleter)(Container* self) = nullptr;
  /*! \brief default constructor */
  Container() {
    dl_tensor.data = nullptr;
    dl_tensor.ndim = 0;
    dl_tensor.shape = nullptr;
    dl_tensor.strides = nullptr;
    dl_tensor.byte_offset = 0;
  }
  /*! \brief developer function, increases reference counter */
  void IncRef() {
    ref_counter_.fetch_add(1, std::memory_order_relaxed);
  }
  /*! \brief developer function, decrease reference counter */
  void DecRef() {
    if (ref_counter_.fetch_sub(1, std::memory_order_release) == 1) {
      std::atomic_thread_fence(std::memory_order_acquire);
      if (this->deleter != nullptr) {
        (*this->deleter)(this);
      }
    }
  }

 private:
  friend class NDArray;
  friend class RPCWrappedFunc;
  /*!
   * \brief The shape container,
   *  can be used for shape data.
   */
  std::vector<int64_t> shape_;
  /*!
   * \brief The stride container,
   *  can be used for stride data.
   */
  std::vector<int64_t> stride_;
  /*! \brief The internal array object */
  std::atomic<int> ref_counter_{0};
};

// implementations of inline functions
// the usages of functions are documented in place.
inline NDArray::NDArray(Container* data)
  : data_(data) {
  if (data_)
    data_->IncRef();
}

inline NDArray::NDArray(const NDArray& other)
  : data_(other.data_) {
  if (data_)
    data_->IncRef();
}

inline void NDArray::reset() {
  if (data_) {
    data_->DecRef();
    data_ = nullptr;
  }
}

inline void NDArray::CopyFrom(DLTensor* other,
                              DGLStreamHandle stream) {
  CHECK(data_ != nullptr);
  CopyFromTo(other, &(data_->dl_tensor), stream);
}

inline void NDArray::CopyFrom(const NDArray& other,
                              DGLStreamHandle stream) {
  CHECK(data_ != nullptr);
  CHECK(other.data_ != nullptr);
  CopyFromTo(&(other.data_->dl_tensor), &(data_->dl_tensor), stream);
}

inline void NDArray::CopyTo(DLTensor *other,
                            const DGLStreamHandle &stream) const {
  CHECK(data_ != nullptr);
  CopyFromTo(&(data_->dl_tensor), other, stream);
}

inline void NDArray::CopyTo(const NDArray &other,
                            const DGLStreamHandle &stream) const {
  CHECK(data_ != nullptr);
  CHECK(other.data_ != nullptr);
  CopyFromTo(&(data_->dl_tensor), &(other.data_->dl_tensor), stream);
}

inline NDArray NDArray::CopyTo(const DLContext &ctx,
                               const DGLStreamHandle &stream) const {
  CHECK(data_ != nullptr);
  const DLTensor* dptr = operator->();
  NDArray ret = Empty(std::vector<int64_t>(dptr->shape, dptr->shape + dptr->ndim),
                      dptr->dtype, ctx);
  this->CopyTo(ret, stream);
  return ret;
}

inline NDArray NDArray::Clone(const DGLStreamHandle &stream) const {
  CHECK(data_ != nullptr);
  const DLTensor* dptr = operator->();
  return this->CopyTo(dptr->ctx, stream);
}

inline void NDArray::PinMemory_() {
  CHECK(data_ != nullptr);
  PinData(&(data_->dl_tensor));
}

inline void NDArray::UnpinMemory_() {
  CHECK(data_ != nullptr);
  UnpinData(&(data_->dl_tensor));
}

inline bool NDArray::IsPinned() const {
  CHECK(data_ != nullptr);
  return IsDataPinned(&(data_->dl_tensor));
}

inline int NDArray::use_count() const {
  if (data_ == nullptr) return 0;
  return data_->ref_counter_.load(std::memory_order_relaxed);
}

inline const DLTensor* NDArray::operator->() const {
  return &(data_->dl_tensor);
}

/*! \brief Magic number for NDArray file */
constexpr uint64_t kDGLNDArrayMagic = 0xDD5E40F096B4A13F;

inline bool SaveDLTensor(dmlc::Stream* strm,
                         DLTensor* tensor) {
  uint64_t header = kDGLNDArrayMagic, reserved = 0;
  strm->Write(header);
  strm->Write(reserved);
  // Always save data as CPU context
  //
  // Parameters that get serialized should be in CPU by default.
  // So even the array's context is GPU, it will be stored as CPU array.
  // This is used to prevent case when another user loads the parameters
  // back on machine that do not have GPU or related context.
  //
  // We can always do array.CopyTo(target_ctx) to get a corresponding
  // array in the target context.
  DLContext cpu_ctx;
  cpu_ctx.device_type = kDLCPU;
  cpu_ctx.device_id = 0;
  strm->Write(cpu_ctx);
  strm->Write(tensor->ndim);
  strm->Write(tensor->dtype);
  int ndim = tensor->ndim;
  strm->WriteArray(tensor->shape, ndim);
  int type_bytes = tensor->dtype.bits / 8;
  int64_t num_elems = 1;
  for (int i = 0; i < ndim; ++i) {
    num_elems *= tensor->shape[i];
  }
  int64_t data_byte_size = type_bytes * num_elems;
  strm->Write(data_byte_size);

  if (DMLC_IO_NO_ENDIAN_SWAP &&
      tensor->ctx.device_type == kDLCPU &&
      tensor->strides == nullptr &&
      tensor->byte_offset == 0) {
    // quick path
    strm->Write(tensor->data, data_byte_size);
  } else {
    std::vector<uint8_t> bytes(data_byte_size);
    CHECK_EQ(DGLArrayCopyToBytes(
        tensor, dmlc::BeginPtr(bytes), data_byte_size), 0)
        << DGLGetLastError();
    if (!DMLC_IO_NO_ENDIAN_SWAP) {
      dmlc::ByteSwap(dmlc::BeginPtr(bytes), type_bytes, num_elems);
    }
    strm->Write(dmlc::BeginPtr(bytes), data_byte_size);
  }
  return true;
}

/*!
 * \brief Convert type code to its name
 * \param type_code The type code .
 * \return The name of type code.
 */
inline const char* TypeCode2Str(int type_code) {
  switch (type_code) {
    case kDLInt: return "int";
    case kDLUInt: return "uint";
    case kDLFloat: return "float";
    case kStr: return "str";
    case kBytes: return "bytes";
    case kHandle: return "handle";
    case kNull: return "NULL";
    case kObjectHandle: return "ObjectHandle";
    case kArrayHandle: return "ArrayHandle";
    case kDGLType: return "DGLType";
    case kDGLContext: return "DGLContext";
    case kFuncHandle: return "FunctionHandle";
    case kModuleHandle: return "ModuleHandle";
    case kNDArrayContainer: return "NDArrayContainer";
    default: LOG(FATAL) << "unknown type_code="
                        << static_cast<int>(type_code); return "";
  }
}

/*!
 * \brief Convert device type code to its name
 * \param device_type The device type code.
 * \return The name of the device.
 */
inline const char* DeviceTypeCode2Str(DLDeviceType device_type) {
  switch (device_type) {
    case kDLCPU: return "cpu";
    case kDLGPU: return "cuda";
    case kDLCPUPinned: return "cpu_pinned";
    case kDLOpenCL: return "opencl";
    case kDLVulkan: return "vulkan";
    case kDLMetal: return "metal";
    case kDLVPI: return "vpi";
    case kDLROCM: return "rocm";
    default: LOG(FATAL) << "Unknown device type code="
                        << static_cast<int>(device_type); return "";
  }
}

/*!
 * \brief convert a string to DGL type.
 * \param s The string to be converted.
 * \return The corresponding dgl type.
 */
inline DGLType String2DGLType(std::string s) {
  DGLType t;
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

/*!
 * \brief convert a DGL type to string.
 * \param t The type to be converted.
 * \return The corresponding dgl type in string.
 */
inline std::string DGLType2String(DGLType t) {
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

// macro to check type code.
#define DGL_CHECK_TYPE_CODE(CODE, T)                           \
  CHECK_EQ(CODE, T) << " expected "                            \
  << TypeCode2Str(T) << " but get " << TypeCode2Str(CODE)      \

}  // namespace runtime
}  // namespace dgl

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, dgl::runtime::NDArray, true);
}  // namespace dmlc

///////////////// Operator overloading for NDArray /////////////////
dgl::runtime::NDArray operator + (const dgl::runtime::NDArray& a1,
                                  const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator - (const dgl::runtime::NDArray& a1,
                                  const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator * (const dgl::runtime::NDArray& a1,
                                  const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator / (const dgl::runtime::NDArray& a1,
                                  const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator % (const dgl::runtime::NDArray& a1,
                                  const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator + (const dgl::runtime::NDArray& a1, int64_t rhs);
dgl::runtime::NDArray operator - (const dgl::runtime::NDArray& a1, int64_t rhs);
dgl::runtime::NDArray operator * (const dgl::runtime::NDArray& a1, int64_t rhs);
dgl::runtime::NDArray operator / (const dgl::runtime::NDArray& a1, int64_t rhs);
dgl::runtime::NDArray operator % (const dgl::runtime::NDArray& a1, int64_t rhs);
dgl::runtime::NDArray operator + (int64_t lhs, const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator - (int64_t lhs, const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator * (int64_t lhs, const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator / (int64_t lhs, const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator % (int64_t lhs, const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator - (const dgl::runtime::NDArray& array);

dgl::runtime::NDArray operator > (const dgl::runtime::NDArray& a1,
                                  const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator < (const dgl::runtime::NDArray& a1,
                                  const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator >= (const dgl::runtime::NDArray& a1,
                                   const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator <= (const dgl::runtime::NDArray& a1,
                                   const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator == (const dgl::runtime::NDArray& a1,
                                   const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator != (const dgl::runtime::NDArray& a1,
                                   const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator > (const dgl::runtime::NDArray& a1, int64_t rhs);
dgl::runtime::NDArray operator < (const dgl::runtime::NDArray& a1, int64_t rhs);
dgl::runtime::NDArray operator >= (const dgl::runtime::NDArray& a1, int64_t rhs);
dgl::runtime::NDArray operator <= (const dgl::runtime::NDArray& a1, int64_t rhs);
dgl::runtime::NDArray operator == (const dgl::runtime::NDArray& a1, int64_t rhs);
dgl::runtime::NDArray operator != (const dgl::runtime::NDArray& a1, int64_t rhs);
dgl::runtime::NDArray operator > (int64_t lhs, const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator < (int64_t lhs, const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator >= (int64_t lhs, const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator <= (int64_t lhs, const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator == (int64_t lhs, const dgl::runtime::NDArray& a2);
dgl::runtime::NDArray operator != (int64_t lhs, const dgl::runtime::NDArray& a2);

std::ostream& operator << (std::ostream& os, dgl::runtime::NDArray array);

///////////////// Operator overloading for DLDataType /////////////////

/*! \brief Check whether two data types are the same.*/
inline bool operator == (const DLDataType& ty1, const DLDataType& ty2) {
  return ty1.code == ty2.code && ty1.bits == ty2.bits && ty1.lanes == ty2.lanes;
}

/*! \brief Check whether two data types are different.*/
inline bool operator != (const DLDataType& ty1, const DLDataType& ty2) {
  return !(ty1 == ty2);
}

#ifndef _LIBCPP_SGX_NO_IOSTREAMS
inline std::ostream& operator << (std::ostream& os, DGLType t) {
  os << dgl::runtime::TypeCode2Str(t.code);
  if (t.code == kHandle) return os;
  os << static_cast<int>(t.bits);
  if (t.lanes != 1) {
    os << 'x' << static_cast<int>(t.lanes);
  }
  return os;
}
#endif

///////////////// Operator overloading for DLContext /////////////////

/*! \brief Check whether two device contexts are the same.*/
inline bool operator == (const DLContext& ctx1, const DLContext& ctx2) {
  return ctx1.device_type == ctx2.device_type && ctx1.device_id == ctx2.device_id;
}

/*! \brief Check whether two device contexts are different.*/
inline bool operator != (const DLContext& ctx1, const DLContext& ctx2) {
  return !(ctx1 == ctx2);
}

#ifndef _LIBCPP_SGX_NO_IOSTREAMS
inline std::ostream& operator << (std::ostream& os, const DLContext& ctx) {
  return os << dgl::runtime::DeviceTypeCode2Str(ctx.device_type) << ":" << ctx.device_id;
}
#endif

#endif  // DGL_RUNTIME_NDARRAY_H_
