/**
 *  Copyright (c) 2016-2022 by Contributors
 * @file dgl/runtime/c_runtime_api.h
 * @brief DGL runtime library.
 *
 * This runtime is adapted from TVM project (commit: 2ce5277)
 */
#ifndef DGL_RUNTIME_C_RUNTIME_API_H_
#define DGL_RUNTIME_C_RUNTIME_API_H_

// Macros to do weak linking
#ifdef _MSC_VER
#define DGL_WEAK __declspec(selectany)
#else
#define DGL_WEAK __attribute__((weak))
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define DGL_DLL EMSCRIPTEN_KEEPALIVE
#endif

#ifndef DGL_DLL
#ifdef _WIN32
#ifdef DGL_EXPORTS
#define DGL_DLL __declspec(dllexport)
#else
#define DGL_DLL __declspec(dllimport)
#endif
#else
#define DGL_DLL
#endif
#endif

// DGL version
#define DGL_VERSION "2.5"

#ifdef __cplusplus
extern "C" {
#endif
#include <stddef.h>
#include <stdint.h>

/** @brief type of array index. */
typedef int64_t dgl_index_t;

/**
 * @brief The device type in DGLContext.
 */
#ifdef __cplusplus
typedef enum : int32_t {
#else
typedef enum {
#endif
  /** @brief CPU device */
  kDGLCPU = 1,
  /** @brief CUDA GPU device */
  kDGLCUDA = 2,
  // add more devices once supported
} DGLDeviceType;

/**
 * @brief The object type code is used in DGL FFI to indicate the types of
 *        objects passed between C and Python.
 */
typedef enum {
  kObjectInt = 0U,
  kObjectUInt = 1U,
  kObjectFloat = 2U,
  kHandle = 3U,
  kNull = 4U,
  kDGLDataType = 5U,
  kDGLContext = 6U,
  kArrayHandle = 7U,
  kObjectHandle = 8U,
  kModuleHandle = 9U,
  kFuncHandle = 10U,
  kStr = 11U,
  kBytes = 12U,
  kNDArrayContainer = 13U,
  // Extension codes for other frameworks to integrate DGL PackedFunc.
  // To make sure each framework's id do not conflict, use first and
  // last sections to mark ranges.
  // Open an issue at the repo if you need a section of code.
  kExtBegin = 15U,
  kNNVMFirst = 16U,
  kNNVMLast = 20U,
  // The following section of code is used for non-reserved types.
  kExtReserveEnd = 64U,
  kExtEnd = 128U
} DGLObjectTypeCode;

/**
 * @brief The type code options DGLDataType.
 */
typedef enum {
  /** @brief signed integer */
  kDGLInt = 0U,
  /** @brief unsigned integer */
  kDGLUInt = 1U,
  /** @brief IEEE floating point */
  kDGLFloat = 2U,
  /** @brief bfloat16 */
  kDGLBfloat = 4U,
  // add more data types if we are going to support them
} DGLDataTypeCode;

/**
 * @brief The data type the tensor can hold. The data type is assumed to follow
 * the native endian-ness. An explicit error message should be raised when
 * attempting to export an array with non-native endianness
 *
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes=1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes=4
 *   - int8: type_code = 0, bits = 8, lanes=1
 */
typedef struct {
  /**
   * @brief Type code of base types.
   * We keep it uint8_t instead of DGLDataTypeCode for minimal memory
   * footprint, but the value should be one of DGLDataTypeCode enum values.
   * */
  uint8_t code;
  /**
   * @brief Number of bits, common choices are 8, 16, 32.
   */
  uint8_t bits;
  /** @brief Number of lanes in the type, used for vector types. */
  uint16_t lanes;
} DGLDataType;

/**
 * @brief The Device information, abstract away common device types.
 */
typedef struct {
  /** @brief The device type used in the device. */
  DGLDeviceType device_type;
  /**
   * @brief The device index.
   * For vanilla CPU memory, pinned memory, or managed memory, this is set to 0.
   */
  int32_t device_id;
} DGLContext;

/**
 * @brief The tensor array stucture to DGL API.
 * The structure is heavily inspired by DLTensor from DLPack.
 */
typedef struct {
  /**
   * @brief The data pointer points to the allocated data.
   *
   * Depending on the device context, it can be a CPU pointer, or a CUDA
   * device pointer or  acl_mem handle in OpenCL.
   * This pointer is always aligned to 256 bytes as in CUDA. Use the
   * `byte_offset` field to mark the beginning of the actual data (if the
   * address is not 256 byte aligned).
   *
   * Note that as of Nov 2021, multiply libraries (CuPy, PyTorch, TensorFlow,
   * TVM, perhaps others) do not adhere to this 256 byte alignment requirement
   * on CPU/CUDA/ROCm, and always use `byte_offset=0`.  This is likely to be
   * fixed in the future; at the moment it is recommended
   * to not rely on the data pointer being correctly aligned.
   *
   * For a DGLArray, the size of memory required to store the contents of
   * data can be calculated as follows:
   *
   * @code{.c}
   * static inline size_t GetDataSize(const DGLArray* t) {
   *   size_t size = 1;
   *   for (int32_t i = 0; i < t->ndim; ++i) {
   *     size *= t->shape[i];
   *   }
   *   size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
   *   return size;
   * }
   * @endcode
   */
  void* data;
  /** @brief The device of the tensor */
  DGLContext ctx;
  /** @brief Number of dimensions */
  int32_t ndim;
  /** @brief The data type of the pointer*/
  DGLDataType dtype;
  /** @brief The shape of the tensor */
  int64_t* shape;
  /**
   * @brief strides of the tensor (in number of elements, not bytes)
   *  can be NULL, indicating tensor is compact and row-majored.
   */
  int64_t* strides;
  /** @brief The offset in bytes to the beginning pointer to data */
  uint64_t byte_offset;
} DGLArray;

/** @brief the array handle */
typedef DGLArray* DGLArrayHandle;

/**
 * @brief Union type of values
 *  being passed through API and function calls.
 */
typedef union {
  int64_t v_int64;
  double v_float64;
  void* v_handle;
  const char* v_str;
  DGLDataType v_type;
  DGLContext v_ctx;
} DGLValue;

/**
 * @brief Byte array type used to pass in byte array
 *  When kBytes is used as data type.
 */
typedef struct {
  const char* data;
  size_t size;
} DGLByteArray;

/** @brief Handle to DGL runtime modules. */
typedef void* DGLModuleHandle;
/** @brief Handle to packed function handle. */
typedef void* DGLFunctionHandle;
/** @brief Handle to hold return value. */
typedef void* DGLRetValueHandle;
/**
 * @brief The stream that is specific to device
 * can be NULL, which indicates the default one.
 */
typedef void* DGLStreamHandle;

/**
 * @brief Used for implementing C API function.
 *  Set last error message before return.
 * @param msg The error message to be set.
 */
DGL_DLL void DGLAPISetLastError(const char* msg);

/**
 * @brief return str message of the last error
 *  all function in this file will return 0 when success
 *  and -1 when an error occured,
 *  DGLGetLastError can be called to retrieve the error
 *
 *  this function is threadsafe and can be called by different thread
 *
 * @return error info
 */
DGL_DLL const char* DGLGetLastError(void);
/**
 * @brief Load module from file.
 * @param file_name The file name to load the module from.
 * @param format The format of the module.
 * @param out The result module
 *
 * @return 0 when success, -1 when failure happens
 * @note The resulting module do not contain import relation.
 *  It can be reconstructed by DGLModImport.
 */
DGL_DLL int DGLModLoadFromFile(
    const char* file_name, const char* format, DGLModuleHandle* out);

/**
 * @brief Add dep to mod's dependency.
 *  This allows functions in this module to use modules.
 *
 * @param mod The module handle.
 * @param dep The dependent module to be imported.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLModImport(DGLModuleHandle mod, DGLModuleHandle dep);

/**
 * @brief Get function from the module.
 * @param mod The module handle.
 * @param func_name The name of the function.
 * @param query_imports Whether to query imported modules
 * @param out The result function, can be NULL if it is not available.
 * @return 0 when no error is thrown, -1 when failure happens
 */
DGL_DLL int DGLModGetFunction(
    DGLModuleHandle mod, const char* func_name, int query_imports,
    DGLFunctionHandle* out);

/**
 * @brief Free front-end extension type resource.
 * @param handle The extension handle.
 * @param type_code The type of of the extension type.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLExtTypeFree(void* handle, int type_code);

/**
 * @brief Free the Module
 * @param mod The module to be freed.
 *
 * @note This may not free up the module's resources.
 *  If there is active DGLFunctionHandle uses the module
 *  Or if this module is imported by another active module.
 *
 *  The all functions remains valid until DGLFuncFree is called.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLModFree(DGLModuleHandle mod);

/**
 * @brief Free the function when it is no longer needed.
 * @param func The function handle
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLFuncFree(DGLFunctionHandle func);

/**
 * @brief Call a Packed DGL Function.
 *
 * @param func node handle of the function.
 * @param arg_values The arguments
 * @param type_codes The type codes of the arguments
 * @param num_args Number of arguments.
 *
 * @param ret_val The return value.
 * @param ret_type_code the type code of return value.
 *
 * @return 0 when success, -1 when failure happens
 * @note DGL calls always exchanges with type bits=64, lanes=1
 *
 * @note API calls always exchanges with type bits=64, lanes=1
 *   If API call returns container handles (e.g. FunctionHandle)
 *   these handles should be managed by the front-end.
 *   The front-end need to call free function (e.g. DGLFuncFree)
 *   to free these handles.
 */
DGL_DLL int DGLFuncCall(
    DGLFunctionHandle func, DGLValue* arg_values, int* type_codes, int num_args,
    DGLValue* ret_val, int* ret_type_code);

/**
 * @brief Set the return value of DGLPackedCFunc.
 *
 *  This function is called by DGLPackedCFunc to set the return value.
 *  When this function is not called, the function returns null by default.
 *
 * @param ret The return value handle, pass by ret in DGLPackedCFunc
 * @param value The value to be returned.
 * @param type_code The type of the value to be returned.
 * @param num_ret Number of return values, for now only 1 is supported.
 */
DGL_DLL int DGLCFuncSetReturn(
    DGLRetValueHandle ret, DGLValue* value, int* type_code, int num_ret);

/**
 * @brief Inplace translate callback argument value to return value.
 *  This is only needed for non-POD arguments.
 *
 * @param value The value to be translated.
 * @param code The type code to be translated.
 * @note This function will do a shallow copy when necessary.
 *
 * @return 0 when success, -1 when failure happens.
 */
DGL_DLL int DGLCbArgToReturn(DGLValue* value, int code);

/**
 * @brief C type of packed function.
 *
 * @param args The arguments
 * @param type_codes The type codes of the arguments
 * @param num_args Number of arguments.
 * @param ret The return value handle.
 * @param resource_handle The handle additional resouce handle from fron-end.
 * @return 0 if success, -1 if failure happens, set error via
 *         DGLAPISetLastError.
 * @sa DGLCFuncSetReturn
 */
typedef int (*DGLPackedCFunc)(
    DGLValue* args, int* type_codes, int num_args, DGLRetValueHandle ret,
    void* resource_handle);

/**
 * @brief C callback to free the resource handle in C packed function.
 * @param resource_handle The handle additional resouce handle from fron-end.
 */
typedef void (*DGLPackedCFuncFinalizer)(void* resource_handle);

/**
 * @brief Signature for extension function declarer.
 *
 *  DGL call this function to get the extension functions
 *  The declarer will call register_func to register function and their name.
 *
 * @param register_func_handle The register function
 * @return 0 if success, -1 if failure happens
 */
typedef int (*DGLExtensionFuncDeclarer)(DGLFunctionHandle register_func_handle);

/**
 * @brief Wrap a DGLPackedCFunc to become a FunctionHandle.
 *
 * The resource_handle will be managed by DGL API, until the function is no
 * longer used.
 *
 * @param func The packed C function.
 * @param resource_handle The resource handle from front-end, can be NULL.
 * @param fin The finalizer on resource handle when the FunctionHandle get
 *        freed, can be NULL.
 * @param out the result function handle.
 * @return 0 when success, -1 when failure happens.
 */
DGL_DLL int DGLFuncCreateFromCFunc(
    DGLPackedCFunc func, void* resource_handle, DGLPackedCFuncFinalizer fin,
    DGLFunctionHandle* out);

/**
 * @brief Register the function to runtime's global table.
 *
 * The registered function then can be pulled by the backend by the name.
 *
 * @param name The name of the function.
 * @param f The function to be registered.
 * @param override Whether allow override already registered function.
 */
DGL_DLL int DGLFuncRegisterGlobal(
    const char* name, DGLFunctionHandle f, int override);

/**
 * @brief Get a global function.
 *
 * @param name The name of the function.
 * @param out the result function pointer, NULL if it does not exist.
 *
 * @note The function handle of global function is managed by DGL runtime,
 *  So DGLFuncFree is should not be called when it get deleted.
 */
DGL_DLL int DGLFuncGetGlobal(const char* name, DGLFunctionHandle* out);

/**
 * @brief List all the globally registered function name
 * @param out_size The number of functions
 * @param out_array The array of function names.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLFuncListGlobalNames(int* out_size, const char*** out_array);

// Array related apis for quick proptyping
/**
 * @brief Allocate a nd-array's memory,
 *  including space of shape, of given spec.
 *
 * @param shape The shape of the array, the data content will be copied to out
 * @param ndim The number of dimension of the array.
 * @param dtype_code The type code of the dtype
 * @param dtype_bits The number of bits of dtype
 * @param dtype_lanes The number of lanes in the dtype.
 * @param device_type The device type of context
 * @param device_id The device id of context.
 * @param out The output handle.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLArrayAlloc(
    const dgl_index_t* shape, int ndim, int dtype_code, int dtype_bits,
    int dtype_lanes, int device_type, int device_id, DGLArrayHandle* out);

/**
 * @brief Allocate a nd-array's with shared memory,
 *  including space of shape, of given spec.
 *
 * @param the name of the shared memory
 * @param shape The shape of the array, the data content will be copied to out
 * @param ndim The number of dimension of the array.
 * @param dtype_code The type code of the dtype
 * @param dtype_bits The number of bits of dtype
 * @param dtype_lanes The number of lanes in the dtype.
 * @param is_create whether the shared memory is created
 * @param out The output handle.
 * @return 0 when success, -1 when failure happens
 */
int DGLArrayAllocSharedMem(
    const char* mem_name, const dgl_index_t* shape, int ndim, int dtype_code,
    int dtype_bits, int dtype_lanes, bool is_create, DGLArrayHandle* out);

/**
 * @brief Free the DGL Array.
 * @param handle The array handle to be freed.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLArrayFree(DGLArrayHandle handle);

/**
 * @brief Copy array data from CPU byte array.
 * @param handle The array handle.
 * @param data the data pointer
 * @param nbytes The number of bytes to copy.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLArrayCopyFromBytes(
    DGLArrayHandle handle, void* data, size_t nbytes);

/**
 * @brief Copy array data to CPU byte array.
 * @param handle The array handle.
 * @param data the data pointer
 * @param nbytes The number of bytes to copy.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLArrayCopyToBytes(
    DGLArrayHandle handle, void* data, size_t nbytes);

/**
 * @brief Copy the array, both from and to must be valid during the copy.
 * @param from The array to be copied from.
 * @param to The target space.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLArrayCopyFromTo(DGLArrayHandle from, DGLArrayHandle to);

/**
 * @brief Create a new runtime stream.
 *
 * @param device_type The device type of context
 * @param device_id The device id of context
 * @param out The new stream handle
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLStreamCreate(
    int device_type, int device_id, DGLStreamHandle* out);

/**
 * @brief Free a created stream handle.
 *
 * @param device_type The device type of context
 * @param device_id The device id of context
 * @param stream The stream to be freed
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLStreamFree(
    int device_type, int device_id, DGLStreamHandle stream);

/**
 * @brief Set the runtime stream of current thread to be stream.
 *  The subsequent calls to the same device_type
 *  will use the setted stream handle.
 *  The specific type of stream is runtime device dependent.
 *
 * @param device_type The device type of context
 * @param device_id The device id of context.
 * @param handle The stream handle.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLSetStream(
    int device_type, int device_id, DGLStreamHandle handle);

/**
 * @brief Get the runtime stream of current thread.
 *
 * @param device_type The device type of context
 * @param device_id The device id of context.
 * @param handle The stream handle.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLGetStream(
    int device_type, int device_id, DGLStreamHandle* handle);

/**
 * @brief Wait until all computations on stream completes.
 *
 * @param device_type The device type of context
 * @param device_id The device id of context.
 * @param stream The stream to be synchronized.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLSynchronize(
    int device_type, int device_id, DGLStreamHandle stream);

/**
 * @brief Synchronize two streams of execution.
 *
 * @param device_type The device type of context
 * @param device_id The device id of context
 * @param src The source stream to synchronize.
 * @param dst The destination stream to synchronize.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLStreamStreamSynchronize(
    int device_type, int device_id, DGLStreamHandle src, DGLStreamHandle dst);

/**
 * @brief Load tensor adapter.
 * @return 0 when success, -1 when failure happens.
 */
DGL_DLL int DGLLoadTensorAdapter(const char* path);

/**
 * @brief Pin host memory.
 */
int DGLArrayPinData(DGLArrayHandle handle, DGLContext ctx);

/**
 * @brief Unpin host memory.
 */
int DGLArrayUnpinData(DGLArrayHandle handle, DGLContext ctx);

/**
 * @brief Record the stream that's using this tensor.
 */
int DGLArrayRecordStream(DGLArrayHandle handle, DGLStreamHandle stream);

/**
 * @brief Bug report macro.
 *
 * This serves as a sanity check on system side to make sure the code is correct
 * by checking whether a condition always holds for complex reasons.  Failing
 * the condition signifies a system bug instead of users giving invalid inputs
 * or using the functionality incorrectly.
 *
 * Hints the user to file a bug report if the condition fails.
 */
#define BUG_IF_FAIL(cond)                                                    \
  CHECK(cond)                                                                \
      << "A bug has been occurred.  "                                        \
         "Please file a bug report at https://github.com/dmlc/dgl/issues.  " \
         "Message: "

#ifdef __cplusplus
}  // DGL_EXTERN_C
#endif
#endif  // DGL_RUNTIME_C_RUNTIME_API_H_
