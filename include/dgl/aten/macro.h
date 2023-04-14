/**
 *  Copyright (c) 2020 by Contributors
 * @file dgl/aten/macro.h
 * @brief Common macros for aten package.
 */

#ifndef DGL_ATEN_MACRO_H_
#define DGL_ATEN_MACRO_H_

///////////////////////// Dispatchers //////////////////////////

/**
 * Dispatch according to device:
 *
 * ATEN_XPU_SWITCH(array->ctx.device_type, XPU, {
 *   // Now XPU is a placeholder for array->ctx.device_type
 *   DeviceSpecificImplementation<XPU>(...);
 * });
 */
#define ATEN_XPU_SWITCH(val, XPU, op, ...)                               \
  do {                                                                   \
    if ((val) == kDGLCPU) {                                              \
      constexpr auto XPU = kDGLCPU;                                      \
      { __VA_ARGS__ }                                                    \
    } else {                                                             \
      LOG(FATAL) << "Operator " << (op) << " does not support "          \
                 << dgl::runtime::DeviceTypeCode2Str(val) << " device."; \
    }                                                                    \
  } while (0)

/**
 * Dispatch according to device:
 *
 * XXX(minjie): temporary macro that allows CUDA operator
 *
 * ATEN_XPU_SWITCH(array->ctx.device_type, XPU, {
 *   // Now XPU is a placeholder for array->ctx.device_type
 *   DeviceSpecificImplementation<XPU>(...);
 * });
 *
 * We treat pinned memory as normal host memory if we don't want
 * to enable CUDA UVA access for this operator
 */
#ifdef DGL_USE_CUDA
#define ATEN_XPU_SWITCH_CUDA(val, XPU, op, ...)                          \
  do {                                                                   \
    if ((val) == kDGLCPU) {                                              \
      constexpr auto XPU = kDGLCPU;                                      \
      { __VA_ARGS__ }                                                    \
    } else if ((val) == kDGLCUDA) {                                      \
      constexpr auto XPU = kDGLCUDA;                                     \
      { __VA_ARGS__ }                                                    \
    } else {                                                             \
      LOG(FATAL) << "Operator " << (op) << " does not support "          \
                 << dgl::runtime::DeviceTypeCode2Str(val) << " device."; \
    }                                                                    \
  } while (0)
#else  // DGL_USE_CUDA
#define ATEN_XPU_SWITCH_CUDA ATEN_XPU_SWITCH
#endif  // DGL_USE_CUDA

/**
 * Dispatch according to integral type (either int32 or int64):
 *
 * ATEN_ID_TYPE_SWITCH(array->dtype, IdType, {
 *   // Now IdType is the type corresponding to data type in array.
 *   // For instance, one can do this for a CPU array:
 *   DType *data = static_cast<DType *>(array->data);
 * });
 */
#define ATEN_ID_TYPE_SWITCH(val, IdType, ...)                   \
  do {                                                          \
    CHECK_EQ((val).code, kDGLInt) << "ID must be integer type"; \
    if ((val).bits == 32) {                                     \
      typedef int32_t IdType;                                   \
      { __VA_ARGS__ }                                           \
    } else if ((val).bits == 64) {                              \
      typedef int64_t IdType;                                   \
      { __VA_ARGS__ }                                           \
    } else {                                                    \
      LOG(FATAL) << "ID can only be int32 or int64";            \
    }                                                           \
  } while (0)

/**
 * Dispatch according to bits (either int32 or int64):
 *
 * ATEN_ID_BITS_SWITCH(bits, IdType, {
 *   // Now IdType is the type corresponding to data type in array.
 *   // For instance, one can do this for a CPU array:
 *   DType *data = static_cast<DType *>(array->data);
 * });
 */
#define ATEN_ID_BITS_SWITCH(bits, IdType, ...)                      \
  do {                                                              \
    CHECK((bits) == 32 || (bits) == 64) << "bits must be 32 or 64"; \
    if ((bits) == 32) {                                             \
      typedef int32_t IdType;                                       \
      { __VA_ARGS__ }                                               \
    } else if ((bits) == 64) {                                      \
      typedef int64_t IdType;                                       \
      { __VA_ARGS__ }                                               \
    } else {                                                        \
      LOG(FATAL) << "ID can only be int32 or int64";                \
    }                                                               \
  } while (0)

/**
 * Dispatch according to float type (either float32 or float64):
 *
 * ATEN_FLOAT_TYPE_SWITCH(array->dtype, FloatType, {
 *   // Now FloatType is the type corresponding to data type in array.
 *   // For instance, one can do this for a CPU array:
 *   FloatType *data = static_cast<FloatType *>(array->data);
 * });
 */
#define ATEN_FLOAT_TYPE_SWITCH(val, FloatType, val_name, ...)               \
  do {                                                                      \
    CHECK_EQ((val).code, kDGLFloat) << (val_name) << " must be float type"; \
    if ((val).bits == 32) {                                                 \
      typedef float FloatType;                                              \
      { __VA_ARGS__ }                                                       \
    } else if ((val).bits == 64) {                                          \
      typedef double FloatType;                                             \
      { __VA_ARGS__ }                                                       \
    } else {                                                                \
      LOG(FATAL) << (val_name) << " can only be float32 or float64";        \
    }                                                                       \
  } while (0)

/**
 * Dispatch according to float type, including 16bits
 * (float16/bfloat16/float32/float64).
 */
#ifdef DGL_USE_CUDA
#if BF16_ENABLED
#define ATEN_FLOAT_TYPE_SWITCH_16BITS(val, FloatType, XPU, val_name, ...)   \
  do {                                                                      \
    CHECK((val).code == kDGLFloat || (val.code == kDGLBfloat))              \
        << (val_name) << " must be float type";                             \
    if ((val).bits == 32) {                                                 \
      typedef float FloatType;                                              \
      { __VA_ARGS__ }                                                       \
    } else if ((val).bits == 64) {                                          \
      typedef double FloatType;                                             \
      { __VA_ARGS__ }                                                       \
    } else if (                                                             \
        XPU == kDGLCUDA && (val).bits == 16 && (val).code == kDGLFloat) {   \
      typedef __half FloatType;                                             \
      { __VA_ARGS__ }                                                       \
    } else if (                                                             \
        XPU == kDGLCUDA && (val).bits == 16 && (val).code == kDGLBfloat) {  \
      typedef __nv_bfloat16 FloatType;                                      \
      { __VA_ARGS__ }                                                       \
    } else if (                                                             \
        XPU == kDGLCPU && (val).bits == 16 && (val).code == kDGLFloat) {    \
      LOG(FATAL) << (val_name) << " can't be float16 on CPU";               \
    } else if (                                                             \
        XPU == kDGLCPU && (val).bits == 16 && (val).code == kDGLBfloat) {   \
      typedef BFloat16 FloatType;                                           \
      { __VA_ARGS__ }                                                       \
    } else {                                                                \
      LOG(FATAL) << (val_name)                                              \
                 << " can only be float16/bfloat16/float32/float64 on GPU"; \
    }                                                                       \
  } while (0)
#else  // BF16_ENABLED
#define ATEN_FLOAT_TYPE_SWITCH_16BITS(val, FloatType, XPU, val_name, ...)  \
  do {                                                                     \
    CHECK((val).code == kDGLFloat || (val.code == kDGLBfloat))             \
        << (val_name) << " must be float type";                            \
    if ((val).bits == 32) {                                                \
      typedef float FloatType;                                             \
      { __VA_ARGS__ }                                                      \
    } else if ((val).bits == 64) {                                         \
      typedef double FloatType;                                            \
      { __VA_ARGS__ }                                                      \
    } else if (                                                            \
        XPU == kDGLCUDA && (val).bits == 16 && (val).code == kDGLFloat) {  \
      typedef __half FloatType;                                            \
      { __VA_ARGS__ }                                                      \
    } else if (                                                            \
        XPU == kDGLCUDA && (val).bits == 16 && (val).code == kDGLBfloat) { \
      LOG(FATAL) << "bfloat16 requires CUDA >= 11.0";                      \
    } else if (                                                            \
        XPU == kDGLCPU && (val).bits == 16 && (val).code == kDGLFloat) {   \
      LOG(FATAL) << (val_name) << " can't be float16 on CPU";              \
    } else if (                                                            \
        XPU == kDGLCPU && (val).bits == 16 && (val).code == kDGLBfloat) {  \
      typedef BFloat16 FloatType;                                          \
      { __VA_ARGS__ }                                                      \
    } else {                                                               \
      LOG(FATAL) << (val_name)                                             \
                 << " can only be float16/float32/float64 on GPU";         \
    }                                                                      \
  } while (0)
#endif  // BF16_ENABLED
#else   // DGL_USE_CUDA
#define ATEN_FLOAT_TYPE_SWITCH_16BITS(val, FloatType, XPU, val_name, ...) \
  do {                                                                    \
    CHECK((val).code == kDGLFloat || (val.code == kDGLBfloat))            \
        << (val_name) << " must be float type";                           \
    if ((val).bits == 32) {                                               \
      typedef float FloatType;                                            \
      { __VA_ARGS__ }                                                     \
    } else if ((val).bits == 64) {                                        \
      typedef double FloatType;                                           \
      { __VA_ARGS__ }                                                     \
    } else if (                                                           \
        XPU == kDGLCPU && (val).bits == 16 && (val).code == kDGLBfloat) { \
      typedef BFloat16 FloatType;                                         \
      { __VA_ARGS__ }                                                     \
    } else {                                                              \
      LOG(FATAL) << (val_name)                                            \
                 << " can only be bfloat16/float32/float64 on CPU";       \
    }                                                                     \
  } while (0)
#endif  // DGL_USE_CUDA

/**
 * Dispatch according to data type (int32, int64, float32 or float64):
 *
 * ATEN_DTYPE_SWITCH(array->dtype, DType, {
 *   // Now DType is the type corresponding to data type in array.
 *   // For instance, one can do this for a CPU array:
 *   DType *data = static_cast<DType *>(array->data);
 * });
 */
#define ATEN_DTYPE_SWITCH(val, DType, val_name, ...)                 \
  do {                                                               \
    if ((val).code == kDGLInt && (val).bits == 32) {                 \
      typedef int32_t DType;                                         \
      { __VA_ARGS__ }                                                \
    } else if ((val).code == kDGLInt && (val).bits == 64) {          \
      typedef int64_t DType;                                         \
      { __VA_ARGS__ }                                                \
    } else if ((val).code == kDGLFloat && (val).bits == 32) {        \
      typedef float DType;                                           \
      { __VA_ARGS__ }                                                \
    } else if ((val).code == kDGLFloat && (val).bits == 64) {        \
      typedef double DType;                                          \
      { __VA_ARGS__ }                                                \
    } else {                                                         \
      LOG(FATAL) << (val_name)                                       \
                 << " can only be int32, int64, float32 or float64"; \
    }                                                                \
  } while (0)

/**
 * Dispatch according to data type (int8, uint8, float32 or float64):
 *
 * ATEN_FLOAT_INT8_UINT8_TYPE_SWITCH(array->dtype, DType, {
 *   // Now DType is the type corresponding to data type in array.
 *   // For instance, one can do this for a CPU array:
 *   DType *data = static_cast<DType *>(array->data);
 * });
 */
#define ATEN_FLOAT_INT8_UINT8_TYPE_SWITCH(val, DType, val_name, ...) \
  do {                                                               \
    if ((val).code == kDGLInt && (val).bits == 8) {                  \
      typedef int8_t DType;                                          \
      { __VA_ARGS__ }                                                \
    } else if ((val).code == kDGLUInt && (val).bits == 8) {          \
      typedef uint8_t DType;                                         \
      { __VA_ARGS__ }                                                \
    } else if ((val).code == kDGLFloat && (val).bits == 32) {        \
      typedef float DType;                                           \
      { __VA_ARGS__ }                                                \
    } else if ((val).code == kDGLFloat && (val).bits == 64) {        \
      typedef double DType;                                          \
      { __VA_ARGS__ }                                                \
    } else {                                                         \
      LOG(FATAL) << (val_name)                                       \
                 << " can only be int8, uint8, float32 or float64";  \
    }                                                                \
  } while (0)

/**
 * Dispatch data type only based on bit-width (8-bit, 16-bit, 32-bit, 64-bit):
 *
 * ATEN_DTYPE_BITS_ONLY_SWITCH(array->dtype, DType, {
 *   // Now DType is the type which has the same bit-width with the
 *   // data type in array.
 *   // Do not use for computation, but only for read and write.
 *   // For instance, one can do this for a CPU array:
 *   DType *data = static_cast<DType *>(array->data);
 * });
 */
#define ATEN_DTYPE_BITS_ONLY_SWITCH(val, DType, val_name, ...)       \
  do {                                                               \
    if ((val).bits == 8) {                                           \
      typedef int8_t DType;                                          \
      { __VA_ARGS__ }                                                \
    } else if ((val).bits == 16) {                                   \
      typedef int16_t DType;                                         \
      { __VA_ARGS__ }                                                \
    } else if ((val).bits == 32) {                                   \
      typedef int32_t DType;                                         \
      { __VA_ARGS__ }                                                \
    } else if ((val).bits == 64) {                                   \
      typedef int64_t DType;                                         \
      { __VA_ARGS__ }                                                \
    } else {                                                         \
      LOG(FATAL) << (val_name)                                       \
                 << " can only be 8-bit, 16-bit, 32-bit, or 64-bit"; \
    }                                                                \
  } while (0)

/**
 * Dispatch according to integral type of CSR graphs.
 * Identical to ATEN_ID_TYPE_SWITCH except for a different error message.
 */
#define ATEN_CSR_DTYPE_SWITCH(val, DType, ...)                    \
  do {                                                            \
    if ((val).code == kDGLInt && (val).bits == 32) {              \
      typedef int32_t DType;                                      \
      { __VA_ARGS__ }                                             \
    } else if ((val).code == kDGLInt && (val).bits == 64) {       \
      typedef int64_t DType;                                      \
      { __VA_ARGS__ }                                             \
    } else {                                                      \
      LOG(FATAL) << "CSR matrix data can only be int32 or int64"; \
    }                                                             \
  } while (0)

// Macro to dispatch according to device context and index type.
#define ATEN_CSR_SWITCH(csr, XPU, IdType, op, ...)                     \
  ATEN_XPU_SWITCH((csr).indptr->ctx.device_type, XPU, op, {            \
    ATEN_ID_TYPE_SWITCH((csr).indptr->dtype, IdType, {{__VA_ARGS__}}); \
  });

// Macro to dispatch according to device context and index type.
#define ATEN_COO_SWITCH(coo, XPU, IdType, op, ...)                  \
  ATEN_XPU_SWITCH((coo).row->ctx.device_type, XPU, op, {            \
    ATEN_ID_TYPE_SWITCH((coo).row->dtype, IdType, {{__VA_ARGS__}}); \
  });

#define CHECK_VALID_CONTEXT(VAR1, VAR2)                          \
  CHECK(                                                         \
      ((VAR1)->ctx == (VAR2)->ctx) || (VAR1).IsPinned() ||       \
      ((VAR1).NumElements() == 0)) /* Let empty arrays pass */   \
      << "Expected " << (#VAR2) << "(" << (VAR2)->ctx << ")"     \
      << " to have the same device "                             \
      << "context as " << (#VAR1) << "(" << (VAR1)->ctx << "). " \
      << "Or " << (#VAR1) << "(" << (VAR1)->ctx << ")"           \
      << " is pinned";

/**
 * Macro to dispatch according to the context of array and dtype of csr
 * to enable CUDA UVA ops.
 * Context check is covered here to avoid confusion with CHECK_SAME_CONTEXT.
 * If csr has the same context with array, same behivor as ATEN_CSR_SWITCH_CUDA.
 * If csr is pinned, array's context will conduct the actual operation.
 */
#define ATEN_CSR_SWITCH_CUDA_UVA(csr, array, XPU, IdType, op, ...)       \
  do {                                                                   \
    CHECK_VALID_CONTEXT(csr.indices, array);                             \
    ATEN_XPU_SWITCH_CUDA(array->ctx.device_type, XPU, op, {              \
      ATEN_ID_TYPE_SWITCH((csr).indptr->dtype, IdType, {{__VA_ARGS__}}); \
    });                                                                  \
  } while (0)

// Macro to dispatch according to device context (allowing cuda)
#ifdef DGL_USE_CUDA
#define ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, op, ...)                \
  ATEN_XPU_SWITCH_CUDA((csr).indptr->ctx.device_type, XPU, op, {       \
    ATEN_ID_TYPE_SWITCH((csr).indptr->dtype, IdType, {{__VA_ARGS__}}); \
  });

// Macro to dispatch according to device context and index type.
#define ATEN_COO_SWITCH_CUDA(coo, XPU, IdType, op, ...)             \
  ATEN_XPU_SWITCH_CUDA((coo).row->ctx.device_type, XPU, op, {       \
    ATEN_ID_TYPE_SWITCH((coo).row->dtype, IdType, {{__VA_ARGS__}}); \
  });
#else  // DGL_USE_CUDA
#define ATEN_CSR_SWITCH_CUDA ATEN_CSR_SWITCH
#define ATEN_COO_SWITCH_CUDA ATEN_COO_SWITCH
#endif  // DGL_USE_CUDA

///////////////////////// Array checks //////////////////////////

#define IS_INT32(a) ((a)->dtype.code == kDGLInt && (a)->dtype.bits == 32)
#define IS_INT64(a) ((a)->dtype.code == kDGLInt && (a)->dtype.bits == 64)
#define IS_FLOAT32(a) ((a)->dtype.code == kDGLFloat && (a)->dtype.bits == 32)
#define IS_FLOAT64(a) ((a)->dtype.code == kDGLFloat && (a)->dtype.bits == 64)

#define CHECK_IF(cond, prop, value_name, dtype_name)                           \
  CHECK(cond) << "Expecting " << (prop) << " of " << (value_name) << " to be " \
              << (dtype_name)

#define CHECK_INT32(value, value_name) \
  CHECK_IF(IS_INT32(value), "dtype", value_name, "int32")
#define CHECK_INT64(value, value_name) \
  CHECK_IF(IS_INT64(value), "dtype", value_name, "int64")
#define CHECK_INT(value, value_name)                           \
  CHECK_IF(                                                    \
      IS_INT32(value) || IS_INT64(value), "dtype", value_name, \
      "int32 or int64")
#define CHECK_FLOAT32(value, value_name) \
  CHECK_IF(IS_FLOAT32(value), "dtype", value_name, "float32")
#define CHECK_FLOAT64(value, value_name) \
  CHECK_IF(IS_FLOAT64(value), "dtype", value_name, "float64")
#define CHECK_FLOAT(value, value_name)                             \
  CHECK_IF(                                                        \
      IS_FLOAT32(value) || IS_FLOAT64(value), "dtype", value_name, \
      "float32 or float64")

#define CHECK_NDIM(value, _ndim, value_name) \
  CHECK_IF((value)->ndim == (_ndim), "ndim", value_name, _ndim)

#define CHECK_SAME_DTYPE(VAR1, VAR2)                                     \
  CHECK((VAR1)->dtype == (VAR2)->dtype)                                  \
      << "Expected " << (#VAR2) << " to be the same type as " << (#VAR1) \
      << "(" << (VAR1)->dtype << ")"                                     \
      << ". But got " << (VAR2)->dtype << ".";

#define CHECK_SAME_CONTEXT(VAR1, VAR2)                                    \
  CHECK((VAR1)->ctx == (VAR2)->ctx)                                       \
      << "Expected " << (#VAR2) << " to have the same device context as " \
      << (#VAR1) << "(" << (VAR1)->ctx << ")"                             \
      << ". But got " << (VAR2)->ctx << ".";

#define CHECK_NO_OVERFLOW(dtype, val)                         \
  do {                                                        \
    if (sizeof(val) == 8 && (dtype).bits == 32)               \
      CHECK_LE((val), 0x7FFFFFFFL)                            \
          << "int32 overflow for argument " << (#val) << "."; \
  } while (0);

#define CHECK_IS_ID_ARRAY(VAR)                                \
  CHECK((VAR)->ndim == 1 && (IS_INT32(VAR) || IS_INT64(VAR))) \
      << "Expected argument " << (#VAR) << " to be an 1D integer array.";

#endif  // DGL_ATEN_MACRO_H_
