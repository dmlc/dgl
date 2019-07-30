/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/common.h
 * \brief Array operator common utilities
 */
#ifndef DGL_ARRAY_COMMON_H_
#define DGL_ARRAY_COMMON_H_

namespace dgl {
namespace aten {

#define ATEN_XPU_SWITCH(val, XPU, ...)                          \
  if ((val) == kDLCPU) {                                        \
    constexpr auto XPU = kDLCPU;                                \
    {__VA_ARGS__}                                               \
  } else {                                                      \
    LOG(FATAL) << "Device type: " << (val) << " is not supported.";  \
  }

#define ATEN_ID_TYPE_SWITCH(val, IdType, ...)                 \
  CHECK_EQ((val).code, kDLInt) << "ID must be integer type";  \
  if ((val).bits == 32) {                                     \
    typedef int32_t IdType;                                   \
    {__VA_ARGS__}                                             \
  } else if ((val).bits == 64) {                              \
    typedef int64_t IdType;                                   \
    {__VA_ARGS__}                                             \
  } else {                                                    \
    LOG(FATAL) << "ID can Only be int32 or int64";            \
  }

#define ATEN_CSR_DTYPE_SWITCH(val, DType, ...)              \
  if ((val).code == kDLInt && (val).bits == 32) {           \
    typedef int32_t DType;                                  \
    {__VA_ARGS__}                                           \
  } else if ((val).code == kDLInt && (val).bits == 64) {    \
    typedef int64_t DType;                                  \
    {__VA_ARGS__}                                           \
  } else {                                                  \
    LOG(FATAL) << "CSR matrix data can only be int32 or int64";  \
  }

// Macro to dispatch according to device context, index type and data type
// TODO(minjie): In our current use cases, data type and id type are the
//   same. For example, data array is used to store edge ids.
#define ATEN_CSR_SWITCH(csr, XPU, IdType, DType, ...)       \
  ATEN_XPU_SWITCH(csr.indptr->ctx.device_type, XPU, {       \
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {        \
      typedef IdType DType;                                 \
      {__VA_ARGS__}                                         \
    });                                                     \
  });

// Macro to dispatch according to device context and index type
#define ATEN_CSR_IDX_SWITCH(csr, XPU, IdType, ...)          \
  ATEN_XPU_SWITCH(csr.indptr->ctx.device_type, XPU, {       \
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {        \
      {__VA_ARGS__}                                         \
    });                                                     \
  });

// Macro to dispatch according to device context, index type and data type
// TODO(minjie): In our current use cases, data type and id type are the
//   same. For example, data array is used to store edge ids.
#define ATEN_COO_SWITCH(coo, XPU, IdType, DType, ...)       \
  ATEN_XPU_SWITCH(coo.row->ctx.device_type, XPU, {          \
    ATEN_ID_TYPE_SWITCH(coo.row->dtype, IdType, {           \
      typedef IdType DType;                                 \
      {__VA_ARGS__}                                         \
    });                                                     \
  });

// Macro to dispatch according to device context and index type
#define ATEN_COO_IDX_SWITCH(coo, XPU, IdType, ...)          \
  ATEN_XPU_SWITCH(coo.row->ctx.device_type, XPU, {          \
    ATEN_ID_TYPE_SWITCH(coo.row->dtype, IdType, {           \
      {__VA_ARGS__}                                         \
    });                                                     \
  });

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_COMMON_H_
