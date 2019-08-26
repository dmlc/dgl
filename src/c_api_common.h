/*!
 *  Copyright (c) 2018 by Contributors
 * \file c_api_common.h
 * \brief DGL C API common util functions
 */
#ifndef DGL_C_API_COMMON_H_
#define DGL_C_API_COMMON_H_

#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>
#include <dgl/array.h>
#include <dgl/graph_interface.h>
#include <algorithm>
#include <vector>

/*! \brief Check whether two data types are the same.*/
inline bool operator == (const DLDataType& ty1, const DLDataType& ty2) {
  return ty1.code == ty2.code && ty1.bits == ty2.bits && ty1.lanes == ty2.lanes;
}

/*! \brief Output the string representation of device context.*/
inline std::ostream& operator << (std::ostream& os, const DLDataType& ty) {
  return os << "code=" << ty.code << ",bits=" << ty.bits << "lanes=" << ty.lanes;
}

/*! \brief Check whether two device contexts are the same.*/
inline bool operator == (const DLContext& ctx1, const DLContext& ctx2) {
  return ctx1.device_type == ctx2.device_type && ctx1.device_id == ctx2.device_id;
}

/*! \brief Output the string representation of device context.*/
inline std::ostream& operator << (std::ostream& os, const DLContext& ctx) {
  return os << ctx.device_type << ":" << ctx.device_id;
}

namespace dgl {

// Communicator handler type
typedef void* CommunicatorHandle;

// KVstore message handler type
typedef void* KVMsgHandle;

/*! \brief Enum type for bool value with unknown */
enum BoolFlag {
  kBoolUnknown = -1,
  kBoolFalse = 0,
  kBoolTrue = 1
};

/*!
 * \brief Convert a vector of NDArray to PackedFunc.
 */
dgl::runtime::PackedFunc ConvertNDArrayVectorToPackedFunc(
    const std::vector<dgl::runtime::NDArray>& vec);

/*!
 * \brief Copy a vector to an int64_t NDArray.
 *
 * The element type of the vector must be convertible to int64_t.
 */
template<typename DType>
dgl::runtime::NDArray CopyVectorToNDArray(
    const std::vector<DType>& vec) {
  using dgl::runtime::NDArray;
  const int64_t len = vec.size();
  NDArray a = NDArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  std::copy(vec.begin(), vec.end(), static_cast<int64_t*>(a->data));
  return a;
}

runtime::PackedFunc ConvertEdgeArrayToPackedFunc(const EdgeArray& ea);

}  // namespace dgl

#endif  // DGL_C_API_COMMON_H_
