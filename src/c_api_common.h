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
#include <string>

using dgl::runtime::operator<<;

/*! \brief Output the string representation of device context.*/
inline std::ostream& operator<<(std::ostream& os, const DLContext& ctx) {
  std::string device_name;
  switch (ctx.device_type) {
    case kDLCPU:
      device_name = "CPU";
      break;
    case kDLGPU:
      device_name = "GPU";
      break;
    default:
      device_name = "Unknown device";
  }
  return os << device_name << ":" << ctx.device_id;
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
