/**
 *  Copyright (c) 2018 by Contributors
 * @file c_api_common.h
 * @brief DGL C API common util functions
 */
#ifndef DGL_C_API_COMMON_H_
#define DGL_C_API_COMMON_H_

#include <dgl/array.h>
#include <dgl/graph_interface.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace dgl {

// Communicator handler type
typedef void* CommunicatorHandle;

// KVstore message handler type
typedef void* KVMsgHandle;

/**
 * @brief Convert a vector of NDArray to PackedFunc.
 */
dgl::runtime::PackedFunc ConvertNDArrayVectorToPackedFunc(
    const std::vector<dgl::runtime::NDArray>& vec);

/**
 * @brief Copy a vector to an NDArray.
 *
 * The data type of the NDArray will be IdType, which must be an integer type.
 * The element type (DType) of the vector must be convertible to IdType.
 */
template <typename IdType, typename DType>
dgl::runtime::NDArray CopyVectorToNDArray(const std::vector<DType>& vec) {
  using dgl::runtime::NDArray;
  const int64_t len = vec.size();
  NDArray a = NDArray::Empty(
      {len}, DGLDataType{kDGLInt, sizeof(IdType) * 8, 1},
      DGLContext{kDGLCPU, 0});
  std::copy(vec.begin(), vec.end(), static_cast<IdType*>(a->data));
  return a;
}

runtime::PackedFunc ConvertEdgeArrayToPackedFunc(const EdgeArray& ea);

}  // namespace dgl

#endif  // DGL_C_API_COMMON_H_
