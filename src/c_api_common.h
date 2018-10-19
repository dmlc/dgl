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
#include <vector>

namespace dgl {

// Graph handler type
typedef void* GraphHandle;

/*!
 * \brief Convert the given DLTensor to DLManagedTensor.
 *
 * Return a temporary DLManagedTensor that does not own memory.
 */
DLManagedTensor* CreateTmpDLManagedTensor(
    const tvm::runtime::TVMArgValue& arg);

/*!
 * \brief Convert a vector of NDArray to PackedFunc.
 */
tvm::runtime::PackedFunc ConvertNDArrayVectorToPackedFunc(
    const std::vector<tvm::runtime::NDArray>& vec);

}  // namespace dgl

#endif  // DGL_C_API_COMMON_H_
