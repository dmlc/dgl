// DGL Common util functions
#ifndef DGL_COMMON_H_
#define DGL_COMMON_H_

#include "runtime/ndarray.h"
#include "runtime/packed_func.h"
#include "runtime/registry.h"
#include <vector>

using tvm::runtime::TVMArgs;
using tvm::runtime::TVMArgValue;
using tvm::runtime::TVMRetValue;
using tvm::runtime::PackedFunc;
using tvm::runtime::NDArray;

namespace dgl {

// Convert the given DLTensor to a temporary DLManagedTensor that does not own memory.
DLManagedTensor* CreateTmpDLManagedTensor(const TVMArgValue& arg);

// Convert a vector of NDArray to PackedFunc
PackedFunc ConvertNDArrayVectorToPackedFunc(const std::vector<NDArray>& vec);

} // namespace dgl

#endif // DGL_COMMON_H_
