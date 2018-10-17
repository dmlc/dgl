// DGL Common util functions
#ifndef DGL_C_API_COMMON_H_
#define DGL_C_API_COMMON_H_

#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>
#include <vector>

namespace dgl {

// Graph handler type
typedef void* GraphHandle;

// Convert the given DLTensor to a temporary DLManagedTensor that does not own memory.
DLManagedTensor* CreateTmpDLManagedTensor(const tvm::runtime::TVMArgValue& arg);

// Convert a vector of NDArray to PackedFunc
tvm::runtime::PackedFunc ConvertNDArrayVectorToPackedFunc(const std::vector<tvm::runtime::NDArray>& vec);

} // namespace dgl

#endif // DGL_COMMON_H_
