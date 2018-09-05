/*!
 *  Copyright (c) 2017 by Contributors
 * \file module_util.h
 * \brief Helper utilities for module building
 */
#ifndef TVM_RUNTIME_MODULE_UTIL_H_
#define TVM_RUNTIME_MODULE_UTIL_H_

#include <dgl/runtime/module.h>
#include <dgl/runtime/c_runtime_api.h>
#include <dgl/runtime/c_backend_api.h>
#include <vector>

extern "C" {
// Function signature for generated packed function in shared library
typedef int (*BackendPackedCFunc)(void* args,
                                  int* type_codes,
                                  int num_args);
}  // extern "C"

namespace tvm {
namespace runtime {
/*!
 * \brief Wrap a BackendPackedCFunc to packed function.
 * \param faddr The function address
 * \param mptr The module pointer node.
 */
PackedFunc WrapPackedFunc(BackendPackedCFunc faddr, const std::shared_ptr<ModuleNode>& mptr);
/*!
 * \brief Load and append module blob to module list
 * \param mblob The module blob.
 * \param module_list The module list to append to
 */
void ImportModuleBlob(const char* mblob, std::vector<Module>* module_list);

/*!
 * \brief Utility to initialize conext function symbols during startup
 * \param flookup A symbol lookup function.
 * \tparam FLookup a function of signature string->void*
 */
template<typename FLookup>
void InitContextFunctions(FLookup flookup) {
  #define TVM_INIT_CONTEXT_FUNC(FuncName)                     \
    if (auto *fp = reinterpret_cast<decltype(&FuncName)*>     \
      (flookup("__" #FuncName))) {                            \
      *fp = FuncName;                                         \
    }
  // Initialize the functions
  TVM_INIT_CONTEXT_FUNC(TVMFuncCall);
  TVM_INIT_CONTEXT_FUNC(TVMAPISetLastError);
  TVM_INIT_CONTEXT_FUNC(TVMBackendGetFuncFromEnv);
  TVM_INIT_CONTEXT_FUNC(TVMBackendAllocWorkspace);
  TVM_INIT_CONTEXT_FUNC(TVMBackendFreeWorkspace);
  TVM_INIT_CONTEXT_FUNC(TVMBackendParallelLaunch);
  TVM_INIT_CONTEXT_FUNC(TVMBackendParallelBarrier);

  #undef TVM_INIT_CONTEXT_FUNC
}
}  // namespace runtime
}  // namespace tvm
#endif   // TVM_RUNTIME_MODULE_UTIL_H_
