/*!
 *  Copyright (c) 2019 by Contributors
 * \file runtime/tensordispatch.cc
 * \brief Adapter library caller
 */

#include <dgl/runtime/tensordispatch.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/env.h>
#include <dgl/packed_func_ext.h>
#if defined(WIN32) || defined(_WIN32)
#include <windows.h>
#else   // !WIN32
#include <dlfcn.h>
#endif  // WIN32
#include <cstring>

namespace dgl {
namespace runtime {

constexpr const char *TensorDispatcher::names_[];

TensorDispatcher::TensorDispatcher() {
  const std::string& path = Env::Global()->ta_path;
  if (path == "")
    // does not have dispatcher library; all operators fall back to DGL's implementation
    return;

#if defined(WIN32) || defined(_WIN32)
  handle_ = LoadLibrary(path.c_str());

  if (!handle_)
    return;

  for (int i = 0; i < num_entries_; ++i)
    entrypoints_[i] = reinterpret_cast<void*>(GetProcAddress(handle_, names_[i]));
#else   // !WIN32
  handle_ = dlopen(path.c_str(), RTLD_LAZY);
  if (!handle_)
    return;
  for (int i = 0; i < num_entries_; ++i)
    entrypoints_[i] = dlsym(handle_, names_[i]);
#endif  // WIN32

  available_ = true;
}

TensorDispatcher::~TensorDispatcher() {
  if (handle_) {
#if defined(WIN32) || defined(_WIN32)
    FreeLibrary(handle_);
#else   // !WIN32
    dlclose(handle_);
#endif  // WIN32
  }
}

};  // namespace runtime
};  // namespace dgl
