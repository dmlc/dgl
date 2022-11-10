/**
 *  Copyright (c) 2019 by Contributors
 * @file runtime/tensordispatch.cc
 * @brief Adapter library caller
 */

#include <dgl/packed_func_ext.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/tensordispatch.h>
#if defined(WIN32) || defined(_WIN32)
#include <windows.h>
#else  // !WIN32
#include <dlfcn.h>
#endif  // WIN32
#include <cstring>

namespace dgl {
namespace runtime {

constexpr const char *TensorDispatcher::names_[];

bool TensorDispatcher::Load(const char *path) {
  CHECK(!available_) << "The tensor adapter can only load once.";

  if (path == nullptr || strlen(path) == 0)
    // does not have dispatcher library; all operators fall back to DGL's
    // implementation
    return false;

#if defined(WIN32) || defined(_WIN32)
  handle_ = LoadLibrary(path);

  if (!handle_) return false;

  for (int i = 0; i < num_entries_; ++i) {
    entrypoints_[i] =
        reinterpret_cast<void *>(GetProcAddress(handle_, names_[i]));
    CHECK(entrypoints_[i]) << "cannot locate symbol " << names_[i];
  }
#else   // !WIN32
  handle_ = dlopen(path, RTLD_LAZY);

  if (!handle_) {
    DLOG(WARNING)
        << "Could not open file: " << dlerror()
        << ". This does not affect DGL's but might impact its performance.";
    return false;
  }

  for (int i = 0; i < num_entries_; ++i) {
    entrypoints_[i] = dlsym(handle_, names_[i]);
    CHECK(entrypoints_[i]) << "cannot locate symbol " << names_[i];
  }
#endif  // WIN32

  available_ = true;
  return true;
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
