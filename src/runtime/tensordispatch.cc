/*!
 *  Copyright (c) 2019 by Contributors
 * \file runtime/tensordispatch.cc
 * \brief Adapter library caller
 */

#include <dgl/runtime/tensordispatch.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/env.h>
#include <dgl/packed_func_ext.h>
#ifdef WIN32
#include <windows.h>
#else   // !WIN32
#include <dlfcn.h>
#endif  // WIN32
#include <cstring>

namespace dgl {
namespace runtime {

namespace {

/*!
 * \brief Get the absolute path of the tensor adapter library to be loaded.
 */
#if defined(WIN32)
#define PATH_PYTORCH  "\\tensoradapter\\pytorch\\tensoradapter_pytorch.dll"
#elif defined(APPLE)
#define PATH_PYTORCH  "/tensoradapter/pytorch/libtensoradapter_pytorch.dylib"
#else
#define PATH_PYTORCH  "/tensoradapter/pytorch/libtensoradapter_pytorch.so"
#endif
std::string getpath() {
  const std::string& backend = Env::Global()->backend;
  const std::string& dir = Env::Global()->dir;

  if (backend == "pytorch")
    return dir + PATH_PYTORCH;
  else
    return "";
}

};

constexpr const char *TensorDispatcher::names_[];

#ifdef WIN32
TensorDispatcher::TensorDispatcher() {
  std::string path = getpath();
  if (path == "")
    return;

  handle_ = LoadLibrary(path.c_str());
  CHECK(handle_) << "Win32 error: " << GetLastError();    // TODO(BarclayII): Get error string
  for (int i = 0; i < num_entries_; ++i)
    entrypoints_[i] = reinterpret_cast<void*>(GetProcAddress(handle_, names_[i]));
  available_ = true;
}

TensorDispatcher::~TensorDispatcher() {
  FreeLibrary(handle_);
}
#else   // !WIN32
TensorDispatcher::TensorDispatcher() {
  std::string path = getpath();
  if (path == "")
    // does not have dispatcher library; all operators fall back to DGL's implementation
    return;

  handle_ = dlopen(path.c_str(), RTLD_LAZY);
  CHECK(handle_) << dlerror();
  for (int i = 0; i < num_entries_; ++i)
    entrypoints_[i] = dlsym(handle_, names_[i]);
  available_ = true;
}

TensorDispatcher::~TensorDispatcher() {
  dlclose(handle_);
}
#endif  // WIN32

};  // namespace aten
};  // namespace dgl
