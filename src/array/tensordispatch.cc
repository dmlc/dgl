#include <dgl/aten/tensordispatch.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/env.h>
#include <dgl/packed_func_ext.h>
#include <dlfcn.h>
#include <libgen.h>
#include <cstring>

namespace dgl {
namespace aten {

using namespace dgl::runtime;

namespace {

/*!
 * \brief Get the absolute path of the tensor adapter library to be loaded.
 *
 * TODO: support Windows.  basename() can be replaced with PathRemoveFileSpec.
 */
std::string getpath() {
  const std::string& backend = Env::Global()->backend;
  const std::string& libpath = Env::Global()->libpath;

  std::string directory = dirname(strdup(libpath.c_str()));

  if (backend == "pytorch")
    return directory + "/tensoradapter/torch/libtensoradapter_torch.so";
  else
    return "";
}

};

constexpr const char *TensorDispatcher::names_[];

TensorDispatcher::TensorDispatcher() {
  std::string path = getpath();
  if (path == "")
    // does not have dispatcher library; all operators fall back to DGL's implementation
    return;

  void *handle = dlopen(path.c_str(), RTLD_LAZY);
  CHECK(handle) << dlerror();
  for (int i = 0; i < num_entries_; ++i)
    entrypoints_[i] = dlsym(handle, names_[i]);
}

};  // namespace aten
};  // namespace dgl
