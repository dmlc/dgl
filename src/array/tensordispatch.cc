#include <dgl/aten/tensordispatch.h>
#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/container.h>
#include <dlfcn.h>
#include <cstring>

namespace dgl {
namespace aten {

namespace {
const char *getpath() {
  const char *platform = getenv("TA_BACKEND");
  if (strcmp(platform, "pytorch") == 0)
    return "./libtensoradapter_torch.so";
  else
    return nullptr;
}
};

constexpr const char *TensorDispatcher::names_[];

TensorDispatcher::TensorDispatcher() {
  const char *path = getpath();
  if (!path)
    // does not have dispatcher library; all operators fall back to DGL's implementation
    return;

  void *handle = dlopen(path, RTLD_LAZY);
  if (handle)
    std::cout << "test" << std::endl;
  CHECK(handle) << "unable to open " << path << " " << dlerror();
  for (int i = 0; i < num_entries_; ++i) {
    entrypoints_[i] = dlsym(handle, names_[i]);
    std::cout << entrypoints_[i] << std::endl;
  }
}

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_Test")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    *rv = TensorDispatcher::Global()->Empty({2, 3}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  });

};  // namespace aten
};  // namespace dgl
