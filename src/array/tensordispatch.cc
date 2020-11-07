#include <dlfcn.h>
#include <tensoradapter/tensoradapter.h>
#include <cstring>
#include "tensordispatch.h"

namespace dgl {
namespace aten {

namespace {
const char *getpath() {
  const char *platform = getenv("TA_BACKEND");
  if (strcmp(platform, "pytorch") == 0)
    return "./libtensordispatcher-pytorch.so";
  else
    return nullptr;
}
};

TensorDispatcher::TensorDispatcher() {
  const char *path = getpath();
  if (!path)
    // does not have dispatcher library; all operators fall back to DGL's implementation
    return;

  void *handle = dlopen(path, RTLD_LAZY);
  CHECK(handle) << "unable to open " << path << " " << dlerror();
  for (int i = 0; i < num_entries_; ++i)
    entrypoints_[i] = dlsym(handle, names_[i]);
}

NDArray TensorDispatcher::Empty(
    std::vector<int64_t> shape,
    DLDataType dtype,
    DLContext ctx) const {
  auto entry = entrypoints_[Op::kEmpty];

  if (!entrypoints_[Op::kEmpty]) {
    return NDArray::Empty(shape, dtype, ctx);
  } else {
    auto result = TA_DISPATCH(tensoradapter::empty, entry, shape, dtype, ctx);
    return NDArray::FromDLPack(result);
  }
}

};  // namespace aten
};  // namespace dgl
