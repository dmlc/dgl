#include <tensoradapter.h>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <dlfcn.h>

namespace tensoradapter {

std::unordered_map<const char *, void *> functbl;

const char *funcnames[] = {
  "empty"
};

const char *getpath() {
  const std::string &platform = getenv("TA_BACKEND");
  if (platform == "pytorch")
    return "libtensoradapter-pytorch.so";
  else
    return nullptr;
}

void __attribute__((constructor)) __init() {
  const char *path = getpath();
  void *lib = dlopen(path, RTLD_LAZY);

  for (const char *name : funcnames) {
    if (!path) {
      functbl[name] = nullptr;
      continue
    }

    functbl[name] = dlsym(lib, name);
  }
}

DLManagedTensor *empty(int64_t n) {
  return (*static_cast<decltype(&empty)>(functbl["empty"]))(n);
}

};  // namespace tensoradapter
