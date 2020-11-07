#ifndef TENSORADAPTER_H_
#define TENSORADAPTER_H_

#include <dlpack/dlpack.h>
#include <vector>

#ifdef WIN32
#define TA_EXPORTS __declspec(dllexport)
#else
#define TA_EXPORTS
#endif

namespace tensoradapter {

extern "C" {

TA_EXPORTS DLManagedTensor *empty(
    std::vector<int64_t> shape, DLDataType dtype, DLContext ctx);

#define TA_DISPATCH(func, entry, ...) \
  ((*reinterpret_cast<decltype(&func)>(entry))(__VA_ARGS__))

}

};  // namespace tensoradapter

#endif  // TENSORADAPTER_H_
