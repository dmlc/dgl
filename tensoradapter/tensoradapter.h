#ifndef TENSORADAPTER_H_
#define TENSORADAPTER_H_

#include <dlpack/dlpack.h>

namespace tensoradapter {

DLManagedTensor *empty(int64_t n);

};

#endif  // TENSORADAPTER_H_
