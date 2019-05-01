#pragma once

#include <cstdlib>

namespace dgl {
namespace kernel {
namespace utils {
inline int FindNumThreads(int dim, int max_nthrs) {
  int ret = max_nthrs;
  while (ret > dim) {
    ret = ret >> 1;
  }
  return ret;
}

template <int XPU, typename DType>
void Fill(DType* ptr, size_t length, DType val);

}  // namespace utils
}  // namespace kernel
}  // namespace dgl
