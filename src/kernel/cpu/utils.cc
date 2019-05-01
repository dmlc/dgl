/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/utils.cc
 * \brief Utility function implementations on CPU
 */
#include "../utils.h"

namespace dgl {
namespace kernel {
namespace utils {

template <int XPU, typename DType>
void Fill(const DLContext& ctx, DType* ptr, size_t length, DType val) {
  for (size_t i = 0; i < length; ++i) {
    *(ptr + i) = val;
  }
}

template void Fill<kDLCPU, float>(const DLContext& ctx, float* ptr, size_t length, float val);
template void Fill<kDLCPU, double>(const DLContext& ctx, double* ptr, size_t length, double val);

}  // namespace utils
}  // namespace kernel
}  // namespace dgl
