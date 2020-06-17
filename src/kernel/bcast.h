/*!
 *  Copyright (c) 2020 by Contributors
 * \file kernel/bcast.h
 * \brief Broadcast related function C++ header.
 */
#ifndef DGL_KERNEL_BCAST_H_
#define DGL_KERNEL_BCAST_H_

#include <dgl/runtime/ndarray.h>
#include <string>
#include <vector>

using namespace dgl::runtime;

namespace dgl {
namespace kernel {

struct BcastOff {
  std::vector<int64_t> lhs_offset, rhs_offset;
  bool use_bcast;
  int64_t lhs_len, rhs_len, out_len, reduce_size;
};

bool HasBcast1(NDArray lhs, NDArray rhs);
BcastOff CalcBcastOff(const std::string& op, NDArray lhs, NDArray rhs);

}   // namespace kernel
}   // namespace dgl

#endif  // DGL_KERNEL_BCAST_H_
