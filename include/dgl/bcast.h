/*!
 *  Copyright (c) 2020 by Contributors
 * \file dgl/aten/bcast.h
 * \brief Broadcast related function C++ header.
 */
#ifndef DGL_BCAST_H_
#define DGL_BCAST_H_

#include <string>
#include <vector>
#include "./runtime/ndarray.h"

using namespace dgl::runtime;

namespace dgl {

/*!
 * \brief: Broadcast information.
 */
struct BcastOff {
  std::vector<int64_t> lhs_offset, rhs_offset;
  bool use_bcast;
  int64_t lhs_len, rhs_len, out_len, reduce_size;
};

/*!
 * \brief: Compute broadcast information given operator and operands.
 */
BcastOff CalcBcastOff(const std::string& op, NDArray lhs, NDArray rhs);

}   // namespace dgl

#endif  // DGL_BCAST_H_
