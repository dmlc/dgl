/*!
 *  Copyright (c) 2020 by Contributors
 * \file kernel/bcast.h
 * \brief Broadcast related function implementations.
 */
#include <dgl/bcast.h>
#include <algorithm>

namespace dgl {

namespace {

bool UseBcast(const std::string& op, NDArray lhs, NDArray rhs) {
  if (op == "copy_u" || op == "copy_e")
    return false;
  if (lhs->ndim != rhs->ndim)
    return true;
  for (int i = 1; i < lhs->ndim; ++i) {
    if (lhs->shape[i] != rhs->shape[i])
      return true;
  }
  return false;
}

}  // namespace

BcastOff CalcBcastOff(const std::string& op, NDArray lhs, NDArray rhs) {
  BcastOff rst;
  rst.lhs_len = 1;
  rst.rhs_len = 1;
  for (int i = 1; i < lhs->ndim; ++i)
    rst.lhs_len *= lhs->shape[i];
  for (int i = 1; i < rhs->ndim; ++i)
    rst.rhs_len *= rhs->shape[i];
  rst.use_bcast = UseBcast(op, lhs, rhs);
  rst.reduce_size = 1;
  if (rst.use_bcast) {
    const int max_ndim = std::max(lhs->ndim, rhs->ndim) - 1;
    int out_len = 1, j = 0;
    if (op == "dot") {
      rst.reduce_size = lhs->shape[lhs->ndim - 1];
      ++j;
    }
    int stride_l = 1, stride_r = 1;
    rst.lhs_offset.push_back(0);
    rst.rhs_offset.push_back(0);
    for (; j < max_ndim; ++j) {
      const int dl = (lhs->ndim - 1 - j < 1) ? 1 : lhs->shape[lhs->ndim - 1 - j];
      const int dr = (rhs->ndim - 1 - j < 1) ? 1 : rhs->shape[rhs->ndim - 1 - j];
      for (int i = 1; i < std::max(dl, dr); ++i) {
        for (int offset = 0; offset < out_len; ++offset) {
          rst.lhs_offset.push_back(rst.lhs_offset[offset] + i * (i < dl) * stride_l);
          rst.rhs_offset.push_back(rst.rhs_offset[offset] + i * (i < dr) * stride_r);
        }
      }
      out_len *= std::max(dl, dr);
      stride_l *= dl;
      stride_r *= dr;
    }
    rst.out_len = out_len;
  } else {
    rst.out_len = (op == "copy_e") ? rst.rhs_len : rst.lhs_len;
    if (op == "dot") {
      rst.reduce_size = lhs->shape[lhs->ndim - 1];
      rst.out_len /= rst.reduce_size;
    }
  }
  std::cout << "lhs_len: " << rst.lhs_len << " " <<
    "rhs_len: " << rst.rhs_len << " " <<
    "out_len: " << rst.out_len << " " <<
    "reduce_size: " << rst.reduce_size << std::endl;
  return rst;
}

}  // namespace dgl
