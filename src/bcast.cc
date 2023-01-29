/**
 *  Copyright (c) 2020 by Contributors
 * @file kernel/bcast.h
 * @brief Broadcast related function implementations.
 */
#include <dgl/bcast.h>
#include <dmlc/logging.h>

#include <algorithm>

namespace dgl {

namespace {
/**
 * @brief Determine whether use broadcasting or not, given the operator
 *        type, lhs array and rhs array.
 */
bool UseBcast(const std::string& op, NDArray lhs, NDArray rhs) {
  if (op == "copy_lhs" || op == "copy_rhs")
    return false;  // broadcasting is not required for copy_u/copy_e
  if (lhs->ndim != rhs->ndim) return true;
  for (int i = 1; i < lhs->ndim; ++i) {
    if (lhs->shape[i] != rhs->shape[i]) return true;
  }
  return false;
}

}  // namespace

/**
 * @brief: Compute broadcast and auxiliary information given operator
 *         and operands for kernel computation.
 * @note: Expect lhs, rhs to have ndim >= 2 and the shape of lhs/rhs
 *        valid for the op computation.
 */
BcastOff CalcBcastOff(const std::string& op, NDArray lhs, NDArray rhs) {
  BcastOff rst;
  rst.lhs_len = 1;
  rst.rhs_len = 1;
  for (int i = 1; i < lhs->ndim; ++i) rst.lhs_len *= lhs->shape[i];
  for (int i = 1; i < rhs->ndim; ++i) rst.rhs_len *= rhs->shape[i];
  rst.use_bcast = UseBcast(op, lhs, rhs);
  rst.reduce_size = 1;  // defaults to 1, except for the case op == 'dot'.
  if (rst.use_bcast) {
    const int max_ndim = std::max(lhs->ndim, rhs->ndim) - 1;
    int out_len = 1, j = 0;
    if (op == "dot") {
      rst.reduce_size = lhs->shape[lhs->ndim - 1];  // set reduce_size for dot.
      ++j;  // do not consider reduce axis in computing lhs_offset and
            // rhs_offset.
    }
    int stride_l = 1, stride_r = 1;
    rst.lhs_offset.push_back(0);  // lhs_offset[0] is always 0
    rst.rhs_offset.push_back(0);  // rhs_offset[0] is always 0
    for (; j < max_ndim; ++j) {   // iterate the axis from back to front.
      // dl refers to the size of lhs array in the current axis, likewise for
      // dr.
      const int dl =
          (lhs->ndim - 1 - j < 1) ? 1 : lhs->shape[lhs->ndim - 1 - j];
      const int dr =
          (rhs->ndim - 1 - j < 1) ? 1 : rhs->shape[rhs->ndim - 1 - j];
      for (int i = 1; i < std::max(dl, dr); ++i) {
        for (int k = 0; k < out_len; ++k) {
          /* Explaination:
           * if current dimension is not broadcast dimension for lhs array
           *   lhs_offset[i * out_len + k] = lhs_offset[k] + i * stride_l
           * else
           *   lhs_offset[i * out_len + k] = lhs_offset[k]
           * likewise for rhs_offset.
           */
          rst.lhs_offset.push_back(rst.lhs_offset[k] + i * (i < dl) * stride_l);
          rst.rhs_offset.push_back(rst.rhs_offset[k] + i * (i < dr) * stride_r);
        }
      }
      out_len *= std::max(dl, dr);
      stride_l *= dl;
      stride_r *= dr;
    }
    rst.out_len = out_len;
  } else {
    rst.out_len = (op == "copy_rhs") ? rst.rhs_len : rst.lhs_len;
    if (op == "dot") {
      // set reduce_size for dot.
      rst.reduce_size = lhs->shape[lhs->ndim - 1];
      // out_len is divied by reduce_size in dot.
      rst.out_len /= rst.reduce_size;
    }
  }
  return rst;
}

}  // namespace dgl
