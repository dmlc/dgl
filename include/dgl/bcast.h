/**
 *  Copyright (c) 2020 by Contributors
 * @file dgl/aten/bcast.h
 * @brief Broadcast related function C++ header.
 */
#ifndef DGL_BCAST_H_
#define DGL_BCAST_H_

#include <string>
#include <vector>

#include "./runtime/ndarray.h"

using namespace dgl::runtime;
namespace dgl {

/**
 * @brief Broadcast offsets and auxiliary information.
 */
struct BcastOff {
  /**
   * @brief offset vector of lhs operand and rhs operand.
   * @note lhs_offset[i] indicates the start position of the scalar
   *       in lhs operand that required to compute the i-th element
   *       in the output, likewise for rhs_offset.
   *
   * For example, when lhs array has shape (1, 3) and rhs array
   * has shape (5, 1), the resulting array would have shape (5, 3),
   * then both lhs_offset and rhs_offset would contain 15 elements.
   *
   * lhs_offset: 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2
   * rhs_offset: 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4
   *
   * in order to compute the 7-th (row 2, column 0) element in the output,
   * we need the 0-th element in the lhs array and the 2-th element in the
   * rhs array.
   */
  std::vector<int64_t> lhs_offset, rhs_offset;
  /** @brief Whether broadcast is required or not. */
  bool use_bcast;
  /**
   * @brief Auxiliary information for kernel computation
   * @note lhs_len refers to the left hand side operand length.
   *       e.g. 15 for shape (1, 3, 5)
   *       rhs_len refers to the right hand side operand length.
   *       e.g. 15 for shape (3, 1, 5)
   *       out_len refers to the output length.
   *       e.g. 45 for shape (3, 3, 5)
   *       reduce_size refers to the reduction size (for op like dot).
   *       e.g. 1 for add, 5 for dot and lhs_shape,rhs_shape=(3,5)
   */
  int64_t lhs_len, rhs_len, out_len, reduce_size;
};

/**
 * @brief: Compute broadcast and auxiliary information given operator
 *         and operands for kernel computation.
 * @param op: a string indicates the operator, could be `add`, `sub`,
 *        `mul`, `div`, `dot`, 'copy_u`, `copy_e`.
 * @param lhs The left hand side operand of NDArray class.
 * @param rhs The right hand side operand of NDArray class.
 * @return the broadcast information of BcastOff class.
 */
BcastOff CalcBcastOff(const std::string& op, NDArray lhs, NDArray rhs);

}  // namespace dgl

#endif  // DGL_BCAST_H_
