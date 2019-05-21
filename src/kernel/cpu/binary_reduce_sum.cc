/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/binary_reduce_sum.cc
 * \brief CPU kernels for binary reduce sum
 */
#include "./binary_reduce_impl.h"
#include "./backward_binary_reduce_impl.h"

namespace dgl {
namespace kernel {

#define REDUCER ReduceSum
#define XPU kDLCPU

GEN_BINARY_OP(GEN_DEFINE, float, SelectSrc, SelectDst)

}  // namespace kernel
}  // namespace dgl
