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

GEN_DEFINE(float, SelectSrc, SelectDst, BinaryAdd)
GEN_DEFINE(float, SelectSrc, SelectDst, BinarySub)
GEN_DEFINE(float, SelectSrc, SelectDst, BinaryMul)
GEN_DEFINE(float, SelectSrc, SelectDst, BinaryDiv)
GEN_DEFINE(float, SelectSrc, SelectDst, BinaryUseLhs)

}  // namespace kernel
}  // namespace dgl
