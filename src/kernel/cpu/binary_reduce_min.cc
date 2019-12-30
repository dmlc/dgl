/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/binary_reduce_min.cc
 * \brief CPU kernels for binary reduce min
 */
#include "./binary_reduce_impl.h"
#include "./backward_binary_reduce_impl.h"

namespace dgl {
namespace kernel {

#define REDUCER ReduceMin
#define XPU kDLCPU

#define IDX int32_t
EVAL(GEN_DTYPE, GEN_OP_TARGET, GEN_DEFINE);
EVAL(GEN_BACKWARD_MODE, GEN_DTYPE, GEN_OP_TARGET, GEN_BACKWARD_DEFINE);
#undef IDX

}  // namespace kernel
}  // namespace dgl
