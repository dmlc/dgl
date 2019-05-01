/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/binary_bcast_reduce_max.cc
 * \brief CPU kernels for braodcasting binary reduce max
 */
#include "./binary_reduce_impl.h"
#include "./backward_binary_reduce_impl.h"

namespace dgl {
namespace kernel {

#define REDUCER ReduceMax
#define XPU kDLCPU

EVAL(GEN_NDIM, GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP, GEN_BCAST_DEFINE);
EVAL(GEN_BACKWARD_MODE, GEN_NDIM, GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP,
     GEN_BACKWARD_BCAST_DEFINE);

}  // namespace kernel
}  // namespace dgl
