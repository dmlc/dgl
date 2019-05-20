/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/binary_reduce_max.cc
 * \brief CPU kernels for binary reduce max
 */
#include "./binary_reduce_impl.h"
#include "./backward_binary_reduce_impl.h"

namespace dgl {
namespace kernel {

#define REDUCER ReduceMax
#define XPU kDLCPU

}  // namespace kernel
}  // namespace dgl
