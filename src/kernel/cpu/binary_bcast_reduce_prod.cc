/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/binary_bcast_reduce_prod.cc
 * \brief CPU kernels for braodcasting binary reduce prod
 */
#include "./binary_reduce_impl.h"
#include "./backward_binary_reduce_impl.h"

namespace dgl {
namespace kernel {

#define REDUCER ReduceProd
#define XPU kDLCPU

}  // namespace kernel
}  // namespace dgl
