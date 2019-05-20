/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/binary_bcast_reduce_none.cu
 * \brief CUDA kernels for braodcasting binary reduce none
 */
#include "./binary_reduce_impl.cuh"
#include "./backward_binary_reduce_impl.cuh"

namespace dgl {
namespace kernel {
namespace cuda {
}  // namespace cuda

#define REDUCER ReduceNone
#define XPU kDLGPU
}  // namespace kernel
}  // namespace dgl
