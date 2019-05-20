/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/binary_bcast_reduce_sum.cu
 * \brief CUDA kernels for braodcasting binary reduce sum
 */
#include "./binary_reduce_impl.cuh"
#include "./backward_binary_reduce_impl.cuh"

namespace dgl {
namespace kernel {
namespace cuda {
}  // namespace cuda

#define REDUCER ReduceSum
#define XPU kDLGPU

}  // namespace kernel
}  // namespace dgl
