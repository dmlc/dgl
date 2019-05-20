/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/binary_reduce_max.cu
 * \brief CUDA kernels for binary reduce max
 */
#include "./binary_reduce_impl.cuh"
#include "./backward_binary_reduce_impl.cuh"

namespace dgl {
namespace kernel {

#define REDUCER ReduceMax
#define XPU kDLGPU

}  // namespace kernel
}  // namespace dgl
