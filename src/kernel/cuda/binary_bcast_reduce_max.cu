/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/binary_bcast_reduce_max.cu
 * \brief CUDA kernels for braodcasting binary reduce max
 */
#include "./binary_reduce_impl.cuh"
#include "./backward_binary_reduce_impl.cuh"

namespace dgl {
namespace kernel {
namespace cuda {
}  // namespace cuda

#define REDUCER ReduceMax
#define XPU kDLGPU
EVAL(GEN_NDIM, GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP, GEN_BCAST_DEFINE)
EVAL(GEN_BACKWARD_MODE, GEN_NDIM, GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP,
     GEN_BACKWARD_BCAST_DEFINE);

}  // namespace kernel
}  // namespace dgl
