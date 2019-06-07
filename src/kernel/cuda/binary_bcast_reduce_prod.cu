/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/binary_bcast_reduce_prod.cu
 * \brief CUDA kernels for braodcasting binary reduce prod
 */
#include "./binary_reduce_impl.cuh"
#include "./backward_binary_reduce_impl.cuh"

namespace dgl {
namespace kernel {
namespace cuda {
}  // namespace cuda

#define REDUCER ReduceProd
#define XPU kDLGPU
#define IDX int32_t
EVAL(GEN_NDIM, GEN_DTYPE, GEN_OP_TARGET, GEN_BCAST_DEFINE)
EVAL(GEN_BACKWARD_MODE, GEN_NDIM, GEN_DTYPE, GEN_OP_TARGET,
     GEN_BACKWARD_BCAST_DEFINE);

}  // namespace kernel
}  // namespace dgl
