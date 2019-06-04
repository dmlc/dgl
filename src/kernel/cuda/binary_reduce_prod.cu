/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/binary_reduce_prod.cu
 * \brief CUDA kernels for binary reduce prod
 */
#include "./binary_reduce_impl.cuh"
#include "./backward_binary_reduce_impl.cuh"

namespace dgl {
namespace kernel {

#define REDUCER ReduceProd
#define XPU kDLGPU
#define IDX int32_t

EVAL(GEN_DTYPE, GEN_OP_TARGET, GEN_DEFINE)
EVAL(GEN_BACKWARD_MODE, GEN_DTYPE, GEN_OP_TARGET, GEN_BACKWARD_DEFINE)

}  // namespace kernel
}  // namespace dgl
