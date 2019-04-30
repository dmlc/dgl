#include "./binary_reduce_impl.cuh"
#include "./backward_binary_reduce_impl.cuh"

namespace dgl {
namespace kernel {
namespace cuda {

#define REDUCER ReduceProd
#define XPU kDLGPU
EVAL(GEN_NDIM, GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP, GEN_BCAST_DEFINE)
EVAL(GEN_BACKWARD_MODE, GEN_NDIM, GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP,
     GEN_BACKWARD_BCAST_DEFINE);

}  // namespace cuda
}  // namespace kernel
}  // namespace dgl
