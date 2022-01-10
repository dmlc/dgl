/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/sddmm.cu
 * \brief SDDMM C APIs and definitions.
 */
#include <dgl/array.h>
#include "./functor.cuh"

namespace dgl {
namespace aten {


// namespace cublas {

// void gatherMM() {

// }

// }  // namespace cublas
/*!
 * \brief CUDA implementation of g-SDDMM on Coo format.
 */
template <int XPU, int bits>
void gatherMM(NDArray h,
              const std::vector<NDArray>& wdict,
              NDArray out) {
  // SWITCH_BITS(bits, DType, {
  //   // cublas::gatherMM<IdType, DType>(lhs, rhs, out);
  // });
}

template void gatherMM<kDLGPU, 16>(
    NDArray h, const std::vector<NDArray>& wdict, NDArray out);
template void gatherMM<kDLGPU, 32>(
    NDArray h, const std::vector<NDArray>& wdict, NDArray out);
template void gatherMM<kDLGPU, 64>(
    NDArray h, const std::vector<NDArray>& wdict, NDArray out);

}  // namespace aten
}  // namespace dgl
