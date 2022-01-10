 
/*!
 *  Copyright (c) 2020 by Contributors
 * \file aten/cpu/sddmm.cc
 * \brief SDDMM C APIs and definitions.
 */
#include "./sddmm.h"
#include <dgl/array.h>

namespace dgl {
namespace aten {

/*! \brief Generalized SDDMM on Coo format. */
template <int XPU, int bits>
void gatherMM(NDArray h,
              const std::vector<NDArray>& wdict,
              NDArray out) {
  // SWITCH_BITS(bits, DType, {
  //   // cpu::gatherMM<IdType, DType>(lhs, rhs, out);
  // });
}

template void gatherMM<kDLCPU, 16>(
    NDArray h, const std::vector<NDArray>& wdict, NDArray out);
template void gatherMM<kDLCPU, 32>(
    NDArray h, const std::vector<NDArray>& wdict, NDArray out);
template void gatherMM<kDLCPU, 64>(
    NDArray h, const std::vector<NDArray>& wdict, NDArray out);

}  // namespace aten
}  // namespace dgl
