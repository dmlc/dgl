 
/*!
 *  Copyright (c) 2020 by Contributors
 * \file aten/cpu/gaher_mm.cc
 * \brief GatherMM C APIs and definitions.
 */
#include "./sddmm.h"
#include <dgl/array.h>

namespace dgl {
namespace aten {

/*! \brief Generalized SDDMM on Coo format. */
template <int XPU, typename IdType, int bits>
void gatherMM(const NDArray E_etype,
              const NDArray h,
              const NDArray w,
              NDArray out) {
  // SWITCH_BITS(bits, DType, {
  //   // cpu::gatherMM<IdType, DType>(lhs, rhs, out);
  // });
}

template void gatherMM<kDLCPU, int32_t, 16>(
    const NDArray E_etype, const NDArray h,
    const NDArray w, NDArray out);
template void gatherMM<kDLCPU, int64_t, 16>(
    const NDArray E_etype, const NDArray h,
    const NDArray w, NDArray out);
template void gatherMM<kDLCPU, int32_t, 32>(
    const NDArray E_etype, const NDArray h,
    const NDArray w, NDArray out);
template void gatherMM<kDLCPU, int64_t, 32>(
    const NDArray E_etype, const NDArray h,
    const NDArray w, NDArray out);
template void gatherMM<kDLCPU, int32_t, 64>(
    const NDArray E_etype, const NDArray h,
    const NDArray w, NDArray out);
template void gatherMM<kDLCPU, int64_t, 64>(
    const NDArray E_etype, const NDArray h,
    const NDArray w, NDArray out);

}  // namespace aten
}  // namespace dgl
