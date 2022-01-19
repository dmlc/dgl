 /*!
 *  Copyright (c) 2020 by Contributors
 * \file aten/cpu/gaher_mm.cc
 * \brief GatherMM C APIs and definitions.
 */
#include <dgl/array.h>

namespace dgl {
namespace aten {

/*! \brief Generalized SDDMM on Coo format. */
template <int XPU, typename IdType, int bits>
void gatherMM(const NDArray H,
          const NDArray W,
          NDArray out,
          const NDArray E_per_rel,
          const NDArray etype,
          bool sortedE, bool H_trans, bool w_trans) {
  // SWITCH_BITS(bits, DType, {
  //   // cpu::gatherMM<IdType, DType>(lhs, rhs, out);
  // });
}

template void gatherMM<kDLCPU, int32_t, 16>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray E_per_rel, const NDArray etype,
    bool sortedE, bool H_trans,  bool W_trans);
template void gatherMM<kDLCPU, int64_t, 16>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray E_per_rel, const NDArray etype,
    bool sortedE, bool H_trans,  bool W_trans);
template void gatherMM<kDLCPU, int32_t, 32>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray E_per_rel, const NDArray etype,
    bool sortedE, bool H_trans,  bool W_trans);
template void gatherMM<kDLCPU, int64_t, 32>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray E_per_rel, const NDArray etype,
    bool sortedE, bool H_trans,  bool W_trans);
template void gatherMM<kDLCPU, int32_t, 64>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray E_per_rel, const NDArray etype,
    bool sortedE, bool H_trans,  bool W_trans);
template void gatherMM<kDLCPU, int64_t, 64>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray E_per_rel, const NDArray etype,
    bool sortedE, bool H_trans,  bool W_trans);

}  // namespace aten
}  // namespace dgl
