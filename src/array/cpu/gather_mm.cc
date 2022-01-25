 /*!
 *  Copyright (c) 2020 by Contributors
 * \file kernel/cpu/gaher_mm.cc
 * \brief GatherMM C APIs and definitions.
 */
#include "./gather_mm.h"
#include <dgl/array.h>

namespace dgl {
namespace aten {

#define SWITCH_BITS(bits, DType, ...)                           \
  do {                                                          \
    if ((bits) == 16 || (bits) == 32) {                         \
      typedef float DType;                                      \
      { __VA_ARGS__ }                                           \
    } else if ((bits) == 64) {                                  \
      typedef double DType;                                     \
      { __VA_ARGS__ }                                           \
    } else {                                                    \
      LOG(FATAL) << "Data type not recognized with bits " << bits; \
    }                                                           \
  } while (0)

/*! \brief Generalized SDDMM on Coo format. */
template <int XPU, typename IdType, int bits>
void gatherMM(const NDArray H,
          const NDArray W,
          NDArray out,
          const NDArray H_per_rel,
          const NDArray W_per_rel,
          const NDArray etype,
          bool sortedE, bool H_trans, bool W_trans) {
    SWITCH_BITS(bits, DType, {
        // cpu::gatherMM<IdType, DType>(lhs, rhs, out);
        if (sortedE)  // similar to low-mem matmul
            cpu::gatherMM_SortedEtype<XPU, IdType, DType>(H, W, out, H_per_rel,
                W_per_rel, H_trans, W_trans);
  });
}

template void gatherMM<kDLCPU, int32_t, 16>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray H_per_rel, const NDArray W_per_rel,
    const NDArray etype, bool sortedE, bool H_trans, bool W_trans);
template void gatherMM<kDLCPU, int64_t, 16>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray H_per_rel, const NDArray W_per_rel,
    const NDArray etype, bool sortedE, bool H_trans, bool W_trans);
template void gatherMM<kDLCPU, int32_t, 32>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray H_per_rel, const NDArray W_per_rel,
    const NDArray etype, bool sortedE, bool H_trans, bool W_trans);
template void gatherMM<kDLCPU, int64_t, 32>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray H_per_rel, const NDArray W_per_rel,
    const NDArray etype, bool sortedE, bool H_trans, bool W_trans);
template void gatherMM<kDLCPU, int32_t, 64>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray H_per_rel, const NDArray W_per_rel,
    const NDArray etype, bool sortedE, bool H_trans, bool W_trans);
template void gatherMM<kDLCPU, int64_t, 64>(
    const NDArray h, const NDArray w, NDArray out,
    const NDArray H_per_rel, const NDArray W_per_rel,
    const NDArray etype, bool sortedE, bool H_trans, bool W_trans);

}  // namespace aten
}  // namespace dgl
