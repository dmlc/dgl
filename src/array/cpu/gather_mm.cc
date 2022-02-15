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


/*! \brief Generalized segmentMM. */
template <int XPU, typename IdType, int bits>
void segmentMM(const NDArray A,
          const NDArray B,
          NDArray C,
          const NDArray seglen_A,
          bool a_trans, bool b_trans) {
    SWITCH_BITS(bits, DType, {
        LOG(FATAL) << "Unsupported CPU kernel for SegmentMM.";
  });
}

/*! \brief Generalized GatherMM. */
template <int XPU, typename IdType, int bits>
void gatherMM(const NDArray A,
          const NDArray B,
          NDArray C,
          const NDArray idx_a,
          const NDArray idx_b,
          const int num_rel) {
    SWITCH_BITS(bits, DType, {
        LOG(FATAL) << "Unsupported CPU kernel for GatherMM.";
  });
}

/*! \brief Generalized GatherMM_scatter. */
template <int XPU, typename IdType, int bits>
void gatherMM_scatter(const NDArray A,
          const NDArray B,
          NDArray C,
          const NDArray idx_a,
          const NDArray idx_b,
          const NDArray idx_c,
          const int num_rel,
          bool a_trans, bool b_trans) {
    SWITCH_BITS(bits, DType, {
        LOG(FATAL) << "Unsupported CPU kernel for GatherMM.";
  });
}

template void gatherMM<kDLCPU, int32_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const int num_rel);
template void gatherMM<kDLCPU, int64_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const int num_rel);
template void gatherMM<kDLCPU, int32_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const int num_rel);
template void gatherMM<kDLCPU, int64_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const int num_rel);
template void gatherMM<kDLCPU, int32_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const int num_rel);
template void gatherMM<kDLCPU, int64_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const int num_rel);

template void gatherMM_scatter<kDLCPU, int32_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c,
    const int num_rel, bool a_trans, bool b_trans);
template void gatherMM_scatter<kDLCPU, int64_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c,
    const int num_rel, bool a_trans, bool b_trans);
template void gatherMM_scatter<kDLCPU, int32_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c,
    const int num_rel, bool a_trans, bool b_trans);
template void gatherMM_scatter<kDLCPU, int64_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c,
    const int num_rel, bool a_trans, bool b_trans);
template void gatherMM_scatter<kDLCPU, int32_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c,
    const int num_rel, bool a_trans, bool b_trans);
template void gatherMM_scatter<kDLCPU, int64_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c,
    const int num_rel, bool a_trans, bool b_trans);

template void segmentMM<kDLCPU, int32_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void segmentMM<kDLCPU, int64_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void segmentMM<kDLCPU, int32_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void segmentMM<kDLCPU, int64_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void segmentMM<kDLCPU, int32_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void segmentMM<kDLCPU, int64_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);

}  // namespace aten
}  // namespace dgl
