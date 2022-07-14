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


/*! \brief Generalized SegmentMM. */
template <int XPU, typename IdType, int bits>
void SegmentMM(const NDArray A,
          const NDArray B,
          NDArray C,
          const NDArray seglen_A,
          bool a_trans, bool b_trans) {
    LOG(FATAL) << "Unsupported CPU kernel for SegmentMM.";
}

template <int XPU, typename IdType, int bits>
void SegmentMMBackwardB(const NDArray A,
                        const NDArray dC,
                        NDArray dB,
                        const NDArray seglen) {
    LOG(FATAL) << "Unsupported CPU kernel for SegmentMMBackwardB.";
}

/*! \brief Generalized GatherMM. */
template <int XPU, typename IdType, int bits>
void GatherMM(const NDArray A,
          const NDArray B,
          NDArray C,
          const NDArray idx_a,
          const NDArray idx_b) {
    LOG(FATAL) << "Unsupported CPU kernel for GatherMM.";
}

/*! \brief Generalized GatherMM_scatter. */
template <int XPU, typename IdType, int bits>
void GatherMMScatter(const NDArray A,
          const NDArray B,
          NDArray C,
          const NDArray idx_a,
          const NDArray idx_b,
          const NDArray idx_c) {
    LOG(FATAL) << "Unsupported CPU kernel for GatherMM.";
}

template void GatherMM<kDLCPU, int32_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b);
template void GatherMM<kDLCPU, int64_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b);
template void GatherMM<kDLCPU, int32_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b);
template void GatherMM<kDLCPU, int64_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b);
template void GatherMM<kDLCPU, int32_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b);
template void GatherMM<kDLCPU, int64_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b);

template void GatherMMScatter<kDLCPU, int32_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c);
template void GatherMMScatter<kDLCPU, int64_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c);
template void GatherMMScatter<kDLCPU, int32_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c);
template void GatherMMScatter<kDLCPU, int64_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c);
template void GatherMMScatter<kDLCPU, int32_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c);
template void GatherMMScatter<kDLCPU, int64_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c);

template void SegmentMM<kDLCPU, int32_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void SegmentMM<kDLCPU, int64_t, 16>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void SegmentMM<kDLCPU, int32_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void SegmentMM<kDLCPU, int64_t, 32>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void SegmentMM<kDLCPU, int32_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);
template void SegmentMM<kDLCPU, int64_t, 64>(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray seglen_A, bool a_trans, bool b_trans);

template void SegmentMMBackwardB<kDLCPU, int32_t, 16>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);
template void SegmentMMBackwardB<kDLCPU, int64_t, 16>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);
template void SegmentMMBackwardB<kDLCPU, int32_t, 32>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);
template void SegmentMMBackwardB<kDLCPU, int64_t, 32>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);
template void SegmentMMBackwardB<kDLCPU, int32_t, 64>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);
template void SegmentMMBackwardB<kDLCPU, int64_t, 64>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);

}  // namespace aten
}  // namespace dgl
