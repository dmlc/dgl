/**
 *  Copyright (c) 2020 by Contributors
 * @file kernel/cpu/gaher_mm.cc
 * @brief GatherMM C APIs and definitions.
 */
#include "./gather_mm.h"

#include <dgl/array.h>

namespace dgl {
namespace aten {

/** @brief Generalized SegmentMM. */
template <int XPU, typename IdType, typename DType>
void SegmentMM(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans) {
  LOG(FATAL) << "Unsupported CPU kernel for SegmentMM.";
}

template <int XPU, typename IdType, typename DType>
void SegmentMMBackwardB(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen) {
  LOG(FATAL) << "Unsupported CPU kernel for SegmentMMBackwardB.";
}

/** @brief Generalized GatherMM. */
template <int XPU, typename IdType, typename DType>
void GatherMM(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b) {
  LOG(FATAL) << "Unsupported CPU kernel for GatherMM.";
}

/** @brief Generalized GatherMM_scatter. */
template <int XPU, typename IdType, typename DType>
void GatherMMScatter(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c) {
  LOG(FATAL) << "Unsupported CPU kernel for GatherMM.";
}

template void GatherMM<kDGLCPU, int32_t, BFloat16>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b);
template void GatherMM<kDGLCPU, int64_t, BFloat16>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b);
template void GatherMM<kDGLCPU, int32_t, float>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b);
template void GatherMM<kDGLCPU, int64_t, float>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b);
template void GatherMM<kDGLCPU, int32_t, double>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b);
template void GatherMM<kDGLCPU, int64_t, double>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b);

template void GatherMMScatter<kDGLCPU, int32_t, BFloat16>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c);
template void GatherMMScatter<kDGLCPU, int64_t, BFloat16>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c);
template void GatherMMScatter<kDGLCPU, int32_t, float>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c);
template void GatherMMScatter<kDGLCPU, int64_t, float>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c);
template void GatherMMScatter<kDGLCPU, int32_t, double>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c);
template void GatherMMScatter<kDGLCPU, int64_t, double>(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c);

template void SegmentMM<kDGLCPU, int32_t, BFloat16>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans);
template void SegmentMM<kDGLCPU, int64_t, BFloat16>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans);
template void SegmentMM<kDGLCPU, int32_t, float>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans);
template void SegmentMM<kDGLCPU, int64_t, float>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans);
template void SegmentMM<kDGLCPU, int32_t, double>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans);
template void SegmentMM<kDGLCPU, int64_t, double>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans);

template void SegmentMMBackwardB<kDGLCPU, int32_t, BFloat16>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);
template void SegmentMMBackwardB<kDGLCPU, int64_t, BFloat16>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);
template void SegmentMMBackwardB<kDGLCPU, int32_t, float>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);
template void SegmentMMBackwardB<kDGLCPU, int64_t, float>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);
template void SegmentMMBackwardB<kDGLCPU, int32_t, double>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);
template void SegmentMMBackwardB<kDGLCPU, int64_t, double>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);

}  // namespace aten
}  // namespace dgl
