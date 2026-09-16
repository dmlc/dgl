#ifndef CSR_ROW_WISE_SAMPLING_UNIFORM_FUSED_TILING_H
#define CSR_ROW_WISE_SAMPLING_UNIFORM_FUSED_TILING_H

#include <cstdint>

// Tiling layout for the multi-core uniform CSR row-wise fused-sampling
// kernel. The device block is the header below (field order must match
// KernelCsrRowWiseSamplingUniformFused::Init). Twin of
// csr_row_wise_sampling_uniform_tiling.h (ADR-0011: separate files, shared
// design); RNG constants are duplicated by decision until the post-merge
// refactor (see ADR-0011 threshold-break note).
constexpr uint32_t kTilingHeaderWordsFused = 9;

struct CsrRowWiseSamplingUniformFusedTiling {
  uint32_t num_rows;        // number of seed rows to sample
  uint32_t num_samples;     // fanout (0 when select_all)
  uint32_t replace;         // 1 = with replacement
  uint32_t has_data;        // 1 = CSR data array present
  uint32_t seed;            // base RNG seed for this launch
  uint32_t select_all;      // 1 = num_samples == -1 (pick every edge)
  uint32_t num_total_rows;  // total rows of the CSR matrix (bounds check)
  uint32_t ub_available;    // per-core UB budget in bytes (runtime query)
  uint32_t map_seed_nodes;  // 1 = write seed_mapping[rid] = row position
};

// Hardware parameters are queried at runtime via aclrtGetDeviceInfo and
// passed through the tiling block — never hard-coded:
//   - vector-core count (ACL_DEV_ATTR_VECTOR_CORE_NUM): AIV counts differ
//     across SoCs (910B family: 40; other families and trimmed vNPU
//     instances differ)
//   - unified-buffer size (ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE): 192KB on
//     910B, 248KB on 950PR
// The values below are only fallbacks for when the query fails.
constexpr uint32_t kDefaultVectorCoreCountFused = 40;  // fallback, 910B
constexpr uint32_t kDefaultUbBytesFused = 192 * 1024;  // fallback, 910B
constexpr uint32_t kUbReservedBytesFused = 2 * 1024;   // runtime reserved

// Upper bound on launched blocks. The prep kernel stages row_split /
// out_starts in UB tables of kMaxSamplingBlocksFused + 1 entries each
// (single allocation, no dynamic growth), so the host must clamp the
// queried vector-core count to this cap: launching more blocks than the
// tables hold would leave the excess cores reading uninitialized
// boundaries. Past the cap the op degrades gracefully — it simply uses
// fewer blocks than there are cores.
constexpr uint32_t kMaxSamplingBlocksFused = 64;

// RNG constants (xorshift32 with Knuth golden-ratio row hashing) — twin of
// the uniform kernel's constants; keep the two in sync.
constexpr uint32_t kGoldenRatioHashFused = 2654435761u;    // 2^32 / phi
constexpr uint32_t kGoldenRatioOffsetFused = 0x9e3779b9u;  // frac(2^32/phi)
constexpr uint32_t kRngFallbackSeedFused = 0x12345678u;    // nonzero guard

// UB window equation (single source of truth, shared by kernel and host
// launcher — the host re-derives the window to bound the fanout).
// Window instances: 2*depth idx queue + 2*depth data queue + 2*depth out
// queue + pick scratch + 3 output staging buffers. Meta staging
// (out_ptr + seed pairs) is reserved as a fixed byte budget below.
constexpr uint32_t kQueueDepthFused = 2;  // double buffering
constexpr uint32_t kMetaRowsFused = 256;  // meta staging rows per flush
constexpr uint32_t kWindowInstancesFused =
    2 * kQueueDepthFused + 2 * kQueueDepthFused + 2 * kQueueDepthFused + 1 + 3;
constexpr uint32_t kMetaUbReserveFused =
    ((kMetaRowsFused + 1) + 2 * kMetaRowsFused) *
    8;  // out_ptr + interleaved seed pairs, int64 worst case

#endif  // CSR_ROW_WISE_SAMPLING_UNIFORM_FUSED_TILING_H
