#ifndef CSR_ROW_WISE_SAMPLING_UNIFORM_TILING_H
#define CSR_ROW_WISE_SAMPLING_UNIFORM_TILING_H

#include <cstdint>

// Tiling layout for the multi-core uniform CSR row-wise sampling kernel.
// The device block is: header (kTilingHeaderWords words, field order below)
// followed by one output-start word per launched block (block prefix sums).
// Field order must match KernelCsrRowWiseSamplingUniform::Init.
constexpr uint32_t kTilingHeaderWords = 8;

struct CsrRowWiseSamplingUniformTiling {
  uint32_t num_rows;        // number of seed rows to sample
  uint32_t num_samples;     // fanout (0 when select_all)
  uint32_t replace;         // 1 = with replacement
  uint32_t has_data;        // 1 = CSR data array present
  uint32_t seed;            // base RNG seed for this launch
  uint32_t select_all;      // 1 = num_samples == -1 (pick every edge)
  uint32_t num_total_rows;  // total rows of the CSR matrix (bounds check)
  uint32_t ub_available;    // per-core UB budget in bytes (runtime query)
};

// Hardware parameters are queried at runtime via aclrtGetDeviceInfo and
// passed through the tiling block — never hard-coded:
//   - vector-core count (ACL_DEV_ATTR_VECTOR_CORE_NUM): AIV counts differ
//     across SoCs (910B family: 40; other families and trimmed vNPU
//     instances differ)
//   - unified-buffer size (ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE): 192KB on
//     910B, 248KB on 950PR
// The values below are only fallbacks for when the query fails.
constexpr uint32_t kDefaultVectorCoreCount = 40;  // fallback, 910B family
constexpr uint32_t kDefaultUbBytes = 192 * 1024;  // fallback, 910B family
constexpr uint32_t kUbReservedBytes = 2 * 1024;   // runtime reserved tail

// Per-row scratch: sampled local indices are staged in UB before the final
// gather, sized for the worst case (fanout picks per row).

// RNG constants (xorshift32 with Knuth golden-ratio row hashing).
constexpr uint32_t kGoldenRatioHash = 2654435761u;    // 2^32 / phi
constexpr uint32_t kGoldenRatioOffset = 0x9e3779b9u;  // frac(2^32 / phi)
constexpr uint32_t kRngFallbackSeed = 0x12345678u;    // nonzero-state guard

#endif  // CSR_ROW_WISE_SAMPLING_UNIFORM_TILING_H
