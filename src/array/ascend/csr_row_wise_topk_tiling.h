#ifndef CSR_ROW_WISE_TOPK_TILING_H
#define CSR_ROW_WISE_TOPK_TILING_H

#include <cstdint>

// Tiling layout for the multi-core CSR row-wise topk kernel.
// Field order must match KernelCsrRowWiseTopk::Init.
constexpr uint32_t kTilingHeaderWords = 8;

struct CsrRowWiseTopkTiling {
  uint32_t num_rows;        // number of seed rows to pick from
  uint32_t k;               // top-k (0 when select_all)
  uint32_t select_all;      // 1 = k == -1 (pick every edge, weight order)
  uint32_t ascending;       // 1 = pick the k smallest (tail of sort)
  uint32_t has_data;        // 1 = CSR data array present (eid mapping)
  uint32_t num_total_rows;  // total rows of the CSR matrix (bounds check)
  uint32_t ub_available;    // per-core UB budget in bytes (runtime query)
  uint32_t reserved;        // padding (must stay 8 words)
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

// One Sort call covers repeatTimes * 32 elements with repeatTimes capped at
// 255; the window formula must stay below this bound (launcher CHECK).
constexpr uint32_t kSortMaxElemsPerCall = 255 * 32;

#endif  // CSR_ROW_WISE_TOPK_TILING_H
