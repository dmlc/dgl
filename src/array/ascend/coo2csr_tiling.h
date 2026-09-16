/**
 * Copyright (c) 2024 by Contributors
 * @file coo2csr_tiling.h
 * @brief Tiling block for the COO->CSR counting-sort kernels.
 */
#ifndef COO2CSR_TILING_H
#define COO2CSR_TILING_H

#include <cstdint>

// Field order must match KernelCooToCsrCount / KernelCooToCsrScatter Init.
struct CooToCsrTiling {
  int64_t nnz;            // number of edges (row array length)
  uint32_t num_rows;      // rows of the COO matrix (indptr length - 1)
  uint32_t ub_available;  // per-core UB budget in bytes (runtime query)
  uint32_t has_data;      // 1 = explicit COO data array present
};

// Hardware parameters are queried at runtime via aclrtGetDeviceInfo and
// passed through the tiling block — never hard-coded (fallbacks only).
constexpr uint32_t kDefaultVectorCoreCount = 40;  // 910B family
constexpr uint32_t kDefaultUbBytes = 192 * 1024;  // 910B family
constexpr uint32_t kUbReservedBytes = 2 * 1024;   // runtime reserved tail

// The counting workspace has num_rows histogram words plus, per block,
// two reduction words (min/max row seen). The per-block reductions live
// in their own cache-line-aligned slab so no two blocks ever scalar-write
// the same 64B line (DataCacheCleanAndInvalid doc example 3).
constexpr uint32_t kReduceWordsPerBlock = 2;  // [min, max]
constexpr uint32_t kCacheLineBytes = 64;
// Sentinel pair written by blocks whose row range saw no edge (host
// treats min == kReduceEmpty as "skip this block's reduction").
constexpr uint32_t kReduceEmpty = 0xFFFFFFFFu;

#endif  // COO2CSR_TILING_H
