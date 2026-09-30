/**
 * Copyright (c) 2024 by Contributors
 * @file full_tiling.h
 * @brief Tiling block shared by the AscendC Full host wrapper (full.cc) and
 * the device kernels (full_kernel.cpp).
 *
 * The struct is passed to the kernel launch entry **by value** (the AscendC
 * packaging flow marshals scalar/POD params automatically), so no device
 * tiling buffer and no GlobalTensor scalar reads are needed on the device
 * side. Field layout is part of the host<->device ABI.
 */
#ifndef DGL_ARRAY_ASCEND_FULL_TILING_H_
#define DGL_ARRAY_ASCEND_FULL_TILING_H_

#include <cstdint>

struct FullTilingData {
  uint64_t n;    ///< element count. 64-bit so that arrays larger than
                 ///< 2^32-1 elements cannot silently truncate (review fix).
  uint64_t val;  ///< fill value as a bit pattern. Host side bit-casts the
                 ///< typed value (int32/float use the low 32 bits).
};

static_assert(sizeof(FullTilingData) == 16, "FullTilingData ABI layout");

// Elements per UB stamp tile. The stamp buffer costs kFullTileLength *
// sizeof(T) UB bytes (worst case 64 KB for 8-byte dtypes), well within the
// 192 KB per-core UB of the 910B (DAV_2201) family.
constexpr uint32_t kFullTileLength = 8192;

#endif  // DGL_ARRAY_ASCEND_FULL_TILING_H_
