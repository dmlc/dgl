// UnaryElewise kernel-shared tiling: {uint32 n} only, block layout derived
// in-kernel from n and GetBlockNum() (blockFormer rounded to 512 elements =
// 2048B, keeping GM segments 32B aligned).
#ifndef DGL_ARRAY_ASCEND_UNARY_ELEWISE_TILING_H_
#define DGL_ARRAY_ASCEND_UNARY_ELEWISE_TILING_H_

#include "kernel_operator.h"

// Core-block alignment: 512 elements = 2048B, keeps every GM segment start
// (and every full-tile DataCopy) 32B aligned on the vectorized path.
constexpr uint32_t kUnaryBlockAlignElems = 512;

// Core-block range: per_core = ceil(n / block_num), rounded up to a multiple
// of kUnaryBlockAlignElems (blockFormer); leading cores get equal blocks, the
// last core carries the remainder capped at n, cores with start >= n idle.
// Intermediates use uint64 so n near the uint32 tiling limit cannot overflow.
__aicore__ inline void UnaryBlockRange(uint32_t n, uint32_t& start,
                                       uint32_t& end) {
  const uint32_t block_id = AscendC::GetBlockIdx();
  const uint32_t block_num = AscendC::GetBlockNum();
  const uint32_t per_core = static_cast<uint32_t>(
      (static_cast<uint64_t>(n) + block_num - 1) / block_num);
  const uint32_t block_former =
      ((per_core + kUnaryBlockAlignElems - 1) / kUnaryBlockAlignElems) *
      kUnaryBlockAlignElems;
  start = block_id * block_former;
  end = static_cast<uint32_t>(
      (static_cast<uint64_t>(start) + block_former < static_cast<uint64_t>(n))
          ? static_cast<uint64_t>(start) + block_former
          : static_cast<uint64_t>(n));
}

#endif  // DGL_ARRAY_ASCEND_UNARY_ELEWISE_TILING_H_
