#pragma once

#include <cstdint>

constexpr uint32_t CONCAT_UB_TOTAL_SIZE = 192 * 1024;
constexpr uint32_t CONCAT_UB_RESERVED = 2 * 1024;
constexpr uint32_t CONCAT_BUFFER_NUM = 2;
constexpr uint32_t CONCAT_ALIGN_BYTES = 32;
constexpr uint32_t CONCAT_CHUNK_BYTES =
    ((CONCAT_UB_TOTAL_SIZE - CONCAT_UB_RESERVED) / CONCAT_BUFFER_NUM) &
    ~(CONCAT_ALIGN_BYTES - 1);

struct ConcatTilingData {
  uint32_t inputCount;
  uint32_t segCount;
  uint32_t dstStride;
  uint32_t chunkElems;
  uint32_t totalChunks;
  uint32_t chunksPerCore;
};

struct ConcatInputDesc {
  uint64_t src;
  uint32_t segElems;
  uint32_t dstBase;
  uint32_t chunksPerSeg;
  uint32_t chunkBase;
};
