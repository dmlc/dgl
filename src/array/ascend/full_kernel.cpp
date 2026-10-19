#include "kernel_operator.h"

#include "full_tiling.h"

using namespace AscendC;

// AscendC Full kernel — fills a 1-D GM buffer with a scalar value.
//
// Design (addresses PR #37 review comments):
//   1. Multi-core: each AIV core fills a contiguous element range. Ranges are
//      derived on device from GetBlockIdx()/GetBlockNum() (blockDim is
//      computed on host from the runtime Vector Core count, never hardcoded).
//   2. Stamp pattern: the tile content is identical for every tile, so the
//      UB stamp is built ONCE per core and then streamed out repeatedly —
//      no per-tile Duplicate, no TQue, no double buffering (the previous
//      TQue<VECOUT, 2> pipeline never actually overlapped).
//   3. No scalar GM access: GlobalTensor::SetValue/GetValue loops are
//      restricted to debug usage; all GM traffic goes through MTE3
//      (DataCopyPad). This also covers the 64-bit dtypes.
//
// Stamp construction per dtype:
//   - 32-bit (int32/float): Duplicate<T> — supported on DAV_2201 for
//     int16/uint16/int32/uint32/float.
//   - 64-bit (int64/double): Duplicate<T> is a static_assert failure on
//     DAV_2201 (verified against CANN 9.0 headers), so the 8-byte pattern is
//     seeded as 32 bytes of interleaved uint32 (lo, hi, lo, hi, ...) and
//     amplified to a full tile with log2-stage UB->UB DataCopy doubling.

namespace {

// Build the kFullTileLength-element UB stamp for `bits`.
template <typename T>
__aicore__ inline void BuildStamp(LocalTensor<T> stamp, uint64_t bits) {
  if constexpr (sizeof(T) <= sizeof(uint32_t)) {
    // Bit-cast the low 32 bits to T (two's complement / IEEE pattern kept).
    union {
      uint32_t u;
      T t;
    } caster;
    caster.u = static_cast<uint32_t>(bits);
    Duplicate<T>(stamp, caster.t, kFullTileLength);
  } else {
    LocalTensor<uint32_t> u32 = stamp.template ReinterpretCast<uint32_t>();
    const uint32_t lo = static_cast<uint32_t>(bits & 0xFFFFFFFFULL);
    const uint32_t hi = static_cast<uint32_t>(bits >> 32);
    // 32-byte seed: 8 interleaved uint32 so every following DataCopy stage
    // moves a whole 32-byte block.
    for (uint32_t i = 0; i < 8; i++) {
      u32.SetValue(i, ((i & 1u) != 0) ? hi : lo);
    }
    // Scalar-pipe seed writes must land before the first MTE2 copy reads them.
    PipeBarrier<PIPE_ALL>();
    constexpr uint32_t kTotalU32 = kFullTileLength * 2u;
    uint32_t len = 8;
    while (len < kTotalU32) {
      const uint32_t copyLen = (kTotalU32 - len) < len ? (kTotalU32 - len) : len;
      DataCopy(u32[len], u32[0], copyLen);
      len += copyLen;
    }
    // MTE2 doubling is asynchronous w.r.t. the MTE3 copy-out stream; force
    // completion of the stamp build before it is streamed to GM (once per
    // core, cost negligible).
    PipeBarrier<PIPE_ALL>();
  }
}

// Stream `elems` elements of the stamp to dstGm[start .. start+elems).
template <typename T>
__aicore__ inline void CopyOut(GlobalTensor<T>& dstGm, LocalTensor<T> stamp,
                               uint64_t start, uint64_t elems) {
  uint64_t offset = 0;
  for (; offset + kFullTileLength <= elems; offset += kFullTileLength) {
    DataCopyExtParams copyParams{1,
                                 static_cast<uint32_t>(kFullTileLength *
                                                       sizeof(T)),
                                 0, 0, 0};
    DataCopyPad(dstGm[start + offset], stamp, copyParams);
  }
  if (offset < elems) {
    DataCopyExtParams padParams{1,
                                static_cast<uint32_t>((elems - offset) *
                                                      sizeof(T)),
                                0, 0, 0};
    DataCopyPad(dstGm[start + offset], stamp, padParams);
  }
}

template <typename T>
__aicore__ inline void FullProcess(GM_ADDR dst, FullTilingData tiling) {

  // Contiguous per-core partition: load diff <= 1 element, good locality.
  const uint32_t blockId = GetBlockIdx();
  const uint32_t blockNum = GetBlockNum();
  const uint64_t perCore = tiling.n / blockNum;
  const uint64_t extra = tiling.n % blockNum;
  const uint64_t start =
      blockId * perCore + (blockId < extra ? blockId : extra);
  const uint64_t elems = perCore + (blockId < extra ? 1u : 0u);
  if (elems == 0) {
    return;
  }

  TPipe pipe;
  TBuf<TPosition::VECCALC> stampBuf;
  pipe.InitBuffer(stampBuf, kFullTileLength * sizeof(T));
  LocalTensor<T> stamp = stampBuf.Get<T>();

  BuildStamp<T>(stamp, tiling.val);

  GlobalTensor<T> dstGm;
  dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst), tiling.n);
  CopyOut<T>(dstGm, stamp, start, elems);
}

}  // namespace

extern "C" __global__ __aicore__ void full_i32(GM_ADDR dst,
                                               FullTilingData tiling) {
  FullProcess<int32_t>(dst, tiling);
}

extern "C" __global__ __aicore__ void full_i64(GM_ADDR dst,
                                               FullTilingData tiling) {
  FullProcess<int64_t>(dst, tiling);
}

extern "C" __global__ __aicore__ void full_f32(GM_ADDR dst,
                                               FullTilingData tiling) {
  FullProcess<float>(dst, tiling);
}

extern "C" __global__ __aicore__ void full_f64(GM_ADDR dst,
                                               FullTilingData tiling) {
  FullProcess<double>(dst, tiling);
}
