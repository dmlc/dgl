// UnaryElewise kernel (arith::Neg, y = -x) -- int64 entry, vectorized via a
// 2 x int32 lane view of the int64 stream (lane 2i = lo_i, lane 2i+1 = hi_i):
// the static [0,-1,0,-1,...] parity pattern makes one Sub(t, borrow, s) compute
// lo' = -lo / hi' = -hi - 1 (exact iff lo != 0). A per-tile zero probe (f32
// Cast + Abs + ReduceMin) routes tiles containing a zero int32 lane (lo == 0
// invalidates the borrow) to the rare-tile exact recompute (element-0
// single-lane copies + one-lane-shifted sub-tile correction, all blockCount=1
// contiguous copies); the probe reaches the scalar branch via SetFlag/WaitFlag<V_S>.
#include "kernel_operator.h"

#include "unary_elewise_tiling.h"

using namespace AscendC;

constexpr uint32_t kI64TileElems = 1024;               // int64 elems per tile
constexpr uint32_t kI64TileLanes = kI64TileElems * 2;  // int32 lanes (8KB)

// int64 vectorized path: y = -x via the 2 x int32 lane decomposition above.
extern "C" __global__ __aicore__ void unary_neg_i64(
    GM_ADDR x, GM_ADDR y, GM_ADDR tiling_ptr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 1);
  uint32_t n = tilingGm.GetValue(0);

  uint32_t start = 0;
  uint32_t end = 0;
  UnaryBlockRange(n, start, end);  // last core carries remainder; idle if start >= n

  // int64 stream viewed as 2n int32 lanes (little-endian: lane 2i = lo_i,
  // lane 2i+1 = hi_i); lane indices use uint64 (2n can exceed uint32).
  GlobalTensor<int32_t> xGm;
  xGm.SetGlobalBuffer((__gm__ int32_t*)x, static_cast<uint64_t>(n) * 2);
  GlobalTensor<int32_t> yGm;
  yGm.SetGlobalBuffer((__gm__ int32_t*)y, static_cast<uint64_t>(n) * 2);

  // UB budget ~104KB of the 192KB UB (queues, pattern, probe, rare-tile bufs).
  AscendC::TPipe pipe;
  TQue<TPosition::VECIN, 2> xQueue;   // int32 lanes per tile, double buffered
  TQue<TPosition::VECOUT, 2> yQueue;  // final tile (negated + borrowed)
  pipe.InitBuffer(xQueue, 2, kI64TileLanes * sizeof(int32_t));
  pipe.InitBuffer(yQueue, 2, kI64TileLanes * sizeof(int32_t));
  TBuf<TPosition::VECCALC> pattern;  // static [0,-1,0,-1,...] + build scratch
  pipe.InitBuffer(pattern, 2 * kI64TileLanes * sizeof(int32_t));
  TBuf<TPosition::VECCALC> probe;  // f32 zero-probe image of the tile
  pipe.InitBuffer(probe, kI64TileLanes * sizeof(float));
  TBuf<TPosition::VECCALC> reduceTmp;  // ReduceMin shared buffer + output
  pipe.InitBuffer(reduceTmp, kI64TileLanes * sizeof(float) + 256);
  TBuf<TPosition::VECCALC> rareBuf;  // rare tile: element-0 lo/hi lanes
  pipe.InitBuffer(rareBuf, 4 * kI64TileElems * sizeof(int32_t));
  TBuf<TPosition::VECCALC> subBuf;  // rare sub-tile: sSub / tmpRSub / tSub
  pipe.InitBuffer(subBuf, 3 * kI64TileLanes * sizeof(int32_t));

  // static lane pattern: parity of the lane index, negated -> [0,-1,0,-1,...]
  // (And() counts int16 lanes regardless of dtype -- only 1024 int32 lanes per
  // call, hence two 1024-lane chunks; typed ops count their own dtype lanes.)
  LocalTensor<int32_t> borrow = pattern.Get<int32_t>();
  LocalTensor<int32_t> idx = pattern.Get<int32_t>()[kI64TileLanes];
  CreateVecIndex(idx, 0, kI64TileLanes);
  Duplicate(borrow, static_cast<int32_t>(1), kI64TileLanes);
  And(borrow, borrow, idx, kI64TileLanes);  // [0,1] x 512 + [1,1] x 512
  And(borrow[kI64TileElems], borrow[kI64TileElems], idx[kI64TileElems],
      kI64TileLanes);                       // second 1024-lane chunk
  Muls(borrow, borrow, static_cast<int32_t>(-1), kI64TileLanes);  // [0,-1,...]
  // parity mask [0,1,0,1,...] for the rare-tile hi-lane correction (idx scratch is dead by now).
  LocalTensor<int32_t> parityOdd = pattern.Get<int32_t>()[kI64TileLanes];
  Muls(parityOdd, borrow, static_cast<int32_t>(-1), kI64TileLanes);  // [0,1,...]

  LocalTensor<float> probeF = probe.Get<float>();
  LocalTensor<float> reduceTmpF = reduceTmp.Get<float>();
  LocalTensor<float> reduceOut = reduceTmpF[kI64TileLanes];
  LocalTensor<int32_t> sSub = subBuf.Get<int32_t>();
  LocalTensor<int32_t> tmpRSub = subBuf.Get<int32_t>()[kI64TileLanes];
  LocalTensor<int32_t> tSub = subBuf.Get<int32_t>()[2 * kI64TileLanes];

  // Cross-pipe event ids allocated once; each Set/Wait pair reuses its id.
  const int32_t evtVToS = pipe.AllocEventID<HardEvent::V_S>();
  const int32_t evtMte2ToV = pipe.AllocEventID<HardEvent::MTE2_V>();
  const int32_t evtVToMte3 = pipe.AllocEventID<HardEvent::V_MTE3>();
  const int32_t evtMte3ToMte2 = pipe.AllocEventID<HardEvent::MTE3_MTE2>();

  for (uint32_t off = start; off < end;) {
    uint32_t cnt = end - off;
    if (cnt > kI64TileElems) cnt = kI64TileElems;
    const uint32_t lanes = cnt * 2;
    const uint64_t laneBase = static_cast<uint64_t>(off) * 2;

    LocalTensor<int32_t> xBuf = xQueue.AllocTensor<int32_t>();
    if (cnt == kI64TileElems) {
      DataCopy(xBuf, xGm[laneBase], kI64TileLanes);
    } else {
      DataCopyExtParams inParams{1,
          static_cast<uint32_t>(cnt * sizeof(int64_t)), 0, 0, 0};
      DataCopyPadExtParams<int32_t> inPad{false, 0, 0, 0};
      DataCopyPad(xBuf, xGm[laneBase], inParams, inPad);
    }
    xQueue.EnQue(xBuf);

    LocalTensor<int32_t> s = xQueue.DeQue<int32_t>();

    // Zero probe: min |lane| == 0 flags a tile containing a zero int32 lane
    // (lo_i == 0 invalidates the borrow; hi_i == 0 over-triggers harmlessly);
    // zero-ness is preserved by the int32->f32 cast.
    Cast(probeF, s, RoundMode::CAST_RINT, static_cast<int32_t>(lanes));
    Abs(probeF, probeF, static_cast<int32_t>(lanes));
    ReduceMin(reduceOut, probeF, reduceTmpF, static_cast<int32_t>(lanes));
    // Directed V -> S sync: the scalar read below waits only on the reduce
    // result, keeping the TQue double-buffer overlap intact.
    SetFlag<HardEvent::V_S>(evtVToS);
    WaitFlag<HardEvent::V_S>(evtVToS);

    if (reduceOut.GetValue(0) == 0.0f) {
      // Rare tile: exact per-element recompute (see file header).
      xQueue.FreeTensor(s);
      LocalTensor<int32_t> rareLo = rareBuf.Get<int32_t>();
      LocalTensor<int32_t> rareHi = rareBuf.Get<int32_t>()[kI64TileElems];
      LocalTensor<int32_t> rareLoOut =
          rareBuf.Get<int32_t>()[2 * kI64TileElems];
      LocalTensor<int32_t> rareHiOut =
          rareBuf.Get<int32_t>()[3 * kI64TileElems];
      const DataCopyExtParams oneLane{1, sizeof(int32_t), 0, 0, 0};
      const DataCopyExtParams oneBlock{1,
          static_cast<uint32_t>((lanes - 2) * sizeof(int32_t)), 0, 0, 0};
      const DataCopyPadExtParams<int32_t> inPad{false, 0, 0, 0};
      // element 0: single-lane copies (lo at GM lane 0, hi at lane 1)
      DataCopyPad(rareLo[0], xGm[laneBase], oneLane, inPad);
      DataCopyPad(rareHi[0], xGm[laneBase + 1], oneLane, inPad);
      if (cnt >= 2) {
        // sub-tile in: tile lanes 2..2cnt-1 and the one-lane-shifted view
        // (tile lanes 1..2cnt-2), both contiguous blockCount=1 transfers
        DataCopyPad(sSub, xGm[laneBase + 2], oneBlock, inPad);
        DataCopyPad(tmpRSub, xGm[laneBase + 1], oneBlock, inPad);
      }
      SetFlag<HardEvent::MTE2_V>(evtMte2ToV);
      WaitFlag<HardEvent::MTE2_V>(evtMte2ToV);
      // element 0: lo' = -lo (mod 2^32), hi' = -hi - [lo != 0]
      Muls(rareLoOut, rareLo, static_cast<int32_t>(-1), 1);
      Min(rareLo, rareLo, rareLoOut, 1);  // min(lo, -lo) <= 0, == 0 iff lo == 0
      Maxs(rareLo, rareLo, static_cast<int32_t>(-1), 1);  // {-1, 0}
      Muls(rareLo, rareLo, static_cast<int32_t>(-1), 1);  // b = {1, 0}
      Muls(rareHiOut, rareHi, static_cast<int32_t>(-1), 1);
      Sub(rareHiOut, rareHiOut, rareLo, 1);
      if (cnt >= 2) {
        const int32_t lanesSub = static_cast<int32_t>(lanes - 2);
        // corrRaw = [tmpRSub == 0]: min(x, -x) <= 0 and == 0 iff x == 0,
        // Max(-1) clamps to {-1, 0}, +1 gives {0, 1}
        Muls(tSub, tmpRSub, static_cast<int32_t>(-1), lanesSub);  // scratch
        Min(tmpRSub, tmpRSub, tSub, lanesSub);
        Maxs(tmpRSub, tmpRSub, static_cast<int32_t>(-1), lanesSub);
        Adds(tmpRSub, tmpRSub, static_cast<int32_t>(1), lanesSub);
        Mul(tmpRSub, tmpRSub, parityOdd, lanesSub);  // mask to hi lanes
        Sub(tSub, borrow, sSub, lanesSub);  // [-lo_i, -hi_i - 1]
        Add(tSub, tSub, tmpRSub, lanesSub);  // hi: -hi - 1 + [lo == 0]
      }
      SetFlag<HardEvent::V_MTE3>(evtVToMte3);
      WaitFlag<HardEvent::V_MTE3>(evtVToMte3);
      DataCopyPad(yGm[laneBase], rareLoOut[0], oneLane);
      DataCopyPad(yGm[laneBase + 1], rareHiOut[0], oneLane);
      if (cnt >= 2) {
        DataCopyPad(yGm[laneBase + 2], tSub, oneBlock);
      }
      // WAR guard: the out-copies above read the rare buffers on MTE3; hold
      // back future MTE2 traffic (next tile's copies) until they have drained.
      SetFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2);
      WaitFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2);
      off += cnt;
      continue;
    }

    // One-op lane negation with borrow folded in:
    //   t = borrow - s = [0 - lo_i, -1 - hi_i] = [-lo_i, -hi_i - 1]
    LocalTensor<int32_t> t = yQueue.AllocTensor<int32_t>();
    Sub(t, borrow, s, static_cast<int32_t>(lanes));

    yQueue.EnQue(t);
    xQueue.FreeTensor(s);

    LocalTensor<int32_t> yDeq = yQueue.DeQue<int32_t>();
    if (cnt == kI64TileElems) {
      DataCopy(yGm[laneBase], yDeq, kI64TileLanes);
    } else {
      DataCopyExtParams outParams{1,
          static_cast<uint32_t>(cnt * sizeof(int64_t)), 0, 0, 0};
      DataCopyPad(yGm[laneBase], yDeq, outParams);  // only cnt valid elems
    }
    yQueue.FreeTensor(yDeq);

    off += cnt;
  }
}
