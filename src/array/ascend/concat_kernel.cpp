#include "kernel_operator.h"

#include "concat_tiling.h"

namespace {

template <typename T>
__aicore__ inline void RunConcat(
    GM_ADDR dst, const __gm__ ConcatTilingData* tiling,
    const __gm__ ConcatInputDesc* inputs) {
  const uint32_t inputCount = tiling->inputCount;
  const uint32_t segCount = tiling->segCount;
  const uint32_t dstStride = tiling->dstStride;
  const uint32_t chunkElems = tiling->chunkElems;
  const uint32_t totalChunks = tiling->totalChunks;

  const uint32_t startChunk =
      AscendC::GetBlockIdx() * tiling->chunksPerCore;
  uint32_t endChunk = startChunk + tiling->chunksPerCore;
  if (endChunk > totalChunks) endChunk = totalChunks;
  if (startChunk >= endChunk) return;

  AscendC::GlobalTensor<T> dstGm;
  dstGm.SetGlobalBuffer((__gm__ T*)dst);

  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, CONCAT_BUFFER_NUM> bufQue;
  pipe.InitBuffer(
      bufQue, CONCAT_BUFFER_NUM,
      static_cast<uint32_t>(chunkElems * sizeof(T)));

  const AscendC::DataCopyPadExtParams<T> padIn{false, 0, 0, static_cast<T>(0)};

  uint32_t lo = 0;
  uint32_t hi = inputCount;
  while (lo < hi) {
    const uint32_t mid = (lo + hi) / 2;
    const uint32_t inputEnd =
        inputs[mid].chunkBase + segCount * inputs[mid].chunksPerSeg;
    if (inputEnd <= startChunk) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  uint32_t chunk = startChunk;
  for (uint32_t i = lo; i < inputCount && chunk < endChunk; ++i) {
    const uint64_t src = inputs[i].src;
    const uint32_t segElems = inputs[i].segElems;
    const uint32_t dstBase = inputs[i].dstBase;
    const uint32_t chunksPerSeg = inputs[i].chunksPerSeg;
    const uint32_t chunkBase = inputs[i].chunkBase;
    const uint32_t inputEnd = chunkBase + segCount * chunksPerSeg;
    if (chunk < chunkBase) chunk = chunkBase;
    if (chunk >= inputEnd) continue;

    AscendC::GlobalTensor<T> srcGm;
    srcGm.SetGlobalBuffer((__gm__ T*)(uintptr_t)src);

    while (chunk < endChunk && chunk < inputEnd) {
      const uint32_t localChunk = chunk - chunkBase;
      const uint32_t seg = localChunk / chunksPerSeg;
      const uint32_t k = localChunk - seg * chunksPerSeg;
      const uint32_t inChunk = k * chunkElems;
      const uint32_t remain = segElems - inChunk;
      const uint32_t len = remain < chunkElems ? remain : chunkElems;
      const uint32_t bytes = len * static_cast<uint32_t>(sizeof(T));
      const uint32_t srcOff = seg * segElems + inChunk;
      const uint32_t dstOff = seg * dstStride + dstBase + inChunk;

      AscendC::LocalTensor<T> buf = bufQue.AllocTensor<T>();
      const AscendC::DataCopyExtParams params{1, bytes, 0, 0, 0};
      AscendC::DataCopyPad(buf, srcGm[srcOff], params, padIn);
      bufQue.EnQue(buf);

      AscendC::LocalTensor<T> ready = bufQue.DeQue<T>();
      AscendC::DataCopyPad(dstGm[dstOff], ready, params);
      AscendC::PipeBarrier<PIPE_MTE3>();
      bufQue.FreeTensor(ready);

      ++chunk;
    }
  }
}

template <typename T>
__aicore__ inline void DispatchConcat(GM_ADDR dst, GM_ADDR control) {
  const __gm__ ConcatTilingData* tiling =
      (const __gm__ ConcatTilingData*)control;
  const __gm__ ConcatInputDesc* inputs =
      (const __gm__ ConcatInputDesc*)((const __gm__ uint8_t*)control +
                                      sizeof(ConcatTilingData));
  RunConcat<T>(dst, tiling, inputs);
}

}  // namespace

extern "C" __global__ __aicore__ void concat_kernel_f32(
    GM_ADDR dst, GM_ADDR control) {
  DispatchConcat<float>(dst, control);
}

extern "C" __global__ __aicore__ void concat_kernel_f16(
    GM_ADDR dst, GM_ADDR control) {
  DispatchConcat<half>(dst, control);
}

extern "C" __global__ __aicore__ void concat_kernel_i32(
    GM_ADDR dst, GM_ADDR control) {
  DispatchConcat<int32_t>(dst, control);
}

extern "C" __global__ __aicore__ void concat_kernel_i64(
    GM_ADDR dst, GM_ADDR control) {
  DispatchConcat<int64_t>(dst, control);
}
