/*
 * @file src/array/ascend/segment_reduce_grad_kernel.cpp
 * @brief Ascend C device kernel for BackwardSegmentCmp (backward phase of
 *        segment min/max reduce: direct scatter-assign).
 *
 * Math (matches CUDA segment_reduce.cuh:72-112 / CPU segment_reduce.h:172):
 *   BackwardSegmentCmp:
 *       out[arg[r,c], c] = feat[r,c]    when arg[r,c] >= 0            (direct assign)
 *
 * The kernel is a scatter-write operator on GM, implemented with the
 * SIMD / MemBase paradigm:
 *   - multi-core split over N (feat rows); blockDim = min(N, 40) set on host;
 *   - TPipe + TQue<VECIN,2> double-buffer for feat MTE2 load;
 *   - TBuf<VECCALC> for arg (per-batch MTE2 preload) and for a
 *     32-byte aligned single-element scratch buffer;
 *   - DataCopyPad for every GM<->UB transfer (non-aligned friendly).
 *
 * Core design: single-element scatter write through a
 * 32-byte aligned scratchBuf.  feat[col] cannot be used directly as a
 * DataCopyPad(UB->GM) source because its offset col*4 is not guaranteed to be
 * 32-byte aligned (DataCopyPad requires the UB source start address to be
 * 32-byte aligned).  Therefore each emitted element is staged as:
 *     GetValue(feat) -> SetValue(scratchBuf[0]) -> DataCopyPad(GM, scratchBuf, blockLen=4)
 *
 * Synchronization (reuses the scatter_add verified paradigm, adapted to the
 * scratchBuf + scalar GetValue/SetValue path):
 *   1. Per batch: feat/arg are MTE2-loaded.  The scalar loop reads them via
 *      GetValue (S pipe).  A PipeBarrier<PIPE_MTE2> after the loads provides
 *      the MTE2->S dependency (arg lives in TBuf without EnQue/DeQue, and
 *      GetValue on featBuf is an S-pipe read that DeQue's MTE2->V sync does
 *      not cover).
 *   2. Per emitted element: SetValue(scratchBuf[0]) [S] -> DataCopyPad(GM,
 *      scratchBuf) [MTE3] is an S->MTE3 dependency, and consecutive elements
 *      reuse scratchBuf[0] so there is also a WAR MTE3->S dependency.  These
 *      are synchronised with SetFlag/WaitFlag<HardEvent::S_MTE3> (S->MTE3)
 *      and SetFlag/WaitFlag<HardEvent::MTE3_S> (MTE3->S), the documented
 *      pattern for "DataCopy must wait for SetValue" (see SetFlag-WaitFlag
 *      doc).
 *   3. Before FreeTensor: PipeBarrier<PIPE_ALL> for multi-queue convergence
 *      (MTE3 writes + S done -> MTE2 buffer reuse), matching scatter_add.
 *
 * arg is unified to uint32_t inside the kernel; int64 inputs are narrowed to
 * int32 on the host (CopyTo(CPU)+AsNumBits(32)+CopyTo(NPU), a width-preserving
 * reinterpret, not a truncation), identical to scatter_add.cc /
 * segment_reduce.cc.
 */
#include "kernel_operator.h"

namespace {
// Shared UB / alignment constants (same values as scatter_add_kernel.cpp).
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_TOTAL_SIZE = 192 * 1024;
constexpr uint32_t UB_RESERVED = 2048;
constexpr uint32_t UB_AVAILABLE = UB_TOTAL_SIZE - UB_RESERVED;
constexpr uint32_t BYTE_ALIGN = 32;
constexpr uint32_t BLOCK_COUNT_MAX = 4095;
constexpr uint32_t SCRATCH_BYTES = 32;  // 1 dataBlock, guarantees 32B-aligned src.
}  // namespace

// =========================================================================
// BackwardSegmentCmp : conditional direct scatter-assign
//   out[arg[r,c], c] = feat[r,c]   when arg[r,c] >= 0
// =========================================================================
class KernelBackwardSegmentCmp {
 public:
  __aicore__ inline void Init(GM_ADDR feat, GM_ADDR arg, GM_ADDR out,
                              GM_ADDR tiling_ptr) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::GlobalTensor<uint32_t> tilingGm;
    tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 3);
    this->N = tilingGm.GetValue(0);
    this->M = tilingGm.GetValue(1);
    this->featDim = tilingGm.GetValue(2);

    this->rowBytes = this->featDim * sizeof(float);
    uint32_t rowAlignedBytes =
        (this->rowBytes + BYTE_ALIGN - 1) / BYTE_ALIGN * BYTE_ALIGN;
    this->rowAlignedElems = rowAlignedBytes / sizeof(float);
    this->rightPadding = this->rowAlignedElems - this->featDim;

    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t blockNum = AscendC::GetBlockNum();
    this->rowsPerCore = (this->N + blockNum - 1) / blockNum;
    this->startRow = blockIdx * this->rowsPerCore;
    this->endRow = (this->startRow + this->rowsPerCore > this->N)
                       ? this->N
                       : (this->startRow + this->rowsPerCore);
    this->myRows = (this->endRow > this->startRow)
                       ? (this->endRow - this->startRow)
                       : 0;

    argGm.SetGlobalBuffer((__gm__ uint32_t*)arg, this->N * this->featDim);
    featGm.SetGlobalBuffer((__gm__ float*)feat, this->N * this->featDim);
    outputGm.SetGlobalBuffer((__gm__ float*)out, this->M * this->featDim);

    // UB budget: subtract scratchBuf (32B), then split over featQueue*2 +
    // argBuf (coefficient 3).  No idxEtypeBuf, no atomic.
    uint32_t ubForFeat =
        (UB_AVAILABLE > SCRATCH_BYTES) ? (UB_AVAILABLE - SCRATCH_BYTES) : 0;
    uint32_t denom = (BUFFER_NUM + 1) * rowAlignedBytes;
    this->batchItems = (denom > 0) ? (ubForFeat / denom) : 0;
    if (this->batchItems == 0) this->batchItems = 1;
    if (this->batchItems > BLOCK_COUNT_MAX) this->batchItems = BLOCK_COUNT_MAX;

    pipe.InitBuffer(
        featQueue, BUFFER_NUM, this->batchItems * rowAlignedBytes);
    pipe.InitBuffer(argBuf, this->batchItems * rowAlignedBytes);
    pipe.InitBuffer(scratchBuf, SCRATCH_BYTES);

    evSMte3_ = static_cast<int32_t>(
        pipe.AllocEventID<AscendC::HardEvent::S_MTE3>());
    evMte3S_ = static_cast<int32_t>(
        pipe.AllocEventID<AscendC::HardEvent::MTE3_S>());
  }

  __aicore__ inline void Process() {
    if (this->myRows == 0) return;

    // No SetAtomicAdd: arg is unique within a column (no write conflict), so
    // element writes are direct assignments (overwrite).
    for (uint32_t row = this->startRow; row < this->endRow;
         row += this->batchItems) {
      uint32_t remaining = this->endRow - row;
      uint32_t itemCount =
          remaining > this->batchItems ? this->batchItems : remaining;
      ProcessBatch(row, itemCount);
    }

    // Drain MTE3 before releasing events.
    AscendC::PipeBarrier<PIPE_ALL>();
    pipe.ReleaseEventID<AscendC::HardEvent::S_MTE3>(evSMte3_);
    pipe.ReleaseEventID<AscendC::HardEvent::MTE3_S>(evMte3S_);
  }

 private:
  __aicore__ inline void ProcessBatch(uint32_t row, uint32_t itemCount) {
    AscendC::LocalTensor<float> featBatch = featQueue.AllocTensor<float>();

    AscendC::DataCopyExtParams copyParams = {
        (uint16_t)itemCount, this->rowBytes, 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> padParams = {
        true, 0, (uint8_t)this->rightPadding, 0.0f};
    AscendC::DataCopyPad<float>(
        featBatch, featGm[row * this->featDim], copyParams, padParams);

    AscendC::LocalTensor<uint32_t> argLocal = argBuf.Get<uint32_t>();
    AscendC::DataCopyPadExtParams<uint32_t> argPad = {
        true, 0, (uint8_t)this->rightPadding, 0u};
    AscendC::DataCopyPad<uint32_t>(
        argLocal, argGm[row * this->featDim], copyParams, argPad);

    featQueue.EnQue(featBatch);
    AscendC::LocalTensor<float> featBuf = featQueue.DeQue<float>();

    // MTE2->S: ensure feat/arg loads complete before scalar GetValue.
    AscendC::PipeBarrier<PIPE_MTE2>();

    AscendC::LocalTensor<float> scratch = scratchBuf.Get<float>();
    AscendC::DataCopyExtParams outParams = {1, sizeof(float), 0, 0, 0};

    for (uint32_t i = 0; i < itemCount; ++i) {
      uint32_t rowBase = i * this->rowAlignedElems;
      for (uint32_t col = 0; col < this->featDim; ++col) {
        uint32_t argVal = argLocal.GetValue(rowBase + col);
        // arg>=0 (as int32): -1 sentinel means "no segment" (empty).
        if (static_cast<int32_t>(argVal) >= 0) {
          float val = featBuf.GetValue(rowBase + col);
          scratch.SetValue(0, val);
          AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(evSMte3_);
          AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(evSMte3_);
          AscendC::DataCopyPad<float>(
              outputGm[argVal * this->featDim + col], scratch, outParams);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(evMte3S_);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(evMte3S_);
        }
      }
    }

    AscendC::PipeBarrier<PIPE_ALL>();
    featQueue.FreeTensor(featBuf);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> featQueue;
  AscendC::TBuf<AscendC::TPosition::VECCALC> argBuf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> scratchBuf;
  AscendC::GlobalTensor<uint32_t> argGm;
  AscendC::GlobalTensor<float> featGm;
  AscendC::GlobalTensor<float> outputGm;
  int32_t evSMte3_;
  int32_t evMte3S_;
  uint32_t N;
  uint32_t M;
  uint32_t featDim;
  uint32_t rowBytes;
  uint32_t rowAlignedElems;
  uint32_t rightPadding;
  uint32_t batchItems;
  uint32_t rowsPerCore;
  uint32_t startRow;
  uint32_t endRow;
  uint32_t myRows;
};

// ---- kernel entry point ----

extern "C" __global__ __aicore__ void kernel_backward_segment_cmp(
    GM_ADDR feat, GM_ADDR arg, GM_ADDR out, GM_ADDR tiling_ptr) {
  KernelBackwardSegmentCmp op;
  op.Init(feat, arg, out, tiling_ptr);
  op.Process();
}
