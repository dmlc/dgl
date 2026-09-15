/*
 * @file src/array/ascend/scatter_add_kernel.cpp
 * @brief Ascend C device kernel for Scatter Add on first dimension.
 *
 * Math: out[idx[i], *] += feat[i, *]
 *
 * - SIMD / MemBase paradigm, mirroring segment_reduce_sum_kernel.cpp.
 * - Accumulation: SetAtomicAdd<float>() + DataCopyPad UB->GM (atomic +=),
 *   kernel does NOT zero out; caller pre-zeros out (matches CPU/CUDA contract).
 * - SetAtomicAdd / SetAtomicNone paired to manage the device-persistent atomic
 *   mode register lifecycle (SetAtomicAdd is per-MTE3-pipe / per-core; the
 *   reset after this core's MTE3 is drained does not race with other cores'
 *   in-flight atomic writes).
 * - Multi-core split over N (feat rows); blockDim = min(N, 40) set on host.
 *
 * Key synchronization mechanisms:
 * - idx pre-loaded to UB via TBuf + DataCopyPad, followed by
 *   PipeBarrier<PIPE_ALL> for explicit MTE2->S (Scalar) synchronization
 *   (one-shot load outside the main loop, non-hot-path).
 * - In ProcessBatch, after feat DeQue, SetFlag/WaitFlag<HardEvent::MTE2_MTE3>
 *   event synchronization (hot-path single cross-pipe dependency; Set enqueued
 *   on MTE2 after DataCopyPad, Wait enqueued on MTE3 before DataCopyPad so MTE3
 *   blocks until MTE2 completes — covers the dependency that EnQue/DeQue's
 *   MTE2->V-only sync does not).
 * - Before FreeTensor in ProcessBatch, PipeBarrier<PIPE_ALL> for multi-queue
 *   convergence (MTE3 atomic write + V pipe TQue internal events + MTE2 buffer
 *   reuse). PIPE_ALL is the only reliable option here; narrower
 *   PipeBarrier<PIPE_MTE3> or SetFlag/WaitFlag<MTE3_MTE2> exhibit intermittent
 *   races (V pipe TQue internal events not drained causes AllocTensor
 *   wait_flag<V_MTE2> to return early).
 * - At the end of Process(), PipeBarrier<PIPE_ALL> followed by SetAtomicNone()
 *   to reset the device atomic-add mode (outside the main loop, non-hot-path).
 */
#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_TOTAL_SIZE = 192 * 1024;
constexpr uint32_t UB_RESERVED = 2048;
constexpr uint32_t UB_AVAILABLE = UB_TOTAL_SIZE - UB_RESERVED;
constexpr uint32_t BYTE_ALIGN = 32;
constexpr uint32_t BLOCK_COUNT_MAX = 4095;

class KernelScatterAdd {
public:
    __aicore__ inline void Init(GM_ADDR feat, GM_ADDR idx, GM_ADDR output,
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

        idxGm.SetGlobalBuffer((__gm__ uint32_t*)idx, this->N);
        featGm.SetGlobalBuffer((__gm__ float*)feat, this->N * this->featDim);
        outputGm.SetGlobalBuffer((__gm__ float*)output, this->M * this->featDim);

        uint32_t myRows = (this->endRow > this->startRow)
                              ? (this->endRow - this->startRow)
                              : 0;
        this->myRows = myRows;
        uint32_t idxRowsForBuf = myRows > 0 ? myRows : 1;
        uint32_t idxAlignedBytes =
            ((idxRowsForBuf * sizeof(uint32_t) + BYTE_ALIGN - 1) / BYTE_ALIGN)
            * BYTE_ALIGN;

        uint32_t ubForFeat = (UB_AVAILABLE > idxAlignedBytes)
                                 ? (UB_AVAILABLE - idxAlignedBytes)
                                 : 0;
        this->batchItems = ubForFeat / (BUFFER_NUM * rowAlignedBytes);
        if (this->batchItems == 0) {
            this->batchItems = 1;
        }
        if (this->batchItems > BLOCK_COUNT_MAX) {
            this->batchItems = BLOCK_COUNT_MAX;
        }

        pipe.InitBuffer(featQueue, BUFFER_NUM,
                        this->batchItems * rowAlignedBytes);
        pipe.InitBuffer(idxBuf, idxAlignedBytes);

        // Event ID for MTE2->MTE3 cross-pipe sync.
        evtMte2ToMte3_ = static_cast<int32_t>(
            pipe.AllocEventID<AscendC::HardEvent::MTE2_MTE3>());
    }

    __aicore__ inline void Process() {
        if (this->myRows == 0) {
            return;
        }
        AscendC::LocalTensor<uint32_t> idxLocal = idxBuf.Get<uint32_t>();
        AscendC::DataCopyExtParams idxParams = {
            1, this->myRows * (uint32_t)sizeof(uint32_t), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> idxPad = {false, 0, 0, 0};
        AscendC::DataCopyPad<uint32_t>(
            idxLocal, idxGm[this->startRow], idxParams, idxPad);
        // MTE2->S sync: ensure idx DataCopyPad completes before scalar GetValue.
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::SetAtomicAdd<float>();
        for (uint32_t row = this->startRow; row < this->endRow;
             row += this->batchItems) {
            uint32_t remaining = this->endRow - row;
            uint32_t itemCount = remaining > this->batchItems
                                      ? this->batchItems
                                      : remaining;
            ProcessBatch(row, itemCount, idxLocal);
        }
        // Drain MTE3 before resetting atomic mode.
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SetAtomicNone();

        pipe.ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(evtMte2ToMte3_);
    }

private:
    __aicore__ inline void ProcessBatch(
        uint32_t startRow, uint32_t itemCount,
        const AscendC::LocalTensor<uint32_t>& idxLocal) {
        AscendC::LocalTensor<float> featBatch = featQueue.AllocTensor<float>();

        AscendC::DataCopyExtParams copyParams = {
            (uint16_t)itemCount, this->rowBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams = {
            true, 0, (uint8_t)this->rightPadding, 0.0f};
        AscendC::DataCopyPad<float>(
            featBatch, featGm[startRow * this->featDim], copyParams, padParams);
        featQueue.EnQue(featBatch);

        AscendC::LocalTensor<float> featBuf = featQueue.DeQue<float>();
        // MTE2->MTE3 sync: MTE3 atomic writes wait for MTE2 source load.
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtMte2ToMte3_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtMte2ToMte3_);
        for (uint32_t i = 0; i < itemCount; ++i) {
            uint32_t writeRow = idxLocal.GetValue(startRow - this->startRow + i);
            AscendC::DataCopyExtParams outParams = {1, this->rowBytes, 0, 0, 0};
            AscendC::DataCopyPad<float>(
                outputGm[writeRow * this->featDim],
                featBuf[i * this->rowAlignedElems], outParams);
        }
        // Multi-queue convergence: drain MTE3+V before FreeTensor.
        AscendC::PipeBarrier<PIPE_ALL>();
        featQueue.FreeTensor(featBuf);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> featQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> idxBuf;
    AscendC::GlobalTensor<uint32_t> idxGm;
    AscendC::GlobalTensor<float> featGm;
    AscendC::GlobalTensor<float> outputGm;
    int32_t evtMte2ToMte3_;  // MTE2->MTE3 cross-pipe event ID.
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

extern "C" __global__ __aicore__ void kernel_scatter_add(
    GM_ADDR feat, GM_ADDR idx, GM_ADDR output, GM_ADDR tiling_ptr) {
    KernelScatterAdd op;
    op.Init(feat, idx, output, tiling_ptr);
    op.Process();
}
