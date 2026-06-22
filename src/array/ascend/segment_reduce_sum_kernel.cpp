#include "kernel_operator.h"
#include <cfloat>

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_TOTAL_SIZE = 192 * 1024;
constexpr uint32_t UB_RESERVED = 2048;
constexpr uint32_t UB_AVAILABLE = UB_TOTAL_SIZE - UB_RESERVED;
constexpr uint32_t BYTE_ALIGN = 32;

#define DEFINE_SEGMENT_REDUCE_KERNEL_LAUNCHER(kernel_name, kernel_class)                                  \
    extern "C" __global__ __aicore__ void kernel_name(GM_ADDR offsets, GM_ADDR feat, GM_ADDR output,      \
                                                       GM_ADDR segment_split, GM_ADDR tiling_ptr)          \
    {                                                                                                      \
        AscendC::GlobalTensor<uint32_t> tilingGm;                                                          \
        tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 3);                                        \
        uint32_t numItems = tilingGm.GetValue(0);                                                          \
        uint32_t numSegments = tilingGm.GetValue(1);                                                       \
        uint32_t featDim = tilingGm.GetValue(2);                                                           \
        kernel_class op;                                                                                   \
        op.Init(offsets, feat, output, segment_split, numItems, numSegments, featDim);                      \
        op.Process();                                                                                      \
    }

class KernelSegmentReduceSum {
public:
    __aicore__ inline void Init(GM_ADDR offsets, GM_ADDR feat, GM_ADDR output,
                                GM_ADDR segmentSplit, uint32_t numItems, uint32_t numSegments, uint32_t featDim)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
        this->featDim = featDim;
        this->rowBytes = this->featDim * sizeof(float);
        uint32_t rowAlignedBytes = (this->rowBytes + BYTE_ALIGN - 1) / BYTE_ALIGN * BYTE_ALIGN;
        this->rowAlignedElems = rowAlignedBytes / sizeof(float);
        this->rightPadding = this->rowAlignedElems - this->featDim;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        segmentSplitGm.SetGlobalBuffer((__gm__ uint32_t *)segmentSplit, blockNum + 1);
        this->startSeg = segmentSplitGm.GetValue(blockIdx);
        this->endSeg = segmentSplitGm.GetValue(blockIdx + 1);

        offsetsGm.SetGlobalBuffer((__gm__ uint32_t *)offsets, numSegments + 1);
        featGm.SetGlobalBuffer((__gm__ float *)feat, numItems * featDim);
        outputGm.SetGlobalBuffer((__gm__ float *)output, numSegments * featDim);

        uint32_t outputBytes = rowAlignedBytes * BUFFER_NUM;
        uint32_t remainingUb = UB_AVAILABLE > outputBytes ? (UB_AVAILABLE - outputBytes) : 0;
        this->batchItems = remainingUb / (BUFFER_NUM * rowAlignedBytes);

        pipe.InitBuffer(outputQueue, BUFFER_NUM, rowAlignedBytes);
        pipe.InitBuffer(featQueue, BUFFER_NUM, this->batchItems * rowAlignedBytes);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t segIdx = this->startSeg; segIdx < this->endSeg; ++segIdx) {
            AscendC::LocalTensor<float> outputSegment = outputQueue.AllocTensor<float>();
            AscendC::Duplicate<float>(outputSegment, 0.0f, this->featDim);
            ProcessSegment(segIdx, outputSegment);
            CopyOutSegment(segIdx);
        }
    }

private:
    __aicore__ inline void ProcessSegment(uint32_t segIdx, AscendC::LocalTensor<float> outputSegment)
    {
        uint32_t start = offsetsGm.GetValue(segIdx);
        uint32_t segLen = offsetsGm.GetValue(segIdx + 1) - start;
        if (segLen == 0) {
            outputQueue.EnQue(outputSegment);
            return;
        }
        for (uint32_t processedItems = 0; processedItems < segLen; processedItems += this->batchItems) {
            uint32_t itemCount = segLen - processedItems > this->batchItems ? this->batchItems : (segLen - processedItems);
            CopyInBatch(start + processedItems, itemCount);
            ReduceBatch(outputSegment, itemCount);
        }
        outputQueue.EnQue(outputSegment);
    }

    __aicore__ inline void CopyInBatch(uint32_t startItem, uint32_t itemCount)
    {
        AscendC::LocalTensor<float> featBatch = featQueue.AllocTensor<float>();
        AscendC::DataCopyExtParams copyParams = {(uint16_t)itemCount, this->rowBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams = {true, 0, (uint8_t)this->rightPadding, 0.0f};
        AscendC::DataCopyPad<float>(featBatch, featGm[startItem * this->featDim], copyParams, padParams);
        featQueue.EnQue(featBatch);
    }

    __aicore__ inline void CopyOutSegment(uint32_t segIdx)
    {
        AscendC::LocalTensor<float> outputSegment = outputQueue.DeQue<float>();
        AscendC::DataCopyExtParams copyParams = {1, this->rowBytes, 0, 0, 0};
        AscendC::DataCopyPad<float>(outputGm[segIdx * this->featDim], outputSegment, copyParams);
        outputQueue.FreeTensor(outputSegment);
    }

    __aicore__ inline void ReduceBatch(AscendC::LocalTensor<float> accum, uint32_t itemCount)
    {
        AscendC::LocalTensor<float> featBatch = featQueue.DeQue<float>();
        for (uint32_t itemIdx = 0; itemIdx < itemCount; ++itemIdx) {
            AscendC::Add(accum, accum, featBatch[itemIdx * this->rowAlignedElems], this->featDim);
        }
        featQueue.FreeTensor(featBatch);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outputQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> featQueue;
    AscendC::GlobalTensor<uint32_t> offsetsGm;
    AscendC::GlobalTensor<uint32_t> segmentSplitGm;
    AscendC::GlobalTensor<float> featGm;
    AscendC::GlobalTensor<float> outputGm;
    uint32_t featDim;
    uint32_t rowBytes;
    uint32_t rowAlignedElems;
    uint32_t rightPadding;
    uint32_t batchItems;
    uint32_t startSeg;
    uint32_t endSeg;
};

DEFINE_SEGMENT_REDUCE_KERNEL_LAUNCHER(segment_reduce_sum, KernelSegmentReduceSum)
