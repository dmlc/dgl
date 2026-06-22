#include <cstdint>
#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TEMP_BUFFER_NUM = 2;
constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint32_t UB_TOTAL_SIZE = 192 * 1024;
constexpr uint32_t UB_RESERVED = 2048;
constexpr uint32_t UB_AVAILABLE = UB_TOTAL_SIZE - UB_RESERVED;

class AivVectorProcessor {
public:
    __aicore__ inline void Init(
        GM_ADDR featureData,
        GM_ADDR outputData,
        GM_ADDR indptrData,
        GM_ADDR indicesData,
        GM_ADDR vectorRowSplitData,
        uint32_t numDstRows,
        uint32_t numSrcRows,
        uint32_t featureDim,
        uint32_t batchCount,
        uint32_t nonZeroCount,
        AscendC::TPipe *pipe)
    {
        this->M = numDstRows;
        this->K = numSrcRows;
        this->featureDim = featureDim;
        this->batchCount = batchCount;
        this->rowWidth = featureDim;
        this->nnz = nonZeroCount;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        this->startRow = 0;
        this->localRowCount = 0;

        rowSplitGm.SetGlobalBuffer((__gm__ uint32_t *)vectorRowSplitData, blockNum + 1);
        this->startRow = rowSplitGm.GetValue(blockIdx);
        uint32_t endRow = rowSplitGm.GetValue(blockIdx + 1);
        this->localRowCount = endRow - startRow;

        featureGm.SetGlobalBuffer((__gm__ half *)featureData, K * batchCount * featureDim);
        outputGm.SetGlobalBuffer((__gm__ half *)outputData, M * batchCount * featureDim);
        indptrGm.SetGlobalBuffer((__gm__ uint32_t *)indptrData, M + 1);
        indicesGm.SetGlobalBuffer((__gm__ uint32_t *)indicesData, nnz);

        uint32_t rowBytes = rowWidth * sizeof(half);
        this->rowAlignedBytes = ((rowBytes + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
        this->rowAlignedElements = this->rowAlignedBytes / sizeof(half);
        uint32_t accumBytes = BUFFER_NUM * rowAlignedBytes;
        uint32_t remainingUb = UB_AVAILABLE > accumBytes ? (UB_AVAILABLE - accumBytes) : 0;
        this->batchSize = remainingUb / (TEMP_BUFFER_NUM * rowAlignedBytes);
        if (this->batchSize == 0) {
            this->batchSize = 1;
        }
        uint32_t batchBufferSize = this->batchSize * rowAlignedBytes;
        pipe->InitBuffer(accumQueue, BUFFER_NUM, rowAlignedBytes);
        pipe->InitBuffer(featureQueue, TEMP_BUFFER_NUM, batchBufferSize);
    }

    __aicore__ inline void Process()
    {
        if (this->localRowCount == 0) {
            return;
        }
        uint32_t rowEnd = this->startRow + this->localRowCount;
        for (uint32_t row = this->startRow; row < rowEnd; ++row) {
            uint32_t rowStartPtr = indptrGm.GetValue(row);
            uint32_t rowEndPtr = indptrGm.GetValue(row + 1);
            uint32_t rowNonZeroCount = rowEndPtr - rowStartPtr;
            for (uint32_t batchIndex = 0; batchIndex < this->batchCount; ++batchIndex) {
                AscendC::LocalTensor<half> accumUb = accumQueue.AllocTensor<half>();
                if (rowNonZeroCount == 0) {
                    AscendC::Duplicate<half>(accumUb, half(0.0f), this->rowWidth);
                } else {
                    AscendC::Duplicate<half>(accumUb, half(-65504.0f), this->rowWidth);
                    uint32_t nnzBatchCount = (rowNonZeroCount + this->batchSize - 1) / this->batchSize;
                    for (uint32_t nnzBatchIndex = 0; nnzBatchIndex < nnzBatchCount; ++nnzBatchIndex) {
                        uint32_t batchStartPtr = rowStartPtr + nnzBatchIndex * this->batchSize;
                        uint32_t currentBatchSize = (batchStartPtr + this->batchSize > rowEndPtr)
                                                   ? (rowEndPtr - batchStartPtr)
                                                   : this->batchSize;
                        CopyInBatch(batchStartPtr, currentBatchSize, batchIndex);
                        ComputeBatch(accumUb, currentBatchSize);
                    }
                }
                accumQueue.EnQue(accumUb);
                CopyOut(row, batchIndex);
            }
        }
    }

    __aicore__ inline void CopyInBatch(uint32_t batchStart, uint32_t batchNnz, uint32_t batchIndex)
    {
        AscendC::LocalTensor<half> featureBatch = featureQueue.AllocTensor<half>();
        AscendC::DataCopyExtParams copyParams = {1, static_cast<uint32_t>(this->rowWidth * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<half> padParams = {true, 0, (uint8_t)(this->rowAlignedElements - this->rowWidth), 0};
        for (uint32_t i = 0; i < batchNnz; ++i) {
            uint32_t neighborIndex = indicesGm.GetValue(batchStart + i);
            uint32_t featureOffset = i * this->rowAlignedElements;
            uint32_t featureGmOffset = (neighborIndex * this->batchCount + batchIndex) * this->featureDim;
            AscendC::DataCopyPad(featureBatch[featureOffset], featureGm[featureGmOffset], copyParams, padParams);
        }
        featureQueue.EnQue(featureBatch);
    }

    __aicore__ inline void ComputeBatch(AscendC::LocalTensor<half>& accumBlock, uint32_t batchNnz)
    {
        AscendC::LocalTensor<half> computeBatch = featureQueue.DeQue<half>();
        for (uint32_t i = 0; i < batchNnz; ++i) {
            uint32_t featureOffset = i * this->rowAlignedElements;
            AscendC::Max(accumBlock, accumBlock, computeBatch[featureOffset], this->rowWidth);
        }
        featureQueue.FreeTensor(computeBatch);
    }

    __aicore__ inline void CopyOut(uint32_t row, uint32_t batchIndex)
    {
        AscendC::LocalTensor<half> accumBlock = accumQueue.DeQue<half>();
        AscendC::DataCopyExtParams copyParams = {1, static_cast<uint32_t>(this->rowWidth * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPad(outputGm[(row * this->batchCount + batchIndex) * this->featureDim], accumBlock, copyParams);
        accumQueue.FreeTensor(accumBlock);
    }

    uint32_t M, K, featureDim, batchCount, rowWidth, nnz;
    uint32_t batchSize;
    uint32_t rowAlignedBytes, rowAlignedElements;
    uint32_t startRow, localRowCount;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> accumQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, TEMP_BUFFER_NUM> featureQueue;
    AscendC::GlobalTensor<half> featureGm;
    AscendC::GlobalTensor<half> outputGm;
    AscendC::GlobalTensor<uint32_t> indptrGm, indicesGm, rowSplitGm;
};

extern "C" __global__ __aicore__ void bspmm_max(
    GM_ADDR featureData,
    GM_ADDR outputData,
    GM_ADDR indptrData,
    GM_ADDR indicesData,
    GM_ADDR vectorRowSplitData,
    uint32_t numDstRows,
    uint32_t numSrcRows,
    uint32_t featureDim,
    uint32_t nonZeroCount,
    uint32_t batchCount)
{
    AscendC::TPipe pipe;
    AivVectorProcessor vectorProcessor;
    vectorProcessor.Init(
        featureData,
        outputData,
        indptrData,
        indicesData,
        vectorRowSplitData,
        numDstRows,
        numSrcRows,
        featureDim,
        batchCount,
        nonZeroCount,
        &pipe);
    vectorProcessor.Process();
}
