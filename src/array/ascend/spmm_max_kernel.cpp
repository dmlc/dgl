#include <cstdint> 
#include "kernel_operator.h"
constexpr uint32_t CUBE_BLOCK_LENGTH = 16;
constexpr uint32_t CUBE_BLOCK_SIZE = 16 * 16;
constexpr uint32_t CUBE_L0A_BUFFER_BYTES = 32 * 1024;
constexpr uint32_t CUBE_L0B_BUFFER_BYTES = 32 * 1024;
constexpr uint32_t CUBE_L0C_BUFFER_BYTES = 64 * 1024;
constexpr uint32_t VECTOR_CUBE_BLOCK = 16;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_TOTAL_SIZE = 192 * 1024;
constexpr uint32_t UB_RESERVED = 2048;
constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint32_t UINT32_PER_ALIGN = ALIGN_BYTES / sizeof(uint32_t);
constexpr uint32_t UB_AVAILABLE = UB_TOTAL_SIZE - UB_RESERVED;

class spmmMax {
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
        uint32_t nonZeroCount,
        AscendC::TPipe *pipe)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
        this->M = numDstRows;
        this->K = numSrcRows;
        this->N = featureDim;
        this->nnz = nonZeroCount;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        this->startRow = 0;
        this->localRowCount = 0;

        rowSplitGm.SetGlobalBuffer((__gm__ uint32_t *)vectorRowSplitData, blockNum + 1);
        this->startRow = rowSplitGm.GetValue(blockIdx);
        uint32_t endRow = rowSplitGm.GetValue(blockIdx + 1);
        this->localRowCount = endRow - startRow;

        featureGm.SetGlobalBuffer((__gm__ half *)featureData, K * N);
        outputGm.SetGlobalBuffer((__gm__ half *)outputData, M * N);
        indptrGm.SetGlobalBuffer((__gm__ uint32_t *)indptrData, M + 1);
        indicesGm.SetGlobalBuffer((__gm__ uint32_t *)indicesData, nnz);

        this->rowBytes = N * sizeof(half);
        this->rowAlignedBytes = (this->rowBytes + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        this->rowAlignedElems = this->rowAlignedBytes / sizeof(half);
        this->rightPadding = this->rowAlignedElems - N;

        uint32_t accumBytes = BUFFER_NUM * this->rowAlignedBytes;
        uint32_t remainingUb = UB_AVAILABLE > accumBytes ? (UB_AVAILABLE - accumBytes) : 0;
        this->batchSize = remainingUb / (BUFFER_NUM * this->rowAlignedBytes);
        if (this->batchSize == 0) {
            this->batchSize = 1;
        }
        uint32_t batchBufferSize = this->batchSize * this->rowAlignedBytes;
        pipe->InitBuffer(accumQueue, BUFFER_NUM, this->rowAlignedBytes);
        pipe->InitBuffer(featureQueue, BUFFER_NUM, batchBufferSize);
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
            AscendC::LocalTensor<half> accumUb = accumQueue.AllocTensor<half>();

            if (rowNonZeroCount == 0) {
                AscendC::Duplicate<half>(accumUb, half(0.0f), this->N);
            } else {
                AscendC::Duplicate<half>(accumUb, half(-65504.0f), this->N);
                uint32_t batchCount = (rowNonZeroCount + this->batchSize - 1) / this->batchSize;
                for (uint32_t batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
                    uint32_t batchStartPtr = rowStartPtr + batchIndex * this->batchSize;
                    uint32_t currentBatchSize = (batchStartPtr + this->batchSize > rowEndPtr)
                                                ? (rowEndPtr - batchStartPtr)
                                                : this->batchSize;
                    CopyInBatch(batchStartPtr, currentBatchSize);
                    ComputeBatchMax(accumUb, currentBatchSize);
                }
            }
            accumQueue.EnQue(accumUb); 
            CopyOut(row);
        }
    }

private:

    __aicore__ inline void CopyInBatch(uint32_t batchStart, uint32_t batchNnz)
    {
        AscendC::LocalTensor<half> featureBatch = featureQueue.AllocTensor<half>();
        AscendC::DataCopyExtParams copyParams = {1, this->rowBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<half> padParams = {true, 0, (uint8_t)this->rightPadding, half(0.0f)};
        for (uint32_t i = 0; i < batchNnz; ++i) {
            uint32_t neighborIndex = indicesGm.GetValue(batchStart + i);
            AscendC::DataCopyPad<half>(featureBatch[i * this->rowAlignedElems], featureGm[neighborIndex * this->N], copyParams, padParams);
        }
        featureQueue.EnQue(featureBatch);
    }

    __aicore__ inline void ComputeBatchMax(AscendC::LocalTensor<half> &accumBlock, uint32_t batchNnz)
    {
        AscendC::LocalTensor<half> computeBatch = featureQueue.DeQue<half>();
        for (uint32_t i = 0; i < batchNnz; ++i) {
            AscendC::Max(accumBlock, accumBlock, computeBatch[i * this->rowAlignedElems], this->N);
        }
        featureQueue.FreeTensor(computeBatch);
    }

    __aicore__ inline void CopyOut(uint32_t row)
    {
        AscendC::LocalTensor<half> accumBlock = accumQueue.DeQue<half>();
        AscendC::DataCopyExtParams copyParams = {1, this->rowBytes, 0, 0, 0};
        AscendC::DataCopyPad<half>(outputGm[row * this->N], accumBlock, copyParams);
        accumQueue.FreeTensor(accumBlock);
    }

    uint32_t M, K, N, nnz;
    uint32_t batchSize;
    uint32_t rowBytes, rowAlignedBytes, rowAlignedElems, rightPadding;
    uint32_t startRow, localRowCount;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> accumQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> featureQueue;
    AscendC::GlobalTensor<half> featureGm;
    AscendC::GlobalTensor<half> outputGm;
    AscendC::GlobalTensor<uint32_t> indptrGm, indicesGm, rowSplitGm;
};

extern "C" __global__ __aicore__ void spmm_max(
    GM_ADDR featureData,
    GM_ADDR outputData,
    GM_ADDR indptrData,
    GM_ADDR indicesData,
    GM_ADDR vectorRowSplitData,
    uint32_t numDstRows,
    uint32_t numSrcRows,
    uint32_t featureDim,
    uint32_t nonZeroCount)
{
    AscendC::TPipe pipe;
    spmmMax vectorProcessor;
    vectorProcessor.Init(
        featureData,
        outputData,
        indptrData,
        indicesData,
        vectorRowSplitData,
        numDstRows,
        numSrcRows,
        featureDim,
        nonZeroCount,
        &pipe);
    vectorProcessor.Process();
}
