/**
 * @file tcg_spmm_mix_producer.h
 * AIV 侧：仅执行未压缩窗口的 Vector SpMM
 */
#ifndef VECTOR_PROCESSOR_H
#define VECTOR_PROCESSOR_H
#include "kernel_operator.h"

#ifndef VECTOR_PROCESSOR_CONST_DEFINED
#define VECTOR_PROCESSOR_CONST_DEFINED
constexpr uint32_t VECTOR_CUBE_BLOCK = 16;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_TOTAL_SIZE = 192 * 1024;
constexpr uint32_t UB_RESERVED = 2048;
constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint32_t UINT32_PER_ALIGN = ALIGN_BYTES / sizeof(uint32_t);
constexpr uint32_t UB_AVAILABLE = UB_TOTAL_SIZE - UB_RESERVED;
#endif
class AivVectorProcessor {
    public:
        __aicore__ inline void Init(
            GM_ADDR featureData, GM_ADDR outputData, GM_ADDR indptrData, GM_ADDR indicesData, GM_ADDR vectorWindowIdsData, GM_ADDR vectorWinSplitData, uint32_t numDstRows, uint32_t numSrcRows,
            uint32_t featureDim, uint32_t batchCount, uint32_t nonZeroCount, uint32_t vectorWindowCount, AscendC::TPipe *pipe)
        {
            this->M = numDstRows;
            this->K = numSrcRows;
            this->featureDim = featureDim;
            this->batchCount = batchCount;
            this->rowWidth = featureDim;
            this->nnz = nonZeroCount;
            uint32_t blockIdx = AscendC::GetBlockIdx();
            uint32_t blockNum = AscendC::GetBlockNum();
            this->startWinOffset = 0;
            this->endWinOffset = 0;
            vectorWinSplitGm.SetGlobalBuffer((__gm__ uint32_t *)vectorWinSplitData, blockNum + 1);

            this->startWinOffset = vectorWinSplitGm.GetValue(blockIdx);
            this->endWinOffset = vectorWinSplitGm.GetValue(blockIdx + 1);
            this->localWinCount = this->endWinOffset - this->startWinOffset;
            this->winIdsAlignedElements = ((this->localWinCount + UINT32_PER_ALIGN - 1) / UINT32_PER_ALIGN) * UINT32_PER_ALIGN;

            featureGm.SetGlobalBuffer((__gm__ half *)featureData, K * batchCount * featureDim);
            outputGm.SetGlobalBuffer((__gm__ half *)outputData, M * batchCount * featureDim);
            indptrGm.SetGlobalBuffer((__gm__ uint32_t *)indptrData, M + 1);
            indicesGm.SetGlobalBuffer((__gm__ uint32_t *)indicesData, nnz);
            vectorWindowIdsGm.SetGlobalBuffer((__gm__ uint32_t *)vectorWindowIdsData, vectorWindowCount);
            
            uint32_t rowBytes = rowWidth * sizeof(half);
            this->rowAlignedBytes = ((rowBytes + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
            this->rowAlignedElements = this->rowAlignedBytes / sizeof(half);
            uint32_t accumBytes = BUFFER_NUM * rowAlignedBytes;
            uint32_t winIdsBytes = this->winIdsAlignedElements * sizeof(uint32_t);
            uint32_t fixedCost = accumBytes + winIdsBytes;
            uint32_t remainingUb = UB_AVAILABLE > fixedCost ? (UB_AVAILABLE - fixedCost) : 0;
            this->batchSize = remainingUb / (BUFFER_NUM * rowAlignedBytes);
            if (this->batchSize == 0) {
                this->batchSize = 1;
            }
            uint32_t batchBufferSize = this->batchSize * rowAlignedBytes;
            pipe->InitBuffer(accumQueue, BUFFER_NUM, rowAlignedBytes);
            pipe->InitBuffer(featureQueue, BUFFER_NUM, batchBufferSize);
            pipe->InitBuffer(winIdsQueue, 1, winIdsBytes);
        }
    
        __aicore__ inline void Process()
        {
            AscendC::LocalTensor<uint32_t> windowIds = winIdsQueue.AllocTensor<uint32_t>();
            AscendC::DataCopyExtParams copyParams = {1, static_cast<uint32_t>(this->localWinCount * sizeof(uint32_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<uint32_t> padParams = {true, 0, (uint8_t)(this->winIdsAlignedElements - this->localWinCount), 0};
            AscendC::DataCopyPad(windowIds, vectorWindowIdsGm[this->startWinOffset], copyParams, padParams);
            winIdsQueue.EnQue(windowIds);
            AscendC::LocalTensor<uint32_t> localWindowIds = winIdsQueue.DeQue<uint32_t>();
            for (uint32_t localWindowIndex = 0; localWindowIndex < this->localWinCount; ++localWindowIndex) {
                uint32_t windowId = localWindowIds.GetValue(localWindowIndex);
                uint32_t rowStart = windowId * VECTOR_CUBE_BLOCK;
                uint32_t rowEnd = rowStart + VECTOR_CUBE_BLOCK;
                if (rowEnd > this->M) {
                    rowEnd = this->M;
                }
                for (uint32_t row = rowStart; row < rowEnd; ++row) {
                    uint32_t rowStartPtr = indptrGm.GetValue(row);
                    uint32_t rowEndPtr = indptrGm.GetValue(row + 1);
                    uint32_t rowNonZeroCount = rowEndPtr - rowStartPtr;
                    for (uint32_t batchIndex = 0; batchIndex < this->batchCount; ++batchIndex) {
                        AscendC::LocalTensor<half> accumUb = accumQueue.AllocTensor<half>();
                        AscendC::Duplicate<half>(accumUb, half(0.0f), this->rowWidth);

                        if (rowNonZeroCount > 0) {
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
            winIdsQueue.FreeTensor(localWindowIds);
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
                AscendC::Add(accumBlock, accumBlock, computeBatch[featureOffset], this->rowWidth);
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
        uint32_t rowAlignedBytes,rowAlignedElements;
        uint32_t startWinOffset, endWinOffset, localWinCount;
        uint32_t winIdsAlignedElements;
        AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> accumQueue;
        AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> featureQueue;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> winIdsQueue;
        AscendC::GlobalTensor<half> featureGm;
        AscendC::GlobalTensor<half> outputGm;
        AscendC::GlobalTensor<uint32_t> indptrGm, indicesGm;
        AscendC::GlobalTensor<uint32_t> vectorWindowIdsGm, vectorWinSplitGm;
    };

#endif // VECTOR_PROCESSOR_H

