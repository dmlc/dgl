// ============================================================================
// spmm_unified_aiv_kernel.cpp — sum 纯 Vector 备选 kernel (AIV only)
// ============================================================================
//
// 基于 dgl-ascend spmm_sum_aiv_kernel.cpp 模板化重构。
//
// 当稠密窗口较少或 Cube 路径不划算时, 使用纯 Vector 路径处理 sum。
// 也可用于 FP32 sum (避免 FP32 Cube 路径的复杂性)。
//
// 模板参数 DType:
//   - float  (FP32): Add<float> 直接累加
//   - half   (FP16): Cast h→f → Add<float> → Cast f→h (升精度累加)
//
// 入口函数: spmm_unified_aiv (AIV_ONLY, blockDim=40)
// ============================================================================

#include <cstdint>
#include "kernel_operator.h"
#include "spmm_unified_tiling.h"

template <typename DType>
class SpmmUnifiedAivSum {
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
        uint32_t isCopyRhs,
        AscendC::TPipe *pipe)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
        this->M = numDstRows;
        this->K = numSrcRows;
        this->featureDim = featureDim;
        this->batchCount = batchCount;
        this->rowWidth = featureDim;
        this->nnz = nonZeroCount;
        this->isCopyRhs = isCopyRhs;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        this->startRow = 0;
        this->localRowCount = 0;

        rowSplitGm.SetGlobalBuffer((__gm__ uint32_t *)vectorRowSplitData, blockNum + 1);
        this->startRow = rowSplitGm.GetValue(blockIdx);
        uint32_t endRow = rowSplitGm.GetValue(blockIdx + 1);
        this->localRowCount = endRow - startRow;

        featureGm.SetGlobalBuffer((__gm__ DType *)featureData, K * batchCount * featureDim);
        outputGm.SetGlobalBuffer((__gm__ DType *)outputData, M * batchCount * featureDim);
        indptrGm.SetGlobalBuffer((__gm__ uint32_t *)indptrData, M + 1);
        indicesGm.SetGlobalBuffer((__gm__ uint32_t *)indicesData, nnz);

        this->rowBytes = rowWidth * sizeof(DType);
        this->rowAlignedBytes = (this->rowBytes + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        this->rowAlignedElements = this->rowAlignedBytes / sizeof(DType);

        uint32_t accumBytes = BUFFER_NUM * this->rowAlignedBytes;
        uint32_t fp32BufBytes = 0;
        if constexpr (sizeof(DType) == 2) {
            uint32_t rowAlignF32 = ((rowWidth * sizeof(float) + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
            fp32BufBytes = rowAlignF32 * 2;
        }
        uint32_t remainingUb = UB_AVAILABLE > (accumBytes + fp32BufBytes)
                                   ? (UB_AVAILABLE - accumBytes - fp32BufBytes)
                                   : 0;
        this->batchSize = remainingUb / (BUFFER_NUM * this->rowAlignedBytes);
        if (this->batchSize == 0) {
            this->batchSize = 1;
        }
        uint32_t batchBufferSize = this->batchSize * this->rowAlignedBytes;
        pipe->InitBuffer(accumQueue, BUFFER_NUM, this->rowAlignedBytes);
        pipe->InitBuffer(featureQueue, BUFFER_NUM, batchBufferSize);
        if constexpr (sizeof(DType) == 2) {
            uint32_t rowAlignF32 = ((rowWidth * sizeof(float) + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
            pipe->InitBuffer(featF32Buf, rowAlignF32);
            pipe->InitBuffer(accumF32Buf, rowAlignF32);
        }
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
                ProcessRow(row, batchIndex, rowStartPtr, rowEndPtr, rowNonZeroCount);
            }
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t row, uint32_t batchIndex,
                                        uint32_t rowStartPtr, uint32_t rowEndPtr, uint32_t rowNonZeroCount)
    {
        AscendC::LocalTensor<DType> accumUb = accumQueue.AllocTensor<DType>();

        if constexpr (sizeof(DType) == 2) {
            // FP16: 使用 FP32 累加 buffer
            AscendC::LocalTensor<float> accumF32 = accumF32Buf.Get<float>();
            AscendC::Duplicate<float>(accumF32, 0.0f, this->rowWidth);

            if (rowNonZeroCount > 0) {
                uint32_t nnzBatchCount = (rowNonZeroCount + this->batchSize - 1) / this->batchSize;
                for (uint32_t nnzBatchIndex = 0; nnzBatchIndex < nnzBatchCount; ++nnzBatchIndex) {
                    uint32_t batchStartPtr = rowStartPtr + nnzBatchIndex * this->batchSize;
                    uint32_t currentBatchSize = (batchStartPtr + this->batchSize > rowEndPtr)
                                                   ? (rowEndPtr - batchStartPtr)
                                                   : this->batchSize;
                    CopyInBatch(batchStartPtr, currentBatchSize, batchIndex);
                    AscendC::LocalTensor<DType> computeBatch = featureQueue.DeQue<DType>();
                    AscendC::LocalTensor<float> featF32 = featF32Buf.Get<float>();
                    for (uint32_t i = 0; i < currentBatchSize; ++i) {
                        AscendC::Cast<float, DType>(featF32, computeBatch[i * this->rowAlignedElements],
                                                     AscendC::RoundMode::CAST_NONE, this->rowWidth);
                        AscendC::Add<float>(accumF32, accumF32, featF32, this->rowWidth);
                    }
                    featureQueue.FreeTensor(computeBatch);
                }
            }
            AscendC::Cast<DType, float>(accumUb, accumF32, AscendC::RoundMode::CAST_ROUND, this->rowWidth);
        } else {
            // FP32: 直接累加
            AscendC::Duplicate<float>(accumUb, 0.0f, this->rowWidth);
            if (rowNonZeroCount > 0) {
                uint32_t nnzBatchCount = (rowNonZeroCount + this->batchSize - 1) / this->batchSize;
                for (uint32_t nnzBatchIndex = 0; nnzBatchIndex < nnzBatchCount; ++nnzBatchIndex) {
                    uint32_t batchStartPtr = rowStartPtr + nnzBatchIndex * this->batchSize;
                    uint32_t currentBatchSize = (batchStartPtr + this->batchSize > rowEndPtr)
                                                   ? (rowEndPtr - batchStartPtr)
                                                   : this->batchSize;
                    CopyInBatch(batchStartPtr, currentBatchSize, batchIndex);
                    AscendC::LocalTensor<DType> computeBatch = featureQueue.DeQue<DType>();
                    for (uint32_t i = 0; i < currentBatchSize; ++i) {
                        AscendC::Add<float>(accumUb, accumUb, computeBatch[i * this->rowAlignedElements], this->rowWidth);
                    }
                    featureQueue.FreeTensor(computeBatch);
                }
            }
        }

        accumQueue.EnQue(accumUb);
        CopyOut(row, batchIndex);
    }

    __aicore__ inline void CopyInBatch(uint32_t batchStart, uint32_t batchNnz, uint32_t batchIndex)
    {
        AscendC::LocalTensor<DType> featureBatch = featureQueue.AllocTensor<DType>();
        AscendC::DataCopyExtParams copyParams = {1, this->rowBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<DType> padParams = {true, 0, (uint8_t)(this->rowAlignedElements - this->rowWidth), DType(0.0f)};
        for (uint32_t i = 0; i < batchNnz; ++i) {
            if (this->isCopyRhs) {
                // copy_rhs (copy_e_sum): feature is edge feature, read sequentially
                uint32_t featureGmOffset = ((batchStart + i) * this->batchCount + batchIndex) * this->featureDim;
                AscendC::DataCopyPad<DType>(featureBatch[i * this->rowAlignedElements],
                                             featureGm[featureGmOffset], copyParams, padParams);
            } else {
                // copy_lhs (copy_u_sum): feature is node feature, gather by indices
                uint32_t neighborIndex = indicesGm.GetValue(batchStart + i);
                uint32_t featureGmOffset = (neighborIndex * this->batchCount + batchIndex) * this->featureDim;
                AscendC::DataCopyPad<DType>(featureBatch[i * this->rowAlignedElements],
                                             featureGm[featureGmOffset], copyParams, padParams);
            }
        }
        featureQueue.EnQue(featureBatch);
    }

    __aicore__ inline void CopyOut(uint32_t row, uint32_t batchIndex)
    {
        AscendC::LocalTensor<DType> accumBlock = accumQueue.DeQue<DType>();
        AscendC::DataCopyExtParams copyParams = {1, this->rowBytes, 0, 0, 0};
        AscendC::DataCopyPad<DType>(outputGm[(row * this->batchCount + batchIndex) * this->featureDim],
                                     accumBlock, copyParams);
        accumQueue.FreeTensor(accumBlock);
    }

    uint32_t M, K, featureDim, batchCount, rowWidth, nnz;
    uint32_t batchSize;
    uint32_t isCopyRhs = 0;
    uint32_t rowBytes, rowAlignedBytes, rowAlignedElements;
    uint32_t startRow, localRowCount;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> accumQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> featureQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> featF32Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumF32Buf;
    AscendC::GlobalTensor<DType> featureGm;
    AscendC::GlobalTensor<DType> outputGm;
    AscendC::GlobalTensor<uint32_t> indptrGm, indicesGm, rowSplitGm;
};

// ============================================================================
// 入口函数: spmm_unified_aiv (AIV_ONLY, blockDim=40)
// 使用 tiling struct 传递参数
// ============================================================================
extern "C" __global__ __aicore__ void spmm_unified_aiv(
    GM_ADDR featureData,
    GM_ADDR outputData,
    GM_ADDR indptrData,
    GM_ADDR indicesData,
    GM_ADDR vectorRowSplitData,
    GM_ADDR tilingData)
{
    AscendC::TPipe pipe;
    const __gm__ SpmmUnifiedAivTilingData* tiling =
        (const __gm__ SpmmUnifiedAivTilingData*)tilingData;
    uint32_t dtype = tiling->dtype;
    if (dtype == DTYPE_FP32) {
        SpmmUnifiedAivSum<float> processor;
        processor.Init(featureData, outputData, indptrData, indicesData,
                        vectorRowSplitData, tiling->numDstRows, tiling->numSrcRows,
                        tiling->featureDim, tiling->batchCount, tiling->nonZeroCount,
                        tiling->isCopyRhs, &pipe);
        processor.Process();
    } else {
        SpmmUnifiedAivSum<half> processor;
        processor.Init(featureData, outputData, indptrData, indicesData,
                        vectorRowSplitData, tiling->numDstRows, tiling->numSrcRows,
                        tiling->featureDim, tiling->batchCount, tiling->nonZeroCount,
                        tiling->isCopyRhs, &pipe);
        processor.Process();
    }
}
