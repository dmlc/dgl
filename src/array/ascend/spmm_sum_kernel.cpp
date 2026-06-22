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

class spmmSumAic {
public:
    __aicore__ inline void Init(
        GM_ADDR denseBlockData, GM_ADDR featureData, GM_ADDR outputData, GM_ADDR cubeWindowIdsData, GM_ADDR cubeWinSplitData, GM_ADDR winEdgePtrData,
        GM_ADDR colToEdgeData, uint32_t numDstRows, uint32_t numSrcRows, uint32_t featureDim, uint32_t totalTcBlocks, uint32_t cubeWindowCount,
        uint32_t columnToEdgeLength, AscendC::TPipe *pipe)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        this->M = numDstRows;
        this->K = numSrcRows;
        this->N = featureDim;
        this->startCubeWin = 0;
        this->localCubeWins = 0;

        cubeWinSplitGm.SetGlobalBuffer((__gm__ uint32_t *)cubeWinSplitData, blockNum + 1);
        this->startCubeWin = cubeWinSplitGm.GetValue(blockIdx);
        this->localCubeWins = cubeWinSplitGm.GetValue(blockIdx + 1) - this->startCubeWin;

        denseBlockGm.SetGlobalBuffer((__gm__ half *)denseBlockData, totalTcBlocks * CUBE_BLOCK_SIZE);  //a压缩后的稠密矩阵
        featureGm.SetGlobalBuffer((__gm__ half *)featureData, K * N);
        cubeWindowIdsGm.SetGlobalBuffer((__gm__ uint32_t *)cubeWindowIdsData, cubeWindowCount);  //窗口索引
        winEdgePtrGm.SetGlobalBuffer((__gm__ uint32_t *)winEdgePtrData, cubeWindowCount + 1);  //每个窗口列维度
        colToEdgeGm.SetGlobalBuffer((__gm__ uint32_t *)colToEdgeData, columnToEdgeLength);  //每个窗口列到边的映射
        outputGm.SetGlobalBuffer((__gm__ half *)outputData, M * N);

        uint32_t prefixTcBlocks = 0;  //该核起始窗口前16X16矩阵的个数
        uint32_t prefixMappedColumns = 0;  //列到边的起始索引位置
        for (uint32_t windowIndex = 0; windowIndex < this->startCubeWin; ++windowIndex) {
            uint32_t windowNonZeroColumns = winEdgePtrGm.GetValue(windowIndex + 1) - winEdgePtrGm.GetValue(windowIndex);
            prefixMappedColumns += windowNonZeroColumns;
            prefixTcBlocks += CeilCubeBlock(windowNonZeroColumns);
        }
        this->startTcOffset = prefixTcBlocks;
        this->startColToEdgeOffset = prefixMappedColumns;

        pipe->InitBuffer(a1Queue, BUFFER_NUM, CUBE_L0A_BUFFER_BYTES);
        pipe->InitBuffer(b1Queue, BUFFER_NUM, CUBE_L0B_BUFFER_BYTES);
        pipe->InitBuffer(a2Queue, BUFFER_NUM, CUBE_L0A_BUFFER_BYTES);
        pipe->InitBuffer(b2Queue, BUFFER_NUM, CUBE_L0B_BUFFER_BYTES);
        pipe->InitBuffer(co1Queue, BUFFER_NUM, CUBE_L0C_BUFFER_BYTES);
    }

    __aicore__ inline void Process()
    {
        uint32_t currentTcOffset = this->startTcOffset;
        uint32_t currentColToEdgeOffset = this->startColToEdgeOffset;
        uint32_t maxKColumns = ComputeMaxKRowsPerSlice();
        // 遍历每一个window
        for (uint32_t localWindowIndex = 0; localWindowIndex < this->localCubeWins; ++localWindowIndex) {
            uint32_t windowIndex = this->startCubeWin + localWindowIndex;
            uint32_t globalWindowId = cubeWindowIdsGm.GetValue(windowIndex);
            uint32_t windowNonZeroColumns = winEdgePtrGm.GetValue(windowIndex + 1) - winEdgePtrGm.GetValue(windowIndex);
            uint32_t windowTcBlockCount = CeilCubeBlock(windowNonZeroColumns);
            uint32_t baseRow = globalWindowId * CUBE_BLOCK_LENGTH;
            uint32_t actualRows = CUBE_BLOCK_LENGTH;
            if (baseRow + actualRows > this->M) {
                actualRows = this->M - baseRow;
            }

            uint32_t computeMaxNCols = ComputeMaxNColsPerSlice(windowNonZeroColumns);
            uint32_t outputMaxCols = ComputeoutputMaxColsPerSlice(actualRows, computeMaxNCols);
            uint32_t outputColStart = 0;
            while (outputColStart < this->N) {
                uint32_t outputChunk = this->N - outputColStart > outputMaxCols ? outputMaxCols : this->N - outputColStart;
                // 每次计算computeMaxNCols列，累积到outputMaxCols再输出
                AscendC::LocalTensor<float> outputAccumulator = co1Queue.AllocTensor<float>();
                for (uint32_t subOutputColOffset = 0; subOutputColOffset < outputChunk; subOutputColOffset += computeMaxNCols) {
                    uint32_t currentNChunk = outputChunk - subOutputColOffset > computeMaxNCols? computeMaxNCols : outputChunk - subOutputColOffset;
                    uint32_t remainingColumns = windowNonZeroColumns;  //a剩余多少列没有参与计算，可能一次性无法计算完windowNonZeroColumns列
                    uint32_t sliceTcOffset = currentTcOffset;  //当前计算TC的起始索引
                    uint32_t sliceColOffset = currentColToEdgeOffset;  //当前列对边映射的起始索引
                    bool initCmatrix = true;
                    while (remainingColumns > 0) {
                        uint32_t currentKColumns = remainingColumns > maxKColumns ? maxKColumns : remainingColumns;
                        uint32_t currentTcBlocks = CeilCubeBlock(currentKColumns);
                        uint32_t currentKRows = currentTcBlocks * CUBE_BLOCK_LENGTH;

                        LoadA1FromWorkspaceNz(sliceTcOffset, currentTcBlocks);
                        SplitAFull(currentKRows);
                        LoadB1FromWorkspaceNz(sliceColOffset, currentKColumns, currentNChunk, outputColStart + subOutputColOffset);
                        SplitBFull(currentKRows, currentNChunk);
                        ComputeFull(currentKRows, currentNChunk, subOutputColOffset, initCmatrix, outputAccumulator);
                        initCmatrix = false;

                        remainingColumns -= currentKColumns;
                        sliceTcOffset += currentTcBlocks;
                        sliceColOffset += currentKColumns;
                    }
                }

                co1Queue.EnQue(outputAccumulator);
                AscendC::LocalTensor<float> outputChunkTensor = co1Queue.DeQue<float>();
                CopyOutFull(baseRow, actualRows, outputColStart, outputChunk, outputChunkTensor);
                co1Queue.FreeTensor(outputChunkTensor);
                outputColStart += outputChunk;
            }

            currentTcOffset += windowTcBlockCount;
            currentColToEdgeOffset += windowNonZeroColumns;
        }
    }

private:
    __aicore__ inline void LoadA1FromWorkspaceNz(uint32_t tcOffset, uint32_t tcBlockCount)
    {
        AscendC::LocalTensor<half> a1Local = a1Queue.AllocTensor<half>();
        uint32_t k = tcBlockCount * CUBE_BLOCK_LENGTH;
        AscendC::Nd2NzParams nd2nzA1Params;
        nd2nzA1Params.ndNum = 1;
        nd2nzA1Params.nValue = CUBE_BLOCK_LENGTH;
        nd2nzA1Params.dValue = k;
        nd2nzA1Params.srcNdMatrixStride = 0;
        nd2nzA1Params.srcDValue = k;
        nd2nzA1Params.dstNzC0Stride = CeilCubeBlock(CUBE_BLOCK_LENGTH) * CUBE_BLOCK_LENGTH;
        nd2nzA1Params.dstNzNStride = 1;
        nd2nzA1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1Local, denseBlockGm[tcOffset * CUBE_BLOCK_SIZE], nd2nzA1Params);
        a1Queue.EnQue(a1Local);
    }

    __aicore__ inline void LoadB1FromWorkspaceNz(uint32_t colToEdgeOffset, uint32_t rows, uint32_t cols, uint32_t ColStart)
    {
        uint32_t k = CeilCubeBlock(rows) * CUBE_BLOCK_LENGTH;
        uint32_t KBlocks = k / CUBE_BLOCK_LENGTH;
        
        uint32_t nBlocksCols = CeilCubeBlock(cols);
        AscendC::LocalTensor<half> b1Nz = b1Queue.AllocTensor<half>();
        AscendC::DataCopyParams copyParams = {(uint16_t)nBlocksCols, 1, 0, (uint16_t)(KBlocks * CUBE_BLOCK_LENGTH - 1)};
        for (uint32_t columnIndex = 0; columnIndex < rows; ++columnIndex) {
            uint32_t kBlockIndex = columnIndex / CUBE_BLOCK_LENGTH;
            uint32_t kInnerIndex = columnIndex % CUBE_BLOCK_LENGTH;
            uint32_t featureRowIndex = colToEdgeGm.GetValue(colToEdgeOffset + columnIndex);
            uint32_t srcOffset = featureRowIndex * this->N + ColStart;
            uint32_t dstOffset = kBlockIndex * CUBE_BLOCK_SIZE + kInnerIndex * CUBE_BLOCK_LENGTH;
            AscendC::DataCopy(b1Nz[dstOffset], featureGm[srcOffset], copyParams);
        }
        
        b1Queue.EnQue<half>(b1Nz);
    }

    __aicore__ inline void SplitAFull(uint32_t kRows)
    {
        AscendC::LocalTensor<half> a1Local = a1Queue.DeQue<half>();
        AscendC::LocalTensor<half> a2Local = a2Queue.AllocTensor<half>();
        uint32_t ceilK = CeilCubeBlock(kRows);
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = (uint8_t)ceilK;
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        AscendC::LoadData(a2Local, a1Local, loadDataParams);
        a2Queue.EnQue<half>(a2Local);
        a1Queue.FreeTensor(a1Local);
    }

    __aicore__ inline void SplitBFull(uint32_t kRows, uint32_t nChunk)
    {
        AscendC::LocalTensor<half> b1Local = b1Queue.DeQue<half>();
        AscendC::LocalTensor<half> b2Local = b2Queue.AllocTensor<half>();
        uint32_t ceilN = CeilCubeBlock(nChunk);
        uint32_t ceilK = CeilCubeBlock(kRows);
        uint32_t dstOffset = ceilN * CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = (uint8_t)ceilN;
        loadDataParams.srcStride = (uint16_t)ceilK;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        for (uint32_t i = 0; i < ceilK; ++i) {
            AscendC::LoadData(b2Local[i * dstOffset], b1Local[i * srcOffset], loadDataParams);
        }
        b2Queue.EnQue<half>(b2Local);
        b1Queue.FreeTensor(b1Local);
    }

    __aicore__ inline void ComputeFull(
        uint32_t kRows, uint32_t nChunk, uint32_t outputChunkOffset, bool initCmatrix, AscendC::LocalTensor<float> &outputAccumulator)
    {
        AscendC::LocalTensor<half> a2Local = a2Queue.DeQue<half>();
        AscendC::LocalTensor<half> b2Local = b2Queue.DeQue<half>();
        AscendC::MmadParams mmadParams;
        mmadParams.m = CUBE_BLOCK_LENGTH;
        mmadParams.n = (uint16_t)nChunk;
        mmadParams.k = (uint16_t)kRows;
        mmadParams.cmatrixInitVal = initCmatrix;
        AscendC::Mmad(outputAccumulator[outputChunkOffset * CUBE_BLOCK_LENGTH], a2Local, b2Local, mmadParams);
        a2Queue.FreeTensor(a2Local);
        b2Queue.FreeTensor(b2Local);
    }

    __aicore__ inline void CopyOutFull(
        uint32_t baseRow, uint32_t actualRows, uint32_t outputColStart, uint32_t outputChunk, AscendC::LocalTensor<float> &outputChunkTensor)
    {
        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = (uint16_t)outputChunk;
        fixpipeParams.mSize = (uint16_t)actualRows;
        fixpipeParams.srcStride = CUBE_BLOCK_LENGTH;
        fixpipeParams.dstStride = this->N;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        fixpipeParams.reluEn = false;
        fixpipeParams.quantPre = QuantMode_t::F322F16;
        AscendC::Fixpipe(outputGm[baseRow * this->N + outputColStart], outputChunkTensor, fixpipeParams);
    }

    __aicore__ inline uint32_t CeilCubeBlock(uint32_t len)
    {
        return (len + CUBE_BLOCK_LENGTH - 1) / CUBE_BLOCK_LENGTH;
    }

    __aicore__ inline uint32_t AlignDownCube(uint32_t len)
    {
        return (len / CUBE_BLOCK_LENGTH) * CUBE_BLOCK_LENGTH;
    }

    __aicore__ inline uint32_t ComputeMaxKRowsPerSlice()
    {
        uint32_t maxHalfElements = CUBE_L0A_BUFFER_BYTES / sizeof(half);
        uint32_t maxKRows = maxHalfElements / CUBE_BLOCK_LENGTH;
        uint32_t alignedKRows = AlignDownCube(maxKRows);
        return alignedKRows;
    }

    __aicore__ inline uint32_t ComputeMaxNColsPerSlice(uint32_t windowNonZeroColumns)
    {
        uint32_t maxHalfElements = CUBE_L0A_BUFFER_BYTES / sizeof(half);
        uint32_t windowKRows = CeilCubeBlock(windowNonZeroColumns) * CUBE_BLOCK_LENGTH;
        uint32_t maxNCols = maxHalfElements / windowKRows;
        uint32_t alignedNCols = AlignDownCube(maxNCols);
        if (alignedNCols > this->N) {
            alignedNCols = this->N;
        }
        return alignedNCols;
    }

    __aicore__ inline uint32_t ComputeoutputMaxColsPerSlice(uint32_t actualRows, uint32_t computeMaxNCols)
    {
        uint32_t maxFloatElements = CUBE_L0C_BUFFER_BYTES / sizeof(float);
        uint32_t maxColsByOutputBuffer = maxFloatElements / actualRows;
        uint32_t cols = maxColsByOutputBuffer;

        if (cols > this->N) {
            cols = this->N;
        }
        cols = (cols / computeMaxNCols) * computeMaxNCols;
        return cols;
    }

    uint32_t M, K, N;
    uint32_t startCubeWin, localCubeWins;
    uint32_t startTcOffset, startColToEdgeOffset;
    
    AscendC::TQue<AscendC::TPosition::A1, BUFFER_NUM> a1Queue;
    AscendC::TQue<AscendC::TPosition::B1, BUFFER_NUM> b1Queue;
    AscendC::TQue<AscendC::TPosition::A2, BUFFER_NUM> a2Queue;
    AscendC::TQue<AscendC::TPosition::B2, BUFFER_NUM> b2Queue;
    AscendC::TQue<AscendC::TPosition::CO1, BUFFER_NUM> co1Queue;

    AscendC::GlobalTensor<half> denseBlockGm;
    AscendC::GlobalTensor<half> featureGm;
    AscendC::GlobalTensor<half> outputGm;
    AscendC::GlobalTensor<uint32_t> cubeWindowIdsGm, cubeWinSplitGm, winEdgePtrGm, colToEdgeGm;
};

class spmmSumAiv {
public:
    __aicore__ inline void Init(
        GM_ADDR featureData, GM_ADDR outputData, GM_ADDR indptrData, GM_ADDR indicesData, GM_ADDR vectorWindowIdsData, GM_ADDR vectorWinSplitData, uint32_t numDstRows, uint32_t numSrcRows,
        uint32_t featureDim, uint32_t nonZeroCount, uint32_t vectorWindowCount, AscendC::TPipe *pipe)
    {
        this->M = numDstRows;
        this->K = numSrcRows;
        this->N = featureDim;
        this->nnz = nonZeroCount;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum() * 2;
        this->startWinOffset = 0;
        this->endWinOffset = 0;
        vectorWinSplitGm.SetGlobalBuffer((__gm__ uint32_t *)vectorWinSplitData, blockNum + 1);

        this->startWinOffset = vectorWinSplitGm.GetValue(blockIdx);
        this->endWinOffset = vectorWinSplitGm.GetValue(blockIdx + 1);
        this->localWinCount = this->endWinOffset - this->startWinOffset;
        this->winIdsAlignedElements = ((this->localWinCount + UINT32_PER_ALIGN - 1) / UINT32_PER_ALIGN) * UINT32_PER_ALIGN;

        featureGm.SetGlobalBuffer((__gm__ half *)featureData, K * N);
        outputGm.SetGlobalBuffer((__gm__ half *)outputData, M * N);
        indptrGm.SetGlobalBuffer((__gm__ uint32_t *)indptrData, M + 1);
        indicesGm.SetGlobalBuffer((__gm__ uint32_t *)indicesData, nnz);
        vectorWindowIdsGm.SetGlobalBuffer((__gm__ uint32_t *)vectorWindowIdsData, vectorWindowCount);
        
        uint32_t rowBytes = N * sizeof(half);
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
                AscendC::LocalTensor<half> accumUb = accumQueue.AllocTensor<half>();
                AscendC::Duplicate<half>(accumUb, half(0.0f), this->N);
                uint32_t rowStartPtr = indptrGm.GetValue(row);
                uint32_t rowEndPtr = indptrGm.GetValue(row + 1);
                uint32_t rowNonZeroCount = rowEndPtr - rowStartPtr;

                if (rowNonZeroCount > 0) {
                    uint32_t batchCount = (rowNonZeroCount + this->batchSize - 1) / this->batchSize;
                    for (uint32_t batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
                        uint32_t batchStartPtr = rowStartPtr + batchIndex * this->batchSize;
                        uint32_t currentBatchSize = (batchStartPtr + this->batchSize > rowEndPtr)
                                                    ? (rowEndPtr - batchStartPtr)
                                                    : this->batchSize;
                        AscendC::LocalTensor<half> featureBatch = featureQueue.AllocTensor<half>();
                        CopyInBatch(featureBatch, batchStartPtr, currentBatchSize);
                        featureQueue.EnQue(featureBatch);

                        AscendC::LocalTensor<half> computeBatch = featureQueue.DeQue<half>();
                        ComputeBatch(accumUb, computeBatch, currentBatchSize);
                        featureQueue.FreeTensor(computeBatch);
                    }
                }
                accumQueue.EnQue(accumUb);
                AscendC::LocalTensor<half> accumBlock = accumQueue.DeQue<half>();
                AscendC::DataCopyExtParams copyParams = {1, static_cast<uint32_t>(this->N * sizeof(half)), 0, 0, 0};
                AscendC::DataCopyPad(outputGm[row * this->N], accumBlock, copyParams);
                accumQueue.FreeTensor(accumBlock);
            }
        }
        winIdsQueue.FreeTensor(localWindowIds);
    }

private:
    __aicore__ inline void CopyInBatch(AscendC::LocalTensor<half> &featureBatch, uint32_t batchStart, uint32_t batchNnz)
    {
        AscendC::DataCopyExtParams copyParams = {1, static_cast<uint32_t>(this->N * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<half> padParams = {true, 0, (uint8_t)(this->rowAlignedElements - this->N), 0};
        for (uint32_t i = 0; i < batchNnz; ++i) {
            uint32_t neighborIndex = indicesGm.GetValue(batchStart + i);
            uint32_t featureOffset = i * this->rowAlignedElements;
            AscendC::DataCopyPad(featureBatch[featureOffset], featureGm[neighborIndex * this->N], copyParams, padParams);
        }
    }

    __aicore__ inline void ComputeBatch(AscendC::LocalTensor<half>& accumBlock, AscendC::LocalTensor<half>& featureBatch, uint32_t batchNnz)
    {
        for (uint32_t i = 0; i < batchNnz; ++i) {
            uint32_t featureOffset = i * this->rowAlignedElements;
            AscendC::Add(accumBlock, accumBlock, featureBatch[featureOffset], this->N);
        }
    }


    uint32_t M, K, N, nnz;
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

extern "C" __global__ __aicore__ void spmm_sum(
    GM_ADDR denseBlockData,
    GM_ADDR featureData,
    GM_ADDR outputData,
    GM_ADDR indptrData,
    GM_ADDR indicesData,
    GM_ADDR vectorWindowIdsData,
    GM_ADDR vectorWinSplitData,
    GM_ADDR cubeWindowIdsData,
    GM_ADDR cubeWinSplitData,
    GM_ADDR winEdgePtrData,
    GM_ADDR colToEdgeData,
    uint32_t numDstRows,
    uint32_t numSrcRows,
    uint32_t featureDim,
    uint32_t nonZeroCount,
    uint32_t totalTcBlocks,
    uint32_t vectorWindowCount,
    uint32_t cubeWindowCount,
    uint32_t columnToEdgeLength)
{
    AscendC::TPipe pipe;
    if ASCEND_IS_AIV {
        spmmSumAiv vectorProcessor;
        vectorProcessor.Init(
            featureData,
            outputData,
            indptrData,
            indicesData,
            vectorWindowIdsData,
            vectorWinSplitData,
            numDstRows,
            numSrcRows,
            featureDim,
            nonZeroCount,
            vectorWindowCount,
            &pipe);
        vectorProcessor.Process();
    }

    if ASCEND_IS_AIC {
        spmmSumAic cubeProcessor;
        cubeProcessor.Init(
            denseBlockData,
            featureData,
            outputData,
            cubeWindowIdsData,
            cubeWinSplitData,
            winEdgePtrData,
            colToEdgeData,
            numDstRows,
            numSrcRows,
            featureDim,
            totalTcBlocks,
            cubeWindowCount,
            columnToEdgeLength,
            &pipe);
        cubeProcessor.Process();
    }
}
