// ============================================================================
// spmm_unified_sum_kernel.cpp — sum 路径 kernel (AIC Cube + AIV Vector)
// ============================================================================
//
// 基于 dgl-ascend spmm_sum_kernel.cpp + bspmm_sum_kernel.cpp 模板化重构。
//
// 模板参数 DType:
//   - float  (FP32): L0A/L0B/L0C 全 float, Mmad<float,float,float>, Fixpipe NoQuant
//   - half   (FP16): L0A/L0B half, L0C float, Mmad<float,half,half>, Fixpipe F322F16
//
// c0Size (K 方向分形宽度):
//   - float:  c0Size=8  (32B / 4B = 8)
//   - half:   c0Size=16 (32B / 2B = 16)
//
// 入口函数: spmm_unified_sum (AIC+AIV, blockDim=20)
//   按 dtype 参数运行时分支选择 <float> 或 <half> 实例化
// ============================================================================

#include <cstdint>
#include "kernel_operator.h"
#include "spmm_unified_tiling.h"

// ============================================================================
// 模板辅助: c0Size 和 CUBE_BLOCK_SIZE 的编译期计算
// ============================================================================
// c0Size: L0A/L0B 分形 C0 维度大小 (K 方向)
//   - half:  16
//   - float: 8
// CUBE_BLOCK_SIZE: L1 NZ 分形元素数 = CUBE_BLOCK_LENGTH * c0Size
//   - half:  16 * 16 = 256
//   - float: 16 * 8  = 128
// ============================================================================

template <typename DType>
constexpr uint32_t KernelC0Size()
{
    return sizeof(DType) == 4 ? 8 : 16;
}

template <typename DType>
constexpr uint32_t KernelCubeBlockSize()
{
    return CUBE_BLOCK_LENGTH * KernelC0Size<DType>();
}

// ============================================================================
// Cube 处理器 (AIC 侧) — 模板化支持 FP32/FP16
// ============================================================================
template <typename DType>
class SpmmUnifiedSumAic {
public:
    static constexpr uint32_t c0Size = KernelC0Size<DType>();
    static constexpr uint32_t CUBE_BLOCK_SIZE_T = KernelCubeBlockSize<DType>();

    __aicore__ inline void Init(
        GM_ADDR denseBlockData, GM_ADDR featureData, GM_ADDR outputData,
        GM_ADDR cubeWindowIdsData, GM_ADDR cubeWinSplitData, GM_ADDR winEdgePtrData,
        GM_ADDR colToEdgeData,
        uint32_t numDstRows, uint32_t numSrcRows, uint32_t featureDim,
        uint32_t batchCount, uint32_t totalTcBlocks, uint32_t cubeWindowCount,
        uint32_t columnToEdgeLength, AscendC::TPipe *pipe)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        this->M = numDstRows;
        this->K = numSrcRows;
        this->featureDim = featureDim;
        this->batchCount = batchCount;
        this->rowWidth = featureDim;
        this->startCubeWin = 0;
        this->localCubeWins = 0;

        cubeWinSplitGm.SetGlobalBuffer((__gm__ uint32_t *)cubeWinSplitData, blockNum + 1);
        this->startCubeWin = cubeWinSplitGm.GetValue(blockIdx);
        this->localCubeWins = cubeWinSplitGm.GetValue(blockIdx + 1) - this->startCubeWin;

        denseBlockGm.SetGlobalBuffer((__gm__ DType *)denseBlockData, totalTcBlocks * CUBE_BLOCK_SIZE_T);
        featureGm.SetGlobalBuffer((__gm__ DType *)featureData, K * batchCount * featureDim);
        cubeWindowIdsGm.SetGlobalBuffer((__gm__ uint32_t *)cubeWindowIdsData, cubeWindowCount);
        winEdgePtrGm.SetGlobalBuffer((__gm__ uint32_t *)winEdgePtrData, cubeWindowCount + 1);
        colToEdgeGm.SetGlobalBuffer((__gm__ uint32_t *)colToEdgeData, columnToEdgeLength);
        outputGm.SetGlobalBuffer((__gm__ DType *)outputData, M * batchCount * featureDim);

        // 前缀偏移: 遍历前面所有窗口, 累积 startTcOffset 和 startColToEdgeOffset
        uint32_t prefixTcBlocks = 0;
        uint32_t prefixMappedColumns = 0;
        for (uint32_t windowIndex = 0; windowIndex < this->startCubeWin; ++windowIndex) {
            uint32_t windowNonZeroColumns = winEdgePtrGm.GetValue(windowIndex + 1) - winEdgePtrGm.GetValue(windowIndex);
            prefixMappedColumns += windowNonZeroColumns;
            prefixTcBlocks += CeilCubeBlock(windowNonZeroColumns);
        }
        this->startTcOffset = prefixTcBlocks;
        this->startColToEdgeOffset = prefixMappedColumns;

        pipe->InitBuffer(a1Queue, CUBE_BUFFER_NUM, CUBE_L0A_BUFFER_BYTES);
        pipe->InitBuffer(b1Queue, CUBE_BUFFER_NUM, CUBE_L0B_BUFFER_BYTES);
        pipe->InitBuffer(a2Queue, CUBE_BUFFER_NUM, CUBE_L0A_BUFFER_BYTES);
        pipe->InitBuffer(b2Queue, CUBE_BUFFER_NUM, CUBE_L0B_BUFFER_BYTES);
        pipe->InitBuffer(co1Queue, CUBE_BUFFER_NUM, CUBE_L0C_BUFFER_BYTES);
    }

    __aicore__ inline void Process()
    {
        uint32_t currentTcOffset = this->startTcOffset;
        uint32_t currentColToEdgeOffset = this->startColToEdgeOffset;
        uint32_t maxKColumns = ComputeMaxKRowsPerSlice();

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

            for (uint32_t batchIndex = 0; batchIndex < this->batchCount; ++batchIndex) {
                uint32_t computeMaxNCols = ComputeMaxNColsPerSlice(windowNonZeroColumns);
                uint32_t outputMaxCols = ComputeOutputMaxColsPerSlice(actualRows, computeMaxNCols);
                uint32_t outputColStart = 0;
                while (outputColStart < this->rowWidth) {
                    uint32_t outputChunk = this->rowWidth - outputColStart > outputMaxCols
                                               ? outputMaxCols
                                               : this->rowWidth - outputColStart;
                    AscendC::LocalTensor<float> outputAccumulator = co1Queue.AllocTensor<float>();
                    for (uint32_t subOutputColOffset = 0; subOutputColOffset < outputChunk; subOutputColOffset += computeMaxNCols) {
                        uint32_t currentNChunk = outputChunk - subOutputColOffset > computeMaxNCols
                                                     ? computeMaxNCols
                                                     : outputChunk - subOutputColOffset;
                        uint32_t remainingColumns = windowNonZeroColumns;
                        uint32_t sliceTcOffset = currentTcOffset;
                        uint32_t sliceColOffset = currentColToEdgeOffset;
                        bool initCmatrix = true;
                        while (remainingColumns > 0) {
                            uint32_t currentKColumns = remainingColumns > maxKColumns ? maxKColumns : remainingColumns;
                            uint32_t currentTcBlocks = CeilCubeBlock(currentKColumns);
                            uint32_t currentKRows = currentTcBlocks * c0Size;

                            LoadA1FromWorkspaceNz(sliceTcOffset, currentTcBlocks);
                            SplitAFull(currentKRows);
                            LoadB1FromWorkspaceNz(sliceColOffset, currentKColumns, currentNChunk,
                                                   outputColStart + subOutputColOffset, batchIndex);
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
                    CopyOutFull(baseRow, actualRows, batchIndex, outputColStart, outputChunk, outputChunkTensor);
                    co1Queue.FreeTensor(outputChunkTensor);
                    outputColStart += outputChunk;
                }
            }

            currentTcOffset += windowTcBlockCount;
            currentColToEdgeOffset += windowNonZeroColumns;
        }
    }

private:
    __aicore__ inline void LoadA1FromWorkspaceNz(uint32_t tcOffset, uint32_t tcBlockCount)
    {
        AscendC::LocalTensor<DType> a1Local = a1Queue.AllocTensor<DType>();
        uint32_t k = tcBlockCount * c0Size;
        AscendC::Nd2NzParams nd2nzA1Params;
        nd2nzA1Params.ndNum = 1;
        nd2nzA1Params.nValue = CUBE_BLOCK_LENGTH;
        nd2nzA1Params.dValue = k;
        nd2nzA1Params.srcNdMatrixStride = 0;
        nd2nzA1Params.srcDValue = k;
        // dstNzC0Stride = CeilAlign(M, BLOCK_CUBE) = 16 (M=16 时)
        nd2nzA1Params.dstNzC0Stride = CUBE_BLOCK_LENGTH;
        nd2nzA1Params.dstNzNStride = 1;
        nd2nzA1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1Local, denseBlockGm[tcOffset * CUBE_BLOCK_SIZE_T], nd2nzA1Params);
        a1Queue.EnQue(a1Local);
    }

    __aicore__ inline void LoadB1FromWorkspaceNz(uint32_t colToEdgeOffset, uint32_t rows, uint32_t cols,
                                                    uint32_t colStart, uint32_t batchIndex)
    {
        uint32_t k = CeilCubeBlock(rows) * c0Size;
        uint32_t kBlocks = k / c0Size;

        uint32_t nBlocksCols = CeilCubeBlock(cols);  // N 方向 block 数, 每个 16 元素
        AscendC::LocalTensor<DType> b1Nz = b1Queue.AllocTensor<DType>();

        // 对每个 K-column, 将其 N 行数据 (cols 个元素) 复制到 L1 NZ 布局
        // FP16: 每个 N-block = 16 half = 32B = 1 DataCopy block, 可一次复制全部 N-block
        // FP32: 每个 N-block = 16 float = 64B = 2 DataCopy block, 需逐 N-block 复制
        if constexpr (sizeof(DType) == 2) {
            // FP16: 原始方式 — 一次 DataCopy 复制所有 N-block
            AscendC::DataCopyParams copyParams = {(uint16_t)nBlocksCols, 1, 0,
                                                    (uint16_t)(kBlocks * CUBE_BLOCK_LENGTH - 1)};
            for (uint32_t columnIndex = 0; columnIndex < rows; ++columnIndex) {
                uint32_t kBlockIndex = columnIndex / c0Size;
                uint32_t kInnerIndex = columnIndex % c0Size;
                uint32_t featureRowIndex = colToEdgeGm.GetValue(colToEdgeOffset + columnIndex);
                uint32_t srcOffset = (featureRowIndex * this->batchCount + batchIndex) * this->featureDim + colStart;
                uint32_t dstOffset = kBlockIndex * CUBE_BLOCK_SIZE_T + kInnerIndex * CUBE_BLOCK_LENGTH;
                AscendC::DataCopy(b1Nz[dstOffset], featureGm[srcOffset], copyParams);
            }
        } else {
            // FP32: 逐 N-block 复制 (每个 N-block = 16 float = 2 DataCopy blocks)
            uint32_t nBlockStride = kBlocks * CUBE_BLOCK_SIZE_T;  // N-block 间距 (元素)
            AscendC::DataCopyParams copyParams = {2, 1, 0, 0};    // 2 blocks × 32B = 64B = 16 float
            for (uint32_t columnIndex = 0; columnIndex < rows; ++columnIndex) {
                uint32_t kBlockIndex = columnIndex / c0Size;
                uint32_t kInnerIndex = columnIndex % c0Size;
                uint32_t featureRowIndex = colToEdgeGm.GetValue(colToEdgeOffset + columnIndex);
                uint32_t srcOffset = (featureRowIndex * this->batchCount + batchIndex) * this->featureDim + colStart;
                uint32_t dstOffset = kBlockIndex * CUBE_BLOCK_SIZE_T + kInnerIndex * CUBE_BLOCK_LENGTH;
                for (uint32_t nBlock = 0; nBlock < nBlocksCols; ++nBlock) {
                    AscendC::DataCopy(b1Nz[dstOffset + nBlock * nBlockStride],
                                       featureGm[srcOffset + nBlock * CUBE_BLOCK_LENGTH], copyParams);
                }
            }
        }

        // Each K row occupies 16 values in B1.  MMAD consumes the padded K,
        // therefore the rows beyond the real K size must contain zeros.
        if (rows < k) {
            constexpr uint32_t blocksPerKRow =
                CUBE_BLOCK_LENGTH * sizeof(DType) / ALIGN_BYTES;
            AscendC::InitConstValueParams<DType> zeroParams;
            zeroParams.repeatTimes = static_cast<uint16_t>(nBlocksCols);
            zeroParams.blockNum = static_cast<uint16_t>((k - rows) * blocksPerKRow);
            zeroParams.dstGap = static_cast<uint16_t>(rows * blocksPerKRow);
            zeroParams.initValue = DType(0);
            AscendC::Fill(b1Nz[rows * CUBE_BLOCK_LENGTH], zeroParams);
        }

        b1Queue.EnQue<DType>(b1Nz);
    }

    __aicore__ inline void SplitAFull(uint32_t kRows)
    {
        AscendC::LocalTensor<DType> a1Local = a1Queue.DeQue<DType>();
        AscendC::LocalTensor<DType> a2Local = a2Queue.AllocTensor<DType>();
        uint32_t ceilK = CeilCubeBlock(kRows);
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = (uint8_t)ceilK;
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        AscendC::LoadData(a2Local, a1Local, loadDataParams);
        a2Queue.EnQue<DType>(a2Local);
        a1Queue.FreeTensor(a1Local);
    }

    __aicore__ inline void SplitBFull(uint32_t kRows, uint32_t nChunk)
    {
        AscendC::LocalTensor<DType> b1Local = b1Queue.DeQue<DType>();
        AscendC::LocalTensor<DType> b2Local = b2Queue.AllocTensor<DType>();
        uint32_t ceilN = CeilCubeBlock(nChunk);   // N 方向 block 数 (每个 16 元素)
        uint32_t ceilK = CeilCubeBlock(kRows);    // K 方向 block 数 (c0Size 元素)
        uint32_t dstOffset = ceilN * CUBE_BLOCK_SIZE_T;
        uint32_t srcOffset = CUBE_BLOCK_SIZE_T;
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = (uint8_t)ceilN;
        loadDataParams.srcStride = (uint16_t)ceilK;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        for (uint32_t i = 0; i < ceilK; ++i) {
            AscendC::LoadData(b2Local[i * dstOffset], b1Local[i * srcOffset], loadDataParams);
        }
        b2Queue.EnQue<DType>(b2Local);
        b1Queue.FreeTensor(b1Local);
    }

    __aicore__ inline void ComputeFull(
        uint32_t kRows, uint32_t nChunk, uint32_t outputChunkOffset, bool initCmatrix,
        AscendC::LocalTensor<float> &outputAccumulator)
    {
        AscendC::LocalTensor<DType> a2Local = a2Queue.DeQue<DType>();
        AscendC::LocalTensor<DType> b2Local = b2Queue.DeQue<DType>();
        AscendC::MmadParams mmadParams;
        mmadParams.m = CUBE_BLOCK_LENGTH;
        mmadParams.n = (uint16_t)nChunk;
        mmadParams.k = (uint16_t)kRows;
        mmadParams.cmatrixInitVal = initCmatrix;
        // FP32: Mmad<float, float, float> (由 LocalTensor 类型推导)
        // FP16: Mmad<float, half, half>  (由 LocalTensor 类型推导)
        AscendC::Mmad(outputAccumulator[outputChunkOffset * CUBE_BLOCK_LENGTH], a2Local, b2Local, mmadParams);
        a2Queue.FreeTensor(a2Local);
        b2Queue.FreeTensor(b2Local);
    }

    __aicore__ inline void CopyOutFull(
        uint32_t baseRow, uint32_t actualRows, uint32_t batchIndex, uint32_t outputColStart,
        uint32_t outputChunk, AscendC::LocalTensor<float> &outputChunkTensor)
    {
        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = (uint16_t)outputChunk;
        fixpipeParams.mSize = (uint16_t)actualRows;
        fixpipeParams.srcStride = CUBE_BLOCK_LENGTH;
        fixpipeParams.dstStride = this->batchCount * this->featureDim;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        fixpipeParams.reluEn = false;
        // FP32: NoQuant (float→float)
        // FP16: F322F16 (float→half)
        if constexpr (sizeof(DType) == 2) {
            fixpipeParams.quantPre = QuantMode_t::F322F16;
        } else {
            fixpipeParams.quantPre = QuantMode_t::NoQuant;
        }
        AscendC::Fixpipe(outputGm[(baseRow * this->batchCount + batchIndex) * this->featureDim + outputColStart],
                          outputChunkTensor, fixpipeParams);
    }

    __aicore__ inline uint32_t CeilCubeBlock(uint32_t len)
    {
        return (len + c0Size - 1) / c0Size;
    }

    __aicore__ inline uint32_t AlignDownCube(uint32_t len)
    {
        return (len / c0Size) * c0Size;
    }

    __aicore__ inline uint32_t ComputeMaxKRowsPerSlice()
    {
        uint32_t maxElements = CUBE_L0A_BUFFER_BYTES / sizeof(DType);
        uint32_t maxKRows = maxElements / CUBE_BLOCK_LENGTH;
        uint32_t alignedKRows = AlignDownCube(maxKRows);
        return alignedKRows;
    }

    __aicore__ inline uint32_t ComputeMaxNColsPerSlice(uint32_t windowNonZeroColumns)
    {
        uint32_t maxElements = CUBE_L0A_BUFFER_BYTES / sizeof(DType);
        uint32_t windowKRows = CeilCubeBlock(windowNonZeroColumns) * c0Size;
        uint32_t maxNCols = maxElements / windowKRows;
        uint32_t alignedNCols = AlignDownCube(maxNCols);
        if (alignedNCols > this->rowWidth) {
            alignedNCols = this->rowWidth;
        }
        if (alignedNCols == 0) {
            alignedNCols = this->rowWidth < CUBE_BLOCK_LENGTH ? this->rowWidth : CUBE_BLOCK_LENGTH;
        }
        return alignedNCols;
    }

    __aicore__ inline uint32_t ComputeOutputMaxColsPerSlice(uint32_t actualRows, uint32_t computeMaxNCols)
    {
        uint32_t maxFloatElements = CUBE_L0C_BUFFER_BYTES / sizeof(float);
        uint32_t maxColsByOutputBuffer = maxFloatElements / actualRows;
        uint32_t cols = maxColsByOutputBuffer;

        if (cols > this->rowWidth) {
            cols = this->rowWidth;
        }
        cols = (cols / computeMaxNCols) * computeMaxNCols;
        if (cols == 0) {
            cols = computeMaxNCols > this->rowWidth ? this->rowWidth : computeMaxNCols;
        }
        return cols;
    }

    uint32_t M, K, featureDim, batchCount, rowWidth;
    uint32_t startCubeWin, localCubeWins;
    uint32_t startTcOffset, startColToEdgeOffset;

    AscendC::TQue<AscendC::TPosition::A1, CUBE_BUFFER_NUM> a1Queue;
    AscendC::TQue<AscendC::TPosition::B1, CUBE_BUFFER_NUM> b1Queue;
    AscendC::TQue<AscendC::TPosition::A2, CUBE_BUFFER_NUM> a2Queue;
    AscendC::TQue<AscendC::TPosition::B2, CUBE_BUFFER_NUM> b2Queue;
    AscendC::TQue<AscendC::TPosition::CO1, CUBE_BUFFER_NUM> co1Queue;

    AscendC::GlobalTensor<DType> denseBlockGm;
    AscendC::GlobalTensor<DType> featureGm;
    AscendC::GlobalTensor<DType> outputGm;
    AscendC::GlobalTensor<uint32_t> cubeWindowIdsGm, cubeWinSplitGm, winEdgePtrGm, colToEdgeGm;
};

// ============================================================================
// Vector 处理器 (AIV 侧) — 处理稀疏窗口, 模板化支持 FP32/FP16
// ============================================================================
// DESIGN.md §2.4.2:
//   FP32: Add<float> 直接累加
//   FP16: Cast h→f → Add<float> → Cast f→h (升精度累加避免溢出)
// ============================================================================
template <typename DType>
class SpmmUnifiedSumAiv {
public:
    __aicore__ inline void Init(
        GM_ADDR featureData, GM_ADDR outputData, GM_ADDR indptrData, GM_ADDR indicesData,
        GM_ADDR vectorWindowIdsData, GM_ADDR vectorWinSplitData,
        uint32_t numDstRows, uint32_t numSrcRows, uint32_t featureDim, uint32_t batchCount,
        uint32_t nonZeroCount, uint32_t vectorWindowCount, AscendC::TPipe *pipe)
    {
        this->M = numDstRows;
        this->K = numSrcRows;
        this->featureDim = featureDim;
        this->batchCount = batchCount;
        this->rowWidth = featureDim;
        this->nnz = nonZeroCount;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum() * 2;  // AIC+AIV 模式下 AIV 数 = AIC 数 × 2
        this->startWinOffset = 0;
        this->endWinOffset = 0;
        vectorWinSplitGm.SetGlobalBuffer((__gm__ uint32_t *)vectorWinSplitData, blockNum + 1);

        this->startWinOffset = vectorWinSplitGm.GetValue(blockIdx);
        this->endWinOffset = vectorWinSplitGm.GetValue(blockIdx + 1);
        this->localWinCount = this->endWinOffset - this->startWinOffset;
        this->winIdsAlignedElements = ((this->localWinCount + UINT32_PER_ALIGN - 1) / UINT32_PER_ALIGN) * UINT32_PER_ALIGN;

        featureGm.SetGlobalBuffer((__gm__ DType *)featureData, K * batchCount * featureDim);
        outputGm.SetGlobalBuffer((__gm__ DType *)outputData, M * batchCount * featureDim);
        indptrGm.SetGlobalBuffer((__gm__ uint32_t *)indptrData, M + 1);
        indicesGm.SetGlobalBuffer((__gm__ uint32_t *)indicesData, nnz);
        vectorWindowIdsGm.SetGlobalBuffer((__gm__ uint32_t *)vectorWindowIdsData, vectorWindowCount);

        uint32_t rowBytes = rowWidth * sizeof(DType);
        this->rowAlignedBytes = ((rowBytes + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
        this->rowAlignedElements = this->rowAlignedBytes / sizeof(DType);

        uint32_t accumBytes = BUFFER_NUM * rowAlignedBytes;
        uint32_t winIdsBytes = this->winIdsAlignedElements * sizeof(uint32_t);
        uint32_t fp32BufBytes = 0;
        // FP16 需要 FP32 中间 buffer (featF32Buf + accumF32Buf)
        if constexpr (sizeof(DType) == 2) {
            uint32_t rowAlignF32 = ((rowWidth * sizeof(float) + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
            fp32BufBytes = rowAlignF32 * 2;  // featF32Buf + accumF32Buf
        }
        uint32_t fixedCost = accumBytes + winIdsBytes + fp32BufBytes;
        uint32_t remainingUb = UB_AVAILABLE > fixedCost ? (UB_AVAILABLE - fixedCost) : 0;
        this->batchSize = remainingUb / (BUFFER_NUM * rowAlignedBytes);
        if (this->batchSize == 0) {
            this->batchSize = 1;
        }
        uint32_t batchBufferSize = this->batchSize * rowAlignedBytes;
        pipe->InitBuffer(accumQueue, BUFFER_NUM, rowAlignedBytes);
        pipe->InitBuffer(featureQueue, BUFFER_NUM, batchBufferSize);
        pipe->InitBuffer(winIdsQueue, 1, winIdsBytes);
        // FP16: 额外的 FP32 中间 buffer
        if constexpr (sizeof(DType) == 2) {
            uint32_t rowAlignF32 = ((rowWidth * sizeof(float) + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
            // TBuf 用单参数 InitBuffer
            pipe->InitBuffer(featF32Buf, rowAlignF32);
            pipe->InitBuffer(accumF32Buf, rowAlignF32);
        }
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
                    ProcessRow(row, batchIndex, rowStartPtr, rowEndPtr, rowNonZeroCount);
                }
            }
        }
        winIdsQueue.FreeTensor(localWindowIds);
    }

private:
    __aicore__ inline void ProcessRow(uint32_t row, uint32_t batchIndex,
                                        uint32_t rowStartPtr, uint32_t rowEndPtr, uint32_t rowNonZeroCount)
    {
        AscendC::LocalTensor<DType> accumUb = accumQueue.AllocTensor<DType>();
        AscendC::Duplicate<DType>(accumUb, DType(0.0f), this->rowWidth);

        // FP16: 初始化 FP32 累加 buffer
        if constexpr (sizeof(DType) == 2) {
            AscendC::LocalTensor<float> accumF32 = accumF32Buf.Get<float>();
            AscendC::Duplicate<float>(accumF32, 0.0f, this->rowWidth);
        }

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

        // FP16: Cast accumF32 → accum(half)
        if constexpr (sizeof(DType) == 2) {
            AscendC::LocalTensor<float> accumF32 = accumF32Buf.Get<float>();
            AscendC::Cast<DType, float>(accumUb, accumF32, AscendC::RoundMode::CAST_ROUND, this->rowWidth);
        }

        accumQueue.EnQue(accumUb);
        CopyOut(row, batchIndex);
    }

    __aicore__ inline void CopyInBatch(uint32_t batchStart, uint32_t batchNnz, uint32_t batchIndex)
    {
        AscendC::LocalTensor<DType> featureBatch = featureQueue.AllocTensor<DType>();
        AscendC::DataCopyExtParams copyParams = {1, static_cast<uint32_t>(this->rowWidth * sizeof(DType)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DType> padParams = {true, 0, (uint8_t)(this->rowAlignedElements - this->rowWidth), 0};
        for (uint32_t i = 0; i < batchNnz; ++i) {
            uint32_t neighborIndex = indicesGm.GetValue(batchStart + i);
            uint32_t featureOffset = i * this->rowAlignedElements;
            uint32_t featureGmOffset = (neighborIndex * this->batchCount + batchIndex) * this->featureDim;
            AscendC::DataCopyPad(featureBatch[featureOffset], featureGm[featureGmOffset], copyParams, padParams);
        }
        featureQueue.EnQue(featureBatch);
    }

    __aicore__ inline void ComputeBatch(AscendC::LocalTensor<DType>& accumBlock, uint32_t batchNnz)
    {
        AscendC::LocalTensor<DType> computeBatch = featureQueue.DeQue<DType>();
        if constexpr (sizeof(DType) == 2) {
            // FP16: Cast h→f → Add<float> → 存入 accumF32Buf
            AscendC::LocalTensor<float> accumF32 = accumF32Buf.Get<float>();
            AscendC::LocalTensor<float> featF32 = featF32Buf.Get<float>();
            for (uint32_t i = 0; i < batchNnz; ++i) {
                uint32_t featureOffset = i * this->rowAlignedElements;
                AscendC::Cast<float, DType>(featF32, computeBatch[featureOffset], AscendC::RoundMode::CAST_NONE, this->rowWidth);
                AscendC::Add<float>(accumF32, accumF32, featF32, this->rowWidth);
            }
        } else {
            // FP32: Add<float> 直接
            for (uint32_t i = 0; i < batchNnz; ++i) {
                uint32_t featureOffset = i * this->rowAlignedElements;
                AscendC::Add<float>(accumBlock, accumBlock, computeBatch[featureOffset], this->rowWidth);
            }
        }
        featureQueue.FreeTensor(computeBatch);
    }

    __aicore__ inline void CopyOut(uint32_t row, uint32_t batchIndex)
    {
        AscendC::LocalTensor<DType> accumBlock = accumQueue.DeQue<DType>();
        AscendC::DataCopyExtParams copyParams = {1, static_cast<uint32_t>(this->rowWidth * sizeof(DType)), 0, 0, 0};
        AscendC::DataCopyPad(outputGm[(row * this->batchCount + batchIndex) * this->featureDim], accumBlock, copyParams);
        accumQueue.FreeTensor(accumBlock);
    }

    uint32_t M, K, featureDim, batchCount, rowWidth, nnz;
    uint32_t batchSize;
    uint32_t rowAlignedBytes, rowAlignedElements;
    uint32_t startWinOffset, endWinOffset, localWinCount;
    uint32_t winIdsAlignedElements;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> accumQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> featureQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> winIdsQueue;
    // FP16 专用 FP32 中间 buffer
    AscendC::TBuf<AscendC::TPosition::VECCALC> featF32Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumF32Buf;
    AscendC::GlobalTensor<DType> featureGm;
    AscendC::GlobalTensor<DType> outputGm;
    AscendC::GlobalTensor<uint32_t> indptrGm, indicesGm;
    AscendC::GlobalTensor<uint32_t> vectorWindowIdsGm, vectorWinSplitGm;
};

// ============================================================================
// 入口函数: spmm_unified_sum (AIC+AIV, blockDim=20)
// 使用 tiling struct 传递 scalar 参数 (避免过多参数导致注册失败)
// ============================================================================
extern "C" __global__ __aicore__ void spmm_unified_sum(
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
    GM_ADDR tilingData)
{
    AscendC::TPipe pipe;
    const __gm__ SpmmUnifiedSumTilingData* tiling =
        (const __gm__ SpmmUnifiedSumTilingData*)tilingData;
    uint32_t dtype = tiling->dtype;
    if ASCEND_IS_AIV {
        if (dtype == DTYPE_FP32) {
            SpmmUnifiedSumAiv<float> vectorProcessor;
            vectorProcessor.Init(
                featureData, outputData, indptrData, indicesData,
                vectorWindowIdsData, vectorWinSplitData,
                tiling->numDstRows, tiling->numSrcRows, tiling->featureDim,
                tiling->batchCount, tiling->nonZeroCount,
                tiling->vectorWindowCount, &pipe);
            vectorProcessor.Process();
        } else {
            SpmmUnifiedSumAiv<half> vectorProcessor;
            vectorProcessor.Init(
                featureData, outputData, indptrData, indicesData,
                vectorWindowIdsData, vectorWinSplitData,
                tiling->numDstRows, tiling->numSrcRows, tiling->featureDim,
                tiling->batchCount, tiling->nonZeroCount,
                tiling->vectorWindowCount, &pipe);
            vectorProcessor.Process();
        }
    }

    if ASCEND_IS_AIC {
        if (dtype == DTYPE_FP32) {
            SpmmUnifiedSumAic<float> cubeProcessor;
            cubeProcessor.Init(
                denseBlockData, featureData, outputData,
                cubeWindowIdsData, cubeWinSplitData, winEdgePtrData,
                colToEdgeData,
                tiling->numDstRows, tiling->numSrcRows, tiling->featureDim,
                tiling->batchCount,
                tiling->totalTcBlocks, tiling->cubeWindowCount,
                tiling->columnToEdgeLength, &pipe);
            cubeProcessor.Process();
        } else {
            SpmmUnifiedSumAic<half> cubeProcessor;
            cubeProcessor.Init(
                denseBlockData, featureData, outputData,
                cubeWindowIdsData, cubeWinSplitData, winEdgePtrData,
                colToEdgeData,
                tiling->numDstRows, tiling->numSrcRows, tiling->featureDim,
                tiling->batchCount,
                tiling->totalTcBlocks, tiling->cubeWindowCount,
                tiling->columnToEdgeLength, &pipe);
            cubeProcessor.Process();
        }
    }
}
