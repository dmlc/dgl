#ifndef SEGMENT_MM_CUBE_PROCESSOR_H
#define SEGMENT_MM_CUBE_PROCESSOR_H
#include "kernel_operator.h"

// ===== Shared Cube (AIC) infrastructure for SegmentMM / SegmentMMBackwardB =====
//
// FP32 transpose note: LoadData2DParams.ifTranspose does NOT support float
// (Load2D.md 表3: ifTranspose 仅支持 uint16_t/int16_t/half). FP32 B/A transpose
// uses LoadData3DParamsV2.enTranspose=true (img2col path); see design §4.4/§7.

constexpr uint32_t SEGMM_BLOCK_LENGTH = 16;          // Cube M/N fractal side
constexpr uint32_t SEGMM_C0_FLOAT = 8;                // 32B / sizeof(float)
constexpr uint32_t SEGMM_FRACTAL_SIZE = 16 * 8;       // L0A/L0B fractal element count (float)
constexpr uint32_t SEGMM_BUFFER_NUM = 2;             // double buffer (O1)
constexpr uint32_t SEGMM_L0A_BUFFER_BYTES = 32 * 1024;
constexpr uint32_t SEGMM_L0B_BUFFER_BYTES = 32 * 1024;
constexpr uint32_t SEGMM_L0C_BUFFER_BYTES = 64 * 1024;

// ===========================================================================
// SegmentMM forward: C_seg[m,n] = A_seg[m,k] @ B[r] (B_trans=false, B=[K,N])
//                                            or A_seg @ B[r]^T (B_trans=true, B=[N,K])
// ===========================================================================
class SegmentMmCubeProcessor {
public:
    __aicore__ inline void Init(
        GM_ADDR aData, GM_ADDR bData, GM_ADDR cData,
        uint32_t m, uint32_t k, uint32_t n,
        uint32_t mGrid, uint32_t nGrid,
        uint32_t nTile, uint32_t kTile, uint32_t bTrans,
        AscendC::TPipe *pipe)
    {
        this->M = m; this->K = k; this->N = n;
        this->mGrid = mGrid; this->nGrid = nGrid;
        this->nTile = nTile; this->kTile = kTile;
        this->bTrans = (bTrans != 0);

        aGm.SetGlobalBuffer((__gm__ float *)aData, static_cast<uint64_t>(m) * k);
        bGm.SetGlobalBuffer((__gm__ float *)bData, static_cast<uint64_t>(k) * n);  // size hint; actual layout depends on bTrans
        cGm.SetGlobalBuffer((__gm__ float *)cData, static_cast<uint64_t>(m) * n);

        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->nIdx = blockIdx % nGrid;
        this->mIdx = blockIdx / nGrid;

        pipe->InitBuffer(a1Queue, SEGMM_BUFFER_NUM, SEGMM_L0A_BUFFER_BYTES);
        pipe->InitBuffer(b1Queue, SEGMM_BUFFER_NUM, SEGMM_L0B_BUFFER_BYTES);
        pipe->InitBuffer(a2Queue, SEGMM_BUFFER_NUM, SEGMM_L0A_BUFFER_BYTES);
        pipe->InitBuffer(b2Queue, SEGMM_BUFFER_NUM, SEGMM_L0B_BUFFER_BYTES);
        pipe->InitBuffer(co1Queue, SEGMM_BUFFER_NUM, SEGMM_L0C_BUFFER_BYTES);
    }

    __aicore__ inline void Process()
    {
        // Idle cores (blockIdx >= mGrid*nGrid) return early.
        if (mIdx >= mGrid) return;

        uint32_t mBlocks = CeilDiv(M, SEGMM_BLOCK_LENGTH);
        uint32_t mBlocksPerCore = CeilDiv(mBlocks, mGrid);
        uint32_t myMBlockLo = mIdx * mBlocksPerCore;
        uint32_t myMBlockHi = Min(myMBlockLo + mBlocksPerCore, mBlocks);
        if (myMBlockLo >= myMBlockHi) return;

        uint32_t nPerCore = CeilDiv(N, nGrid);
        uint32_t myNLo = nIdx * nPerCore;
        uint32_t myNHi = Min(myNLo + nPerCore, N);
        if (myNLo >= myNHi) return;

        // Iterate over this core's M-blocks (16 rows each) and N-tiles.
        for (uint32_t mBlk = myMBlockLo; mBlk < myMBlockHi; ++mBlk) {
            uint32_t mOff = mBlk * SEGMM_BLOCK_LENGTH;
            uint32_t actualRows = Min(SEGMM_BLOCK_LENGTH, M - mOff);

            for (uint32_t nOff = myNLo; nOff < myNHi; nOff += nTile) {
                uint32_t nChunk = Min(nTile, myNHi - nOff);

                AscendC::LocalTensor<float> co1Local = co1Queue.AllocTensor<float>();
                bool initC = true;

                // K-direction split-and-accumulate (O3: cmatrixInitVal=false across k_tile)
                for (uint32_t kOff = 0; kOff < K; kOff += kTile) {
                    uint32_t kChunk = Min(kTile, K - kOff);

                    LoadA1(mOff, kOff, actualRows, kChunk);
                    SplitA(kChunk);

                    LoadB1(kOff, nOff, kChunk, nChunk);
                    if (bTrans) {
                        SplitBNoTranspose(kChunk, nChunk);
                    } else {
                        SplitBTranspose(kChunk, nChunk);
                    }

                    Compute(actualRows, nChunk, kChunk, initC, co1Local);
                    initC = false;
                }

                co1Queue.EnQue(co1Local);
                AscendC::LocalTensor<float> co1Out = co1Queue.DeQue<float>();
                CopyOut(mOff, nOff, actualRows, nChunk, co1Out);
                co1Queue.FreeTensor(co1Out);
            }
        }
    }

private:
    // A1: Nd2Nz GM(ND [actualRows, kChunk]) -> UB(NZ). A is [M, K] row-major.
    __aicore__ inline void LoadA1(uint32_t mOff, uint32_t kOff, uint32_t actualRows, uint32_t kChunk)
    {
        AscendC::LocalTensor<float> a1Local = a1Queue.AllocTensor<float>();
        AscendC::Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = static_cast<uint16_t>(actualRows);
        p.dValue = static_cast<uint16_t>(kChunk);
        p.srcNdMatrixStride = 0;
        p.srcDValue = static_cast<uint16_t>(K);                 // GM row stride for A is K
        p.dstNzC0Stride = static_cast<uint16_t>(CeilAlign(actualRows, SEGMM_BLOCK_LENGTH));
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1Local, aGm[static_cast<uint64_t>(mOff) * K + kOff], p);
        a1Queue.EnQue(a1Local);
    }

    // A1(NZ) -> A2(ZZ), A not transposed: plain LoadData2D (ifTranspose=false).
    __aicore__ inline void SplitA(uint32_t kChunk)
    {
        AscendC::LocalTensor<float> a1Local = a1Queue.DeQue<float>();
        AscendC::LocalTensor<float> a2Local = a2Queue.AllocTensor<float>();
        AscendC::LoadData2DParams lp;
        lp.repeatTimes = static_cast<uint8_t>(CeilDiv(kChunk, SEGMM_C0_FLOAT));
        lp.srcStride = 1;
        lp.dstGap = 0;
        lp.ifTranspose = false;
        AscendC::LoadData(a2Local, a1Local, lp);
        a2Queue.EnQue<float>(a2Local);
        a1Queue.FreeTensor(a1Local);
    }

    // B1: Nd2Nz GM(ND) -> UB(NZ). Layout depends on bTrans.
    __aicore__ inline void LoadB1(uint32_t kOff, uint32_t nOff, uint32_t kChunk, uint32_t nChunk)
    {
        AscendC::LocalTensor<float> b1Local = b1Queue.AllocTensor<float>();
        AscendC::Nd2NzParams p;
        p.ndNum = 1;
        p.srcNdMatrixStride = 0;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;
        if (bTrans) {
            // B stored [N, K] row-major, GM row stride = K. Load block [nChunk, kChunk].
            p.nValue = static_cast<uint16_t>(nChunk);
            p.dValue = static_cast<uint16_t>(kChunk);
            p.srcDValue = static_cast<uint16_t>(K);
            p.dstNzC0Stride = static_cast<uint16_t>(CeilAlign(nChunk, SEGMM_BLOCK_LENGTH));
            AscendC::DataCopy(b1Local, bGm[static_cast<uint64_t>(nOff) * K + kOff], p);
        } else {
            // B stored [K, N] row-major, GM row stride = N. Load block [kChunk, nChunk].
            p.nValue = static_cast<uint16_t>(kChunk);
            p.dValue = static_cast<uint16_t>(nChunk);
            p.srcDValue = static_cast<uint16_t>(N);
            p.dstNzC0Stride = static_cast<uint16_t>(CeilAlign(kChunk, SEGMM_BLOCK_LENGTH));
            AscendC::DataCopy(b1Local, bGm[static_cast<uint64_t>(kOff) * N + nOff], p);
        }
        b1Queue.EnQue(b1Local);
    }

    // B transposed (B=[N,K] in B1 NZ): NZ already matches ZN orientation of B2 ->
    // plain LoadData2D (ifTranspose=false). Loop over K-fractals.
    __aicore__ inline void SplitBNoTranspose(uint32_t kChunk, uint32_t nChunk)
    {
        AscendC::LocalTensor<float> b1Local = b1Queue.DeQue<float>();
        AscendC::LocalTensor<float> b2Local = b2Queue.AllocTensor<float>();
        uint32_t nFractals = CeilDiv(nChunk, SEGMM_BLOCK_LENGTH);
        uint32_t kFractals = CeilDiv(kChunk, SEGMM_C0_FLOAT);
        AscendC::LoadData2DParams lp;
        lp.repeatTimes = static_cast<uint8_t>(nFractals);
        lp.srcStride = 1;
        lp.dstGap = 0;
        lp.ifTranspose = false;
        uint32_t stride = nFractals * SEGMM_FRACTAL_SIZE;
        for (uint32_t i = 0; i < kFractals; ++i) {
            AscendC::LoadData(b2Local[i * stride], b1Local[i * stride], lp);
        }
        b2Queue.EnQue<float>(b2Local);
        b1Queue.FreeTensor(b1Local);
    }

    // B non-transposed (B=[K,N] in B1 NZ): need NZ->ZN fractal conversion ->
    // LoadData3DParamsV2 with enTranspose=true (FP32 supported on 910B3).
    __aicore__ inline void SplitBTranspose(uint32_t kChunk, uint32_t nChunk)
    {
        AscendC::LocalTensor<float> b1Local = b1Queue.DeQue<float>();
        AscendC::LocalTensor<float> b2Local = b2Queue.AllocTensor<float>();
        AscendC::LoadData3DParamsV2<float> lp;
        lp.l1H = static_cast<uint16_t>(CeilAlign(kChunk, SEGMM_BLOCK_LENGTH));
        lp.l1W = 1;
        lp.channelSize = static_cast<uint16_t>(CeilAlign(nChunk, SEGMM_C0_FLOAT));
        lp.kExtension = static_cast<uint16_t>(CeilAlign(nChunk, SEGMM_C0_FLOAT));
        lp.mExtension = static_cast<uint16_t>(CeilAlign(kChunk, SEGMM_BLOCK_LENGTH));
        lp.strideW = 1; lp.strideH = 1;
        lp.filterW = 1; lp.filterH = 1;
        lp.dilationFilterW = 1; lp.dilationFilterH = 1;
        lp.filterSizeW = false; lp.filterSizeH = false;
        lp.enTranspose = true;
        lp.fMatrixCtrl = false;
        AscendC::LoadData(b2Local, b1Local, lp);
        b2Queue.EnQue<float>(b2Local);
        b1Queue.FreeTensor(b1Local);
    }

    __aicore__ inline void Compute(
        uint32_t actualRows, uint32_t nChunk, uint32_t kChunk,
        bool initC, AscendC::LocalTensor<float> &co1Local)
    {
        AscendC::LocalTensor<float> a2Local = a2Queue.DeQue<float>();
        AscendC::LocalTensor<float> b2Local = b2Queue.DeQue<float>();
        AscendC::MmadParams mp;
        mp.m = static_cast<uint16_t>(actualRows);
        mp.n = static_cast<uint16_t>(nChunk);
        mp.k = static_cast<uint16_t>(kChunk);
        mp.cmatrixInitVal = initC;
        mp.kDirectionAlign = false;   // A not transposed in forward
        AscendC::Mmad(co1Local, a2Local, b2Local, mp);
        AscendC::PipeBarrier<PIPE_M>();
        a2Queue.FreeTensor(a2Local);
        b2Queue.FreeTensor(b2Local);
    }

    __aicore__ inline void CopyOut(
        uint32_t mOff, uint32_t nOff, uint32_t actualRows, uint32_t nChunk,
        AscendC::LocalTensor<float> &co1Local)
    {
        AscendC::FixpipeParamsV220 fp;
        fp.nSize = static_cast<uint16_t>(nChunk);
        fp.mSize = static_cast<uint16_t>(actualRows);
        fp.srcStride = static_cast<uint16_t>(CeilAlign(actualRows, SEGMM_BLOCK_LENGTH));
        fp.dstStride = N;                  // output row stride (elements) = N
        fp.ndNum = 1;
        fp.srcNdStride = 0;
        fp.dstNdStride = 0;
        fp.quantPre = QuantMode_t::NoQuant;   // FP32 -> FP32
        AscendC::Fixpipe(cGm[static_cast<uint64_t>(mOff) * N + nOff], co1Local, fp);
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
    __aicore__ inline uint32_t CeilAlign(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }
    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }

    uint32_t M, K, N, mGrid, nGrid, nTile, kTile;
    uint32_t mIdx, nIdx;
    bool bTrans;

    AscendC::TQue<AscendC::TPosition::A1, SEGMM_BUFFER_NUM> a1Queue;
    AscendC::TQue<AscendC::TPosition::B1, SEGMM_BUFFER_NUM> b1Queue;
    AscendC::TQue<AscendC::TPosition::A2, SEGMM_BUFFER_NUM> a2Queue;
    AscendC::TQue<AscendC::TPosition::B2, SEGMM_BUFFER_NUM> b2Queue;
    AscendC::TQue<AscendC::TPosition::CO1, SEGMM_BUFFER_NUM> co1Queue;

    AscendC::GlobalTensor<float> aGm, bGm, cGm;
};

// ===========================================================================
// SegmentMMBackwardB: dB[r][n,m] = A_seg[k,n]^T @ dC_seg[k,m]
//   Left  = A_seg stored [K, N], transposed -> want [N, K] in L0A (ZZ).
//   Right = dC_seg stored [K, M], NOT transposed -> want [K, M] in L0B (ZN).
//   Mmad computes C[N, M] = A2[N,K] @ B2[K,M]; output dB[r] is [N, M] row-major.
// ===========================================================================
class SegmentMmBwdBCubeProcessor {
public:
    __aicore__ inline void Init(
        GM_ADDR aData, GM_ADDR dCData, GM_ADDR dBData,
        uint32_t k, uint32_t n, uint32_t m,
        uint32_t mGrid, uint32_t nGrid,
        uint32_t nTile, uint32_t kTile,
        AscendC::TPipe *pipe)
    {
        // GEMM view: outM=n, outN=m, innerK=k.
        this->K = k; this->N = n; this->M = m;
        this->mGrid = mGrid; this->nGrid = nGrid;
        this->nTile = nTile; this->kTile = kTile;

        aGm.SetGlobalBuffer((__gm__ float *)aData, static_cast<uint64_t>(k) * n);
        dCGm.SetGlobalBuffer((__gm__ float *)dCData, static_cast<uint64_t>(k) * m);
        dBGm.SetGlobalBuffer((__gm__ float *)dBData, static_cast<uint64_t>(n) * m);

        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->nIdx = blockIdx % nGrid;
        this->mIdx = blockIdx / nGrid;

        pipe->InitBuffer(a1Queue, SEGMM_BUFFER_NUM, SEGMM_L0A_BUFFER_BYTES);
        pipe->InitBuffer(b1Queue, SEGMM_BUFFER_NUM, SEGMM_L0B_BUFFER_BYTES);
        pipe->InitBuffer(a2Queue, SEGMM_BUFFER_NUM, SEGMM_L0A_BUFFER_BYTES);
        pipe->InitBuffer(b2Queue, SEGMM_BUFFER_NUM, SEGMM_L0B_BUFFER_BYTES);
        pipe->InitBuffer(co1Queue, SEGMM_BUFFER_NUM, SEGMM_L0C_BUFFER_BYTES);
    }

    __aicore__ inline void Process()
    {
        if (mIdx >= mGrid) return;

        // Output M direction = N (=n). Tile in 16-row blocks.
        uint32_t mBlocks = CeilDiv(N, SEGMM_BLOCK_LENGTH);
        uint32_t mBlocksPerCore = CeilDiv(mBlocks, mGrid);
        uint32_t myMBlockLo = mIdx * mBlocksPerCore;
        uint32_t myMBlockHi = Min(myMBlockLo + mBlocksPerCore, mBlocks);
        if (myMBlockLo >= myMBlockHi) return;

        // Output N direction = M (=m).
        uint32_t nPerCore = CeilDiv(M, nGrid);
        uint32_t myNLo = nIdx * nPerCore;
        uint32_t myNHi = Min(myNLo + nPerCore, M);
        if (myNLo >= myNHi) return;

        for (uint32_t mBlk = myMBlockLo; mBlk < myMBlockHi; ++mBlk) {
            uint32_t nOff = mBlk * SEGMM_BLOCK_LENGTH;          // output row index in [0,N)
            uint32_t actualRows = Min(SEGMM_BLOCK_LENGTH, N - nOff);

            for (uint32_t mOff = myNLo; mOff < myNHi; mOff += nTile) {
                uint32_t nChunk = Min(nTile, myNHi - mOff);

                AscendC::LocalTensor<float> co1Local = co1Queue.AllocTensor<float>();
                bool initC = true;

                for (uint32_t kOff = 0; kOff < K; kOff += kTile) {
                    uint32_t kChunk = Min(kTile, K - kOff);

                    // A_seg[kOff.., nOff..] block [kChunk, actualRows]; want transposed -> A2 [actualRows, kChunk].
                    LoadA1(kOff, nOff, kChunk, actualRows);
                    SplitATranspose(kChunk, actualRows);

                    // dC_seg[kOff.., mOff..] block [kChunk, nChunk]; B non-transposed -> B2 ZN [kChunk, nChunk].
                    LoadB1(kOff, mOff, kChunk, nChunk);
                    SplitBTranspose(kChunk, nChunk);

                    Compute(actualRows, nChunk, kChunk, initC, co1Local);
                    initC = false;
                }

                co1Queue.EnQue(co1Local);
                AscendC::LocalTensor<float> co1Out = co1Queue.DeQue<float>();
                CopyOut(nOff, mOff, actualRows, nChunk, co1Out);
                co1Queue.FreeTensor(co1Out);
            }
        }
    }

private:
    // A1: A_seg stored [K, N] row-major (GM row stride = N). Load block [kChunk, actualRows].
    __aicore__ inline void LoadA1(uint32_t kOff, uint32_t nOff, uint32_t kChunk, uint32_t actualRows)
    {
        AscendC::LocalTensor<float> a1Local = a1Queue.AllocTensor<float>();
        AscendC::Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = static_cast<uint16_t>(kChunk);
        p.dValue = static_cast<uint16_t>(actualRows);
        p.srcNdMatrixStride = 0;
        p.srcDValue = static_cast<uint16_t>(N);   // A_seg GM row stride = N
        p.dstNzC0Stride = static_cast<uint16_t>(CeilAlign(kChunk, SEGMM_BLOCK_LENGTH));
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1Local, aGm[static_cast<uint64_t>(kOff) * N + nOff], p);
        a1Queue.EnQue(a1Local);
    }

    // A1(NZ of [K, actualRows]) -> A2(ZZ of [actualRows, K]) with semantic transpose.
    // FP32: LoadData3DParamsV2 + enTranspose=true; Mmad needs kDirectionAlign=true.
    __aicore__ inline void SplitATranspose(uint32_t kChunk, uint32_t actualRows)
    {
        AscendC::LocalTensor<float> a1Local = a1Queue.DeQue<float>();
        AscendC::LocalTensor<float> a2Local = a2Queue.AllocTensor<float>();
        AscendC::LoadData3DParamsV2<float> lp;
        lp.l1H = static_cast<uint16_t>(CeilAlign(kChunk, SEGMM_BLOCK_LENGTH));   // K rows in L1
        lp.l1W = 1;
        lp.channelSize = static_cast<uint16_t>(CeilAlign(actualRows, SEGMM_C0_FLOAT));
        lp.kExtension = static_cast<uint16_t>(CeilAlign(actualRows, SEGMM_C0_FLOAT));
        lp.mExtension = static_cast<uint16_t>(CeilAlign(kChunk, SEGMM_BLOCK_LENGTH));
        lp.strideW = 1; lp.strideH = 1;
        lp.filterW = 1; lp.filterH = 1;
        lp.dilationFilterW = 1; lp.dilationFilterH = 1;
        lp.filterSizeW = false; lp.filterSizeH = false;
        lp.enTranspose = true;
        lp.fMatrixCtrl = false;
        AscendC::LoadData(a2Local, a1Local, lp);
        a2Queue.EnQue<float>(a2Local);
        a1Queue.FreeTensor(a1Local);
    }

    // dC1: dC_seg stored [K, M] row-major (GM row stride = M). Load block [kChunk, nChunk].
    __aicore__ inline void LoadB1(uint32_t kOff, uint32_t mOff, uint32_t kChunk, uint32_t nChunk)
    {
        AscendC::LocalTensor<float> b1Local = b1Queue.AllocTensor<float>();
        AscendC::Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = static_cast<uint16_t>(kChunk);
        p.dValue = static_cast<uint16_t>(nChunk);
        p.srcNdMatrixStride = 0;
        p.srcDValue = static_cast<uint16_t>(M);   // dC GM row stride = M
        p.dstNzC0Stride = static_cast<uint16_t>(CeilAlign(kChunk, SEGMM_BLOCK_LENGTH));
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;
        AscendC::DataCopy(b1Local, dCGm[static_cast<uint64_t>(kOff) * M + mOff], p);
        b1Queue.EnQue(b1Local);
    }

    // dC non-transposed: NZ->ZN fractal conversion via LoadData3D enTranspose.
    __aicore__ inline void SplitBTranspose(uint32_t kChunk, uint32_t nChunk)
    {
        AscendC::LocalTensor<float> b1Local = b1Queue.DeQue<float>();
        AscendC::LocalTensor<float> b2Local = b2Queue.AllocTensor<float>();
        AscendC::LoadData3DParamsV2<float> lp;
        lp.l1H = static_cast<uint16_t>(CeilAlign(kChunk, SEGMM_BLOCK_LENGTH));
        lp.l1W = 1;
        lp.channelSize = static_cast<uint16_t>(CeilAlign(nChunk, SEGMM_C0_FLOAT));
        lp.kExtension = static_cast<uint16_t>(CeilAlign(nChunk, SEGMM_C0_FLOAT));
        lp.mExtension = static_cast<uint16_t>(CeilAlign(kChunk, SEGMM_BLOCK_LENGTH));
        lp.strideW = 1; lp.strideH = 1;
        lp.filterW = 1; lp.filterH = 1;
        lp.dilationFilterW = 1; lp.dilationFilterH = 1;
        lp.filterSizeW = false; lp.filterSizeH = false;
        lp.enTranspose = true;
        lp.fMatrixCtrl = false;
        AscendC::LoadData(b2Local, b1Local, lp);
        b2Queue.EnQue<float>(b2Local);
        b1Queue.FreeTensor(b1Local);
    }

    __aicore__ inline void Compute(
        uint32_t actualRows, uint32_t nChunk, uint32_t kChunk,
        bool initC, AscendC::LocalTensor<float> &co1Local)
    {
        AscendC::LocalTensor<float> a2Local = a2Queue.DeQue<float>();
        AscendC::LocalTensor<float> b2Local = b2Queue.DeQue<float>();
        AscendC::MmadParams mp;
        mp.m = static_cast<uint16_t>(actualRows);
        mp.n = static_cast<uint16_t>(nChunk);
        mp.k = static_cast<uint16_t>(kChunk);
        mp.cmatrixInitVal = initC;
        mp.kDirectionAlign = true;    // A is transposed (Mmad doc requirement for float)
        AscendC::Mmad(co1Local, a2Local, b2Local, mp);
        AscendC::PipeBarrier<PIPE_M>();
        a2Queue.FreeTensor(a2Local);
        b2Queue.FreeTensor(b2Local);
    }

    __aicore__ inline void CopyOut(
        uint32_t nOff, uint32_t mOff, uint32_t actualRows, uint32_t nChunk,
        AscendC::LocalTensor<float> &co1Local)
    {
        AscendC::FixpipeParamsV220 fp;
        fp.nSize = static_cast<uint16_t>(nChunk);
        fp.mSize = static_cast<uint16_t>(actualRows);
        fp.srcStride = static_cast<uint16_t>(CeilAlign(actualRows, SEGMM_BLOCK_LENGTH));
        fp.dstStride = M;                  // dB[r] row stride = M
        fp.ndNum = 1;
        fp.srcNdStride = 0;
        fp.dstNdStride = 0;
        fp.quantPre = QuantMode_t::NoQuant;
        AscendC::Fixpipe(dBGm[static_cast<uint64_t>(nOff) * M + mOff], co1Local, fp);
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
    __aicore__ inline uint32_t CeilAlign(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }
    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }

    uint32_t K, N, M, mGrid, nGrid, nTile, kTile;
    uint32_t mIdx, nIdx;

    AscendC::TQue<AscendC::TPosition::A1, SEGMM_BUFFER_NUM> a1Queue;
    AscendC::TQue<AscendC::TPosition::B1, SEGMM_BUFFER_NUM> b1Queue;
    AscendC::TQue<AscendC::TPosition::A2, SEGMM_BUFFER_NUM> a2Queue;
    AscendC::TQue<AscendC::TPosition::B2, SEGMM_BUFFER_NUM> b2Queue;
    AscendC::TQue<AscendC::TPosition::CO1, SEGMM_BUFFER_NUM> co1Queue;

    AscendC::GlobalTensor<float> aGm, dCGm, dBGm;
};

#endif // SEGMENT_MM_CUBE_PROCESSOR_H
