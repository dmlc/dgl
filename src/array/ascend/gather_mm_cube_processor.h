#ifndef GATHER_MM_CUBE_PROCESSOR_H
#define GATHER_MM_CUBE_PROCESSOR_H
#include "kernel_operator.h"

// ===== Cube (AIC) processors for GatherMM / GatherMMScatter =====
//
// Reuses SegmentMM Cube GEMM patterns (Nd2Nz / LoadData2D / LoadData3D / Mmad /
// Fixpipe). FP32 transpose uses LoadData3DParamsV2 enTranspose=true (ifTranspose
// does not support float — see segment_mm_cube_processor.h).
//
// Per-row index gather uses DataCopy per-row into NZ L1 buffer (bspmm pattern,
// adapted for float C0=8). Mmad ignores data beyond m/k/n params (Mmad doc:
// "无效数据在执行矩阵乘法时会被忽略"), so CeilAlign padding in the NZ buffer
// is safe.

constexpr uint32_t GMM_BLOCK_LENGTH = 16;
constexpr uint32_t GMM_C0_FLOAT = 8;
constexpr uint32_t GMM_FRACTAL_SIZE = 16 * 8;
constexpr uint32_t GMM_BUFFER_NUM = 2;
constexpr uint32_t GMM_L0A_BUFFER_BYTES = 32 * 1024;
constexpr uint32_t GMM_L0B_BUFFER_BYTES = 32 * 1024;
constexpr uint32_t GMM_L0C_BUFFER_BYTES = 64 * 1024;
constexpr uint32_t GMM_MAX_KTILE = 64;
constexpr uint32_t GMM_MAX_NTILE = 64;

// ===========================================================================
// K1 processor: GatherMM / GatherMMScatter (B 3D)
//   Per group g (grouped by idx_b):
//     C_ws[group_offset + mOff, nOff] = A_gathered[mOff, k] @ B[idxBGroup[g], k, n]
//   A rows gathered per-row by idx_a_perm; B slice loaded via Nd2Nz (contiguous).
// ===========================================================================
class GatherMmK1CubeProcessor {
public:
    __aicore__ inline void Init(
        GM_ADDR aData, GM_ADDR bData, GM_ADDR cWsData,
        GM_ADDR seglenData, GM_ADDR groupOffsetsData,
        GM_ADDR idxAPermData, GM_ADDR idxBGroupData,
        uint32_t m, uint32_t k, uint32_t n,
        uint32_t numGroups, uint32_t nTile, uint32_t kTile,
        AscendC::TPipe *pipe)
    {
        this->M = m; this->K = k; this->N = n;
        this->numGroups = numGroups;
        this->nTile = nTile; this->kTile = kTile;

        aGm.SetGlobalBuffer((__gm__ float *)aData, static_cast<uint64_t>(m) * k);
        bGm.SetGlobalBuffer((__gm__ float *)bData);
        cWsGm.SetGlobalBuffer((__gm__ float *)cWsData, static_cast<uint64_t>(m) * n);
        seglenGm.SetGlobalBuffer((__gm__ uint32_t *)seglenData, numGroups);
        groupOffsetsGm.SetGlobalBuffer((__gm__ uint32_t *)groupOffsetsData, numGroups);
        idxAPermGm.SetGlobalBuffer((__gm__ uint32_t *)idxAPermData, m);
        idxBGroupGm.SetGlobalBuffer((__gm__ uint32_t *)idxBGroupData, numGroups);

        pipe->InitBuffer(a1Queue, GMM_BUFFER_NUM, GMM_L0A_BUFFER_BYTES);
        pipe->InitBuffer(b1Queue, GMM_BUFFER_NUM, GMM_L0B_BUFFER_BYTES);
        pipe->InitBuffer(a2Queue, GMM_BUFFER_NUM, GMM_L0A_BUFFER_BYTES);
        pipe->InitBuffer(b2Queue, GMM_BUFFER_NUM, GMM_L0B_BUFFER_BYTES);
        pipe->InitBuffer(co1Queue, GMM_BUFFER_NUM, GMM_L0C_BUFFER_BYTES);
    }

    __aicore__ inline void Process()
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        if (numGroups == 0) return;

        uint32_t groupsPerCore = CeilDiv(numGroups, blockNum);
        uint32_t gLo = blockIdx * groupsPerCore;
        uint32_t gHi = Min(gLo + groupsPerCore, numGroups);
        if (gLo >= gHi) return;

        for (uint32_t g = gLo; g < gHi; ++g) {
            uint32_t Mg = seglenGm.GetValue(g);
            uint32_t groupOffset = groupOffsetsGm.GetValue(g);
            uint32_t bIdx = idxBGroupGm.GetValue(g);
            if (Mg == 0 || K == 0 || N == 0) continue;

            ProcessGroup(Mg, groupOffset, bIdx);
        }
    }

private:
    __aicore__ inline void ProcessGroup(uint32_t Mg, uint32_t groupOffset, uint32_t bIdx)
    {
        uint32_t mBlocks = CeilDiv(Mg, GMM_BLOCK_LENGTH);
        for (uint32_t mBlk = 0; mBlk < mBlocks; ++mBlk) {
            uint32_t mOff = mBlk * GMM_BLOCK_LENGTH;
            uint32_t actualRows = Min(GMM_BLOCK_LENGTH, Mg - mOff);

            for (uint32_t nOff = 0; nOff < N; nOff += nTile) {
                uint32_t nChunk = Min(nTile, N - nOff);

                AscendC::LocalTensor<float> co1Local = co1Queue.AllocTensor<float>();
                bool initC = true;

                for (uint32_t kOff = 0; kOff < K; kOff += kTile) {
                    uint32_t kChunk = Min(kTile, K - kOff);

                    LoadA1(groupOffset + mOff, kOff, actualRows, kChunk);
                    SplitA(kChunk);

                    LoadB1(bIdx, kOff, nOff, kChunk, nChunk);
                    SplitBTranspose(kChunk, nChunk);

                    Compute(actualRows, nChunk, kChunk, initC, co1Local);
                    initC = false;
                }

                co1Queue.EnQue(co1Local);
                AscendC::LocalTensor<float> co1Out = co1Queue.DeQue<float>();
                CopyOut(groupOffset + mOff, nOff, actualRows, nChunk, co1Out);
                co1Queue.FreeTensor(co1Out);
            }
        }
    }

    // Per-row gather A[idxAPerm[row]] -> A1 NZ buffer.
    // NZ fractal for float: [16, C0=8] row-major. Row r's 8 elements are contiguous (32B).
    __aicore__ inline void LoadA1(uint32_t cwsRowStart, uint32_t kOff,
                                   uint32_t actualRows, uint32_t kChunk)
    {
        AscendC::LocalTensor<float> a1Local = a1Queue.AllocTensor<float>();
        uint32_t mBlocks = CeilDiv(actualRows, GMM_BLOCK_LENGTH);
        uint32_t kBlocks = CeilDiv(kChunk, GMM_C0_FLOAT);
        // DataCopy: blockCount blocks of 32B (8 floats) each, dstGap jumps to next K-block fractal.
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(kBlocks);
        copyParams.blockLen = 1;          // 1 x 32B = 8 floats per block
        copyParams.srcGap = 0;            // source contiguous
        copyParams.dstGap = static_cast<uint16_t>(mBlocks * GMM_BLOCK_LENGTH - 1);

        for (uint32_t i = 0; i < actualRows; ++i) {
            uint32_t mBlock = i / GMM_BLOCK_LENGTH;
            uint32_t r = i % GMM_BLOCK_LENGTH;
            uint32_t srcRowIdx = idxAPermGm.GetValue(cwsRowStart + i);
            uint64_t srcOffset = static_cast<uint64_t>(srcRowIdx) * K + kOff;
            uint32_t dstOffset = mBlock * GMM_FRACTAL_SIZE + r * GMM_C0_FLOAT;
            AscendC::DataCopy(a1Local[dstOffset], aGm[srcOffset], copyParams);
        }
        a1Queue.EnQue<float>(a1Local);
    }

    // A1(NZ) -> A2(ZZ), A not transposed: plain LoadData2D (ifTranspose=false).
    __aicore__ inline void SplitA(uint32_t kChunk)
    {
        AscendC::LocalTensor<float> a1Local = a1Queue.DeQue<float>();
        AscendC::LocalTensor<float> a2Local = a2Queue.AllocTensor<float>();
        AscendC::LoadData2DParams lp;
        lp.repeatTimes = static_cast<uint8_t>(CeilDiv(kChunk, GMM_C0_FLOAT));
        lp.srcStride = 1;
        lp.dstGap = 0;
        lp.ifTranspose = false;
        AscendC::LoadData(a2Local, a1Local, lp);
        a2Queue.EnQue<float>(a2Local);
        a1Queue.FreeTensor(a1Local);
    }

    // B1: Nd2Nz B[bIdx, kOff:kOff+kChunk, nOff:nOff+nChunk] -> B1 NZ.
    // B is 3D [R, K, N], B[bIdx] is [K, N] at offset bIdx*K*N.
    __aicore__ inline void LoadB1(uint32_t bIdx, uint32_t kOff, uint32_t nOff,
                                   uint32_t kChunk, uint32_t nChunk)
    {
        AscendC::LocalTensor<float> b1Local = b1Queue.AllocTensor<float>();
        AscendC::Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = static_cast<uint16_t>(kChunk);
        p.dValue = static_cast<uint16_t>(nChunk);
        p.srcNdMatrixStride = 0;
        p.srcDValue = static_cast<uint16_t>(N);   // B[bIdx] row stride = N
        p.dstNzC0Stride = static_cast<uint16_t>(CeilAlign(kChunk, GMM_BLOCK_LENGTH));
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;
        uint64_t bOffset = static_cast<uint64_t>(bIdx) * K * N + static_cast<uint64_t>(kOff) * N + nOff;
        AscendC::DataCopy(b1Local, bGm[bOffset], p);
        b1Queue.EnQue<float>(b1Local);
    }

    // B non-transposed (B=[K,N] in B1 NZ): NZ->ZN fractal conversion via LoadData3D enTranspose.
    __aicore__ inline void SplitBTranspose(uint32_t kChunk, uint32_t nChunk)
    {
        AscendC::LocalTensor<float> b1Local = b1Queue.DeQue<float>();
        AscendC::LocalTensor<float> b2Local = b2Queue.AllocTensor<float>();
        AscendC::LoadData3DParamsV2<float> lp;
        lp.l1H = static_cast<uint16_t>(CeilAlign(kChunk, GMM_BLOCK_LENGTH));
        lp.l1W = 1;
        lp.channelSize = static_cast<uint16_t>(CeilAlign(nChunk, GMM_C0_FLOAT));
        lp.kExtension = static_cast<uint16_t>(CeilAlign(nChunk, GMM_C0_FLOAT));
        lp.mExtension = static_cast<uint16_t>(CeilAlign(kChunk, GMM_BLOCK_LENGTH));
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
        mp.kDirectionAlign = false;   // A not transposed
        AscendC::Mmad(co1Local, a2Local, b2Local, mp);
        AscendC::PipeBarrier<PIPE_M>();
        a2Queue.FreeTensor(a2Local);
        b2Queue.FreeTensor(b2Local);
    }

    __aicore__ inline void CopyOut(
        uint32_t cwsRowOff, uint32_t nOff, uint32_t actualRows, uint32_t nChunk,
        AscendC::LocalTensor<float> &co1Local)
    {
        AscendC::FixpipeParamsV220 fp;
        fp.nSize = static_cast<uint16_t>(nChunk);
        fp.mSize = static_cast<uint16_t>(actualRows);
        fp.srcStride = static_cast<uint16_t>(CeilAlign(actualRows, GMM_BLOCK_LENGTH));
        fp.dstStride = N;
        fp.ndNum = 1;
        fp.srcNdStride = 0;
        fp.dstNdStride = 0;
        fp.quantPre = QuantMode_t::NoQuant;
        AscendC::Fixpipe(cWsGm[static_cast<uint64_t>(cwsRowOff) * N + nOff], co1Local, fp);
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
    __aicore__ inline uint32_t CeilAlign(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }
    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }

    uint32_t M, K, N, numGroups, nTile, kTile;

    AscendC::TQue<AscendC::TPosition::A1, GMM_BUFFER_NUM> a1Queue;
    AscendC::TQue<AscendC::TPosition::B1, GMM_BUFFER_NUM> b1Queue;
    AscendC::TQue<AscendC::TPosition::A2, GMM_BUFFER_NUM> a2Queue;
    AscendC::TQue<AscendC::TPosition::B2, GMM_BUFFER_NUM> b2Queue;
    AscendC::TQue<AscendC::TPosition::CO1, GMM_BUFFER_NUM> co1Queue;

    AscendC::GlobalTensor<float> aGm, bGm, cWsGm;
    AscendC::GlobalTensor<uint32_t> seglenGm, groupOffsetsGm, idxAPermGm, idxBGroupGm;
};

// ===========================================================================
// K2 processor: GatherMMScatter (B 2D, outer product scatter)
//   Per group g (grouped by idx_c):
//     C[idxCGroup[g], k, n] = A_gathered^T[k, Mg] @ B_gathered[Mg, n]
//   GEMM: outM=k, outN=n, innerK=Mg.
//   A transposed (SplitATranspose), B non-transposed (SplitBTranspose).
//   Both A and B rows gathered per-row. Fixpipe direct write to C (unique writer).
// ===========================================================================
class GatherMmK2CubeProcessor {
public:
    __aicore__ inline void Init(
        GM_ADDR aData, GM_ADDR bData, GM_ADDR cData,
        GM_ADDR seglenData, GM_ADDR groupOffsetsData,
        GM_ADDR idxAPermData, GM_ADDR idxBPermData, GM_ADDR idxCGroupData,
        uint32_t m, uint32_t k, uint32_t n,
        uint32_t numGroups, uint32_t nTile,
        AscendC::TPipe *pipe)
    {
        // GEMM view: outM=k, outN=n, innerK=Mg (per group).
        this->M = m; this->K = k; this->N = n;
        this->numGroups = numGroups;
        this->nTile = nTile;

        aGm.SetGlobalBuffer((__gm__ float *)aData, static_cast<uint64_t>(m) * k);
        bGm.SetGlobalBuffer((__gm__ float *)bData);
        cGm.SetGlobalBuffer((__gm__ float *)cData);
        seglenGm.SetGlobalBuffer((__gm__ uint32_t *)seglenData, numGroups);
        groupOffsetsGm.SetGlobalBuffer((__gm__ uint32_t *)groupOffsetsData, numGroups);
        idxAPermGm.SetGlobalBuffer((__gm__ uint32_t *)idxAPermData, m);
        idxBPermGm.SetGlobalBuffer((__gm__ uint32_t *)idxBPermData, m);
        idxCGroupGm.SetGlobalBuffer((__gm__ uint32_t *)idxCGroupData, numGroups);

        pipe->InitBuffer(a1Queue, GMM_BUFFER_NUM, GMM_L0A_BUFFER_BYTES);
        pipe->InitBuffer(b1Queue, GMM_BUFFER_NUM, GMM_L0B_BUFFER_BYTES);
        pipe->InitBuffer(a2Queue, GMM_BUFFER_NUM, GMM_L0A_BUFFER_BYTES);
        pipe->InitBuffer(b2Queue, GMM_BUFFER_NUM, GMM_L0B_BUFFER_BYTES);
        pipe->InitBuffer(co1Queue, GMM_BUFFER_NUM, GMM_L0C_BUFFER_BYTES);
    }

    __aicore__ inline void Process()
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        if (numGroups == 0) return;

        uint32_t groupsPerCore = CeilDiv(numGroups, blockNum);
        uint32_t gLo = blockIdx * groupsPerCore;
        uint32_t gHi = Min(gLo + groupsPerCore, numGroups);
        if (gLo >= gHi) return;

        for (uint32_t g = gLo; g < gHi; ++g) {
            uint32_t Mg = seglenGm.GetValue(g);
            uint32_t groupOffset = groupOffsetsGm.GetValue(g);
            uint32_t cIdx = idxCGroupGm.GetValue(g);
            if (Mg == 0 || K == 0 || N == 0) continue;

            // innerK = Mg; compute kTile for this group.
            uint32_t innerKtile = Min(GMM_MAX_KTILE, Mg);
            innerKtile = (innerKtile / GMM_C0_FLOAT) * GMM_C0_FLOAT;
            if (innerKtile == 0) innerKtile = GMM_C0_FLOAT;

            ProcessGroup(Mg, groupOffset, cIdx, innerKtile);
        }
    }

private:
    __aicore__ inline void ProcessGroup(uint32_t Mg, uint32_t groupOffset, uint32_t cIdx,
                                         uint32_t innerKtile)
    {
        // GEMM: outM=K, outN=N, innerK=Mg.
        // mBlock loop over outM (K direction, 16-row blocks).
        uint32_t mBlocks = CeilDiv(K, GMM_BLOCK_LENGTH);
        for (uint32_t mBlk = 0; mBlk < mBlocks; ++mBlk) {
            uint32_t mOffK = mBlk * GMM_BLOCK_LENGTH;
            uint32_t actualRowsK = Min(GMM_BLOCK_LENGTH, K - mOffK);

            for (uint32_t nOff = 0; nOff < N; nOff += nTile) {
                uint32_t nChunk = Min(nTile, N - nOff);

                AscendC::LocalTensor<float> co1Local = co1Queue.AllocTensor<float>();
                bool initC = true;

                // innerK loop (Mg direction).
                for (uint32_t kOff = 0; kOff < Mg; kOff += innerKtile) {
                    uint32_t kChunk = Min(innerKtile, Mg - kOff);

                    // A_gathered[kOff:kOff+kChunk, mOffK:mOffK+actualRowsK] -> A1 NZ [kChunk, actualRowsK]
                    LoadA1(groupOffset + kOff, mOffK, kChunk, actualRowsK);
                    SplitATranspose(kChunk, actualRowsK);

                    // B_gathered[kOff:kOff+kChunk, nOff:nOff+nChunk] -> B1 NZ [kChunk, nChunk]
                    LoadB1(groupOffset + kOff, nOff, kChunk, nChunk);
                    SplitBTranspose(kChunk, nChunk);

                    Compute(actualRowsK, nChunk, kChunk, initC, co1Local);
                    initC = false;
                }

                co1Queue.EnQue(co1Local);
                AscendC::LocalTensor<float> co1Out = co1Queue.DeQue<float>();
                CopyOut(cIdx, mOffK, nOff, actualRowsK, nChunk, co1Out);
                co1Queue.FreeTensor(co1Out);
            }
        }
    }

    // Per-row gather A[idxAPerm[row]] columns [mOffK, mOffK+actualRowsK) -> A1 NZ [kChunk, actualRowsK]
    // A is [M, K] row-major; each row has K elements. We load actualRowsK elements per row.
    __aicore__ inline void LoadA1(uint32_t cwsRowStart, uint32_t mOffK,
                                   uint32_t kChunk, uint32_t actualRowsK)
    {
        AscendC::LocalTensor<float> a1Local = a1Queue.AllocTensor<float>();
        uint32_t nBlocks = CeilDiv(kChunk, GMM_BLOCK_LENGTH);
        uint32_t dBlocks = CeilDiv(actualRowsK, GMM_C0_FLOAT);
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(dBlocks);
        copyParams.blockLen = 1;
        copyParams.srcGap = 0;
        copyParams.dstGap = static_cast<uint16_t>(nBlocks * GMM_BLOCK_LENGTH - 1);

        for (uint32_t i = 0; i < kChunk; ++i) {
            uint32_t nBlock = i / GMM_BLOCK_LENGTH;
            uint32_t r = i % GMM_BLOCK_LENGTH;
            uint32_t srcRowIdx = idxAPermGm.GetValue(cwsRowStart + i);
            uint64_t srcOffset = static_cast<uint64_t>(srcRowIdx) * K + mOffK;
            uint32_t dstOffset = nBlock * GMM_FRACTAL_SIZE + r * GMM_C0_FLOAT;
            AscendC::DataCopy(a1Local[dstOffset], aGm[srcOffset], copyParams);
        }
        a1Queue.EnQue<float>(a1Local);
    }

    // A1(NZ of [kChunk, actualRowsK]) -> A2(ZZ of [actualRowsK, kChunk]) with semantic transpose.
    // FP32: LoadData3DParamsV2 enTranspose=true; Mmad needs kDirectionAlign=true.
    __aicore__ inline void SplitATranspose(uint32_t kChunk, uint32_t actualRowsK)
    {
        AscendC::LocalTensor<float> a1Local = a1Queue.DeQue<float>();
        AscendC::LocalTensor<float> a2Local = a2Queue.AllocTensor<float>();
        AscendC::LoadData3DParamsV2<float> lp;
        lp.l1H = static_cast<uint16_t>(CeilAlign(kChunk, GMM_BLOCK_LENGTH));
        lp.l1W = 1;
        lp.channelSize = static_cast<uint16_t>(CeilAlign(actualRowsK, GMM_C0_FLOAT));
        lp.kExtension = static_cast<uint16_t>(CeilAlign(actualRowsK, GMM_C0_FLOAT));
        lp.mExtension = static_cast<uint16_t>(CeilAlign(kChunk, GMM_BLOCK_LENGTH));
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

    // Per-row gather B[idxBPerm[row]] columns [nOff, nOff+nChunk) -> B1 NZ [kChunk, nChunk]
    // B is 2D [R, N] row-major; each row has N elements.
    __aicore__ inline void LoadB1(uint32_t cwsRowStart, uint32_t nOff,
                                   uint32_t kChunk, uint32_t nChunk)
    {
        AscendC::LocalTensor<float> b1Local = b1Queue.AllocTensor<float>();
        uint32_t nBlocks = CeilDiv(kChunk, GMM_BLOCK_LENGTH);
        uint32_t dBlocks = CeilDiv(nChunk, GMM_C0_FLOAT);
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(dBlocks);
        copyParams.blockLen = 1;
        copyParams.srcGap = 0;
        copyParams.dstGap = static_cast<uint16_t>(nBlocks * GMM_BLOCK_LENGTH - 1);

        for (uint32_t i = 0; i < kChunk; ++i) {
            uint32_t nBlock = i / GMM_BLOCK_LENGTH;
            uint32_t r = i % GMM_BLOCK_LENGTH;
            uint32_t srcRowIdx = idxBPermGm.GetValue(cwsRowStart + i);
            uint64_t srcOffset = static_cast<uint64_t>(srcRowIdx) * N + nOff;
            uint32_t dstOffset = nBlock * GMM_FRACTAL_SIZE + r * GMM_C0_FLOAT;
            AscendC::DataCopy(b1Local[dstOffset], bGm[srcOffset], copyParams);
        }
        b1Queue.EnQue<float>(b1Local);
    }

    // B non-transposed: NZ->ZN fractal conversion via LoadData3D enTranspose.
    __aicore__ inline void SplitBTranspose(uint32_t kChunk, uint32_t nChunk)
    {
        AscendC::LocalTensor<float> b1Local = b1Queue.DeQue<float>();
        AscendC::LocalTensor<float> b2Local = b2Queue.AllocTensor<float>();
        AscendC::LoadData3DParamsV2<float> lp;
        lp.l1H = static_cast<uint16_t>(CeilAlign(kChunk, GMM_BLOCK_LENGTH));
        lp.l1W = 1;
        lp.channelSize = static_cast<uint16_t>(CeilAlign(nChunk, GMM_C0_FLOAT));
        lp.kExtension = static_cast<uint16_t>(CeilAlign(nChunk, GMM_C0_FLOAT));
        lp.mExtension = static_cast<uint16_t>(CeilAlign(kChunk, GMM_BLOCK_LENGTH));
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
        uint32_t actualRowsK, uint32_t nChunk, uint32_t kChunk,
        bool initC, AscendC::LocalTensor<float> &co1Local)
    {
        AscendC::LocalTensor<float> a2Local = a2Queue.DeQue<float>();
        AscendC::LocalTensor<float> b2Local = b2Queue.DeQue<float>();
        AscendC::MmadParams mp;
        mp.m = static_cast<uint16_t>(actualRowsK);
        mp.n = static_cast<uint16_t>(nChunk);
        mp.k = static_cast<uint16_t>(kChunk);
        mp.cmatrixInitVal = initC;
        mp.kDirectionAlign = true;    // A is transposed
        AscendC::Mmad(co1Local, a2Local, b2Local, mp);
        AscendC::PipeBarrier<PIPE_M>();
        a2Queue.FreeTensor(a2Local);
        b2Queue.FreeTensor(b2Local);
    }

    // Fixpipe direct write to C[cIdx, mOffK, nOff]. C is 3D [numC, K, N].
    __aicore__ inline void CopyOut(
        uint32_t cIdx, uint32_t mOffK, uint32_t nOff,
        uint32_t actualRowsK, uint32_t nChunk,
        AscendC::LocalTensor<float> &co1Local)
    {
        AscendC::FixpipeParamsV220 fp;
        fp.nSize = static_cast<uint16_t>(nChunk);
        fp.mSize = static_cast<uint16_t>(actualRowsK);
        fp.srcStride = static_cast<uint16_t>(CeilAlign(actualRowsK, GMM_BLOCK_LENGTH));
        fp.dstStride = N;   // C[cIdx] row stride = N
        fp.ndNum = 1;
        fp.srcNdStride = 0;
        fp.dstNdStride = 0;
        fp.quantPre = QuantMode_t::NoQuant;
        uint64_t dstOffset = static_cast<uint64_t>(cIdx) * K * N + static_cast<uint64_t>(mOffK) * N + nOff;
        AscendC::Fixpipe(cGm[dstOffset], co1Local, fp);
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
    __aicore__ inline uint32_t CeilAlign(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }
    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }

    uint32_t M, K, N, numGroups, nTile;

    AscendC::TQue<AscendC::TPosition::A1, GMM_BUFFER_NUM> a1Queue;
    AscendC::TQue<AscendC::TPosition::B1, GMM_BUFFER_NUM> b1Queue;
    AscendC::TQue<AscendC::TPosition::A2, GMM_BUFFER_NUM> a2Queue;
    AscendC::TQue<AscendC::TPosition::B2, GMM_BUFFER_NUM> b2Queue;
    AscendC::TQue<AscendC::TPosition::CO1, GMM_BUFFER_NUM> co1Queue;

    AscendC::GlobalTensor<float> aGm, bGm, cGm;
    AscendC::GlobalTensor<uint32_t> seglenGm, groupOffsetsGm, idxAPermGm, idxBPermGm, idxCGroupGm;
};

#endif // GATHER_MM_CUBE_PROCESSOR_H
