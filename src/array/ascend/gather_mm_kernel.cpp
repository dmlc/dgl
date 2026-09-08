#include "gather_mm_cube_processor.h"

// GatherMM / GatherMMScatter device kernels (Cube(AIC) + MemBase-vector(AIV) mix).

// ===== Kernel1: GatherMM / GatherMMScatter (B 3D) =====
// AIC: per-group Cube GEMM (A gather @ B slice) -> C_ws (group-sorted, contiguous).
// AIV: returns early (scatter is a separate kernel launch).
extern "C" __global__ __aicore__ void gather_mm_k1(
    GM_ADDR aData,
    GM_ADDR bData,
    GM_ADDR cWsData,
    GM_ADDR seglenData,
    GM_ADDR groupOffsetsData,
    GM_ADDR idxAPermData,
    GM_ADDR idxBGroupData,
    uint32_t m,
    uint32_t k,
    uint32_t n,
    uint32_t numGroups,
    uint32_t nTile,
    uint32_t kTile)
{
    AscendC::TPipe pipe;
    if ASCEND_IS_AIV {
        return;
    }

    if ASCEND_IS_AIC {
        GatherMmK1CubeProcessor proc;
        proc.Init(aData, bData, cWsData, seglenData, groupOffsetsData,
                  idxAPermData, idxBGroupData,
                  m, k, n, numGroups, nTile, kTile, &pipe);
        proc.Process();
    }
}

// ===== AIV scatter kernel: C_ws -> C (gather_mm_k1 post-pass) =====
// Per row i: C[scatterIdx[i]] += C_ws[i]  (atomic if isScatter, else direct write).
// Launched after gather_mm_k1 (host synchronizes between).
//
// Optimization for isScatter=1: UB-internal aggregation by idx_c.
// 1. Sort scatter indices within the chunk, building a row permutation.
// 2. Traverse sorted rows, accumulating C_ws rows with the same target in aggBuf.
// 3. Write aggBuf to C[target] once per unique target.
// This reduces atomic writes from myRows to unique(idx_c_in_chunk).
extern "C" __global__ __aicore__ void gather_mm_scatter_aiv(
    GM_ADDR cWsData,
    GM_ADDR cData,
    GM_ADDR scatterIdxData,
    uint32_t m,
    uint32_t n,
    uint32_t isScatter)
{
    AscendC::TPipe pipe;
    if ASCEND_IS_AIC {
        return;
    }

    if ASCEND_IS_AIV {
        AscendC::GlobalTensor<float> cWsGm;
        AscendC::GlobalTensor<float> cGm;
        AscendC::GlobalTensor<uint32_t> scatterIdxGm;
        cWsGm.SetGlobalBuffer((__gm__ float *)cWsData, static_cast<uint64_t>(m) * n);
        cGm.SetGlobalBuffer((__gm__ float *)cData);
        scatterIdxGm.SetGlobalBuffer((__gm__ uint32_t *)scatterIdxData, m);

        // Common constants.
        constexpr uint32_t UB_AVAILABLE = 192 * 1024;
        uint32_t ubElements = n;
        if (ubElements < 8) ubElements = 8;
        ubElements = (ubElements + 7) / 8 * 8;  // Round up to 8 for 32B alignment.

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t rowsPerCore = (m + blockNum - 1) / blockNum;
        uint32_t rLo = blockIdx * rowsPerCore;
        uint32_t rHi = rLo + rowsPerCore;
        if (rHi > m) rHi = m;
        if (rLo >= rHi) return;
        uint32_t myRows = rHi - rLo;

        // Index buffer: load scatter indices into UB via DataCopyPad (coherent DMA).
        AscendC::TBuf<AscendC::TPosition::VECCALC> idxBuf;
        uint32_t idxElems = myRows;
        if (idxElems < 8) idxElems = 8;
        idxElems = (idxElems + 7) / 8 * 8;
        pipe.InitBuffer(idxBuf, idxElems * sizeof(uint32_t));
        AscendC::LocalTensor<uint32_t> idxLocal = idxBuf.Get<uint32_t>();
        AscendC::DataCopyExtParams idxLoadParams;
        idxLoadParams.blockCount = 1;
        idxLoadParams.blockLen = static_cast<uint32_t>(myRows * sizeof(uint32_t));
        idxLoadParams.srcStride = 0;
        idxLoadParams.dstStride = 0;
        idxLoadParams.rsv = 0;
        AscendC::DataCopyPadExtParams<uint32_t> idxPadParams{false, 0, 0, 0};
        AscendC::DataCopyPad(idxLocal, scatterIdxGm[rLo], idxLoadParams, idxPadParams);
        AscendC::PipeBarrier<PIPE_MTE2>();

        // ===== Path selection =====
        // isScatter=0: direct write (no collision), keep original row-by-row code.
        // isScatter=1: UB aggregation optimization.
        if (!isScatter) {
            // ---- Original direct-write path (GatherMM, no collision) ----
            uint32_t ubAlloc = 4 * ubElements * sizeof(float) + 256;
            if (ubAlloc > UB_AVAILABLE) {
                KERNEL_LOG(KERNEL_ERROR, "gather_mm_scatter_aiv: UB allocation %u exceeds %u",
                           ubAlloc, UB_AVAILABLE);
                return;
            }
            AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueue;
            AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueue;
            pipe.InitBuffer(inQueue, 2, ubElements * sizeof(float));
            pipe.InitBuffer(outQueue, 2, ubElements * sizeof(float));

            AscendC::DataCopyExtParams loadParams;
            loadParams.blockCount = 1;
            loadParams.blockLen = static_cast<uint32_t>(n * sizeof(float));
            loadParams.srcStride = 0;
            loadParams.dstStride = 0;
            loadParams.rsv = 0;
            AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

            AscendC::DataCopyExtParams storeParams;
            storeParams.blockCount = 1;
            storeParams.blockLen = static_cast<uint32_t>(n * sizeof(float));
            storeParams.srcStride = 0;
            storeParams.dstStride = 0;
            storeParams.rsv = 0;

            for (uint32_t i = rLo; i < rHi; ++i) {
                uint32_t target = idxLocal.GetValue(i - rLo);

                // Load C_ws[i * n] -> UB.
                AscendC::LocalTensor<float> rowLocal = inQueue.AllocTensor<float>();
                AscendC::DataCopyPad(rowLocal, cWsGm[static_cast<uint64_t>(i) * n], loadParams, padParams);
                inQueue.EnQue<float>(rowLocal);
                AscendC::LocalTensor<float> rowIn = inQueue.DeQue<float>();

                // Copy to outQueue (UB->UB) for MTE3 sync via double-buffer.
                AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
                AscendC::DataCopy(outLocal, rowIn, ubElements);
                outQueue.EnQue<float>(outLocal);
                AscendC::LocalTensor<float> outReady = outQueue.DeQue<float>();

                // Direct write (no atomic needed for isScatter=0).
                AscendC::DataCopyPad(cGm[static_cast<uint64_t>(target) * n], outReady, storeParams);

                outQueue.FreeTensor(outReady);
                inQueue.FreeTensor(rowIn);
            }
        } else {
            // ---- Aggregation optimization path (scatter with collision) ----
            // UB budget: inQueue(2) + outQueue(2) + aggBuf(1) = 5 row buffers,
            // plus idxBuf + rowPermBuf + safety margin.
            // Note: outQueue is retained (in addition to aggBuf) for the proven
            // MTE3 store synchronization mechanism; direct aggBuf->GM stores
            // showed a Vector->MTE3 race in practice.
            uint32_t ubAlloc = 5 * ubElements * sizeof(float) + 2 * myRows * sizeof(uint32_t) + 256;
            if (ubAlloc > UB_AVAILABLE) {
                KERNEL_LOG(KERNEL_ERROR, "gather_mm_scatter_aiv: UB allocation %u exceeds %u",
                           ubAlloc, UB_AVAILABLE);
                return;
            }
            AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueue;
            AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueue;
            AscendC::TBuf<AscendC::TPosition::VECCALC> aggBufCalc;
            pipe.InitBuffer(inQueue, 2, ubElements * sizeof(float));
            pipe.InitBuffer(outQueue, 2, ubElements * sizeof(float));
            pipe.InitBuffer(aggBufCalc, ubElements * sizeof(float));
            AscendC::LocalTensor<float> aggBuf = aggBufCalc.Get<float>();

            // Row permutation buffer: rowPermBuf[i] = original row index within chunk.
            AscendC::TBuf<AscendC::TPosition::VECCALC> rowPermBufCalc;
            pipe.InitBuffer(rowPermBufCalc, idxElems * sizeof(uint32_t));
            AscendC::LocalTensor<uint32_t> rowPermBuf = rowPermBufCalc.Get<uint32_t>();

            // DataCopy params for row loads (C_ws -> UB).
            AscendC::DataCopyExtParams loadParams;
            loadParams.blockCount = 1;
            loadParams.blockLen = static_cast<uint32_t>(n * sizeof(float));
            loadParams.srcStride = 0;
            loadParams.dstStride = 0;
            loadParams.rsv = 0;
            AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

            // DataCopy params for GM stores (UB -> C).
            AscendC::DataCopyExtParams storeParams;
            storeParams.blockCount = 1;
            storeParams.blockLen = static_cast<uint32_t>(n * sizeof(float));
            storeParams.srcStride = 0;
            storeParams.dstStride = 0;
            storeParams.rsv = 0;

            // ----- Step 1: Initialize rowPermBuf -----
            for (uint32_t j = 0; j < myRows; ++j) {
                rowPermBuf.SetValue(j, j);
            }

            // ----- Step 2: Sort idxLocal by value, permuting rowPermBuf in parallel -----
            // Insertion sort O(n^2). For chunks up to ~5000 rows this is ~25M scalar
            // comparisons, acceptable on AIV. Uses LocalTensor::GetValue/SetValue
            // which are scalar UB operations (not the prohibited GlobalTensor variants).
            for (uint32_t i = 1; i < myRows; ++i) {
                uint32_t key = idxLocal.GetValue(i);
                uint32_t keyRow = rowPermBuf.GetValue(i);
                int32_t j = static_cast<int32_t>(i) - 1;
                while (j >= 0) {
                    uint32_t uj = static_cast<uint32_t>(j);
                    uint32_t val = idxLocal.GetValue(uj);
                    if (val <= key) break;
                    idxLocal.SetValue(uj + 1, val);
                    rowPermBuf.SetValue(uj + 1, rowPermBuf.GetValue(uj));
                    --j;
                }
                uint32_t insertPos = static_cast<uint32_t>(j + 1);
                idxLocal.SetValue(insertPos, key);
                rowPermBuf.SetValue(insertPos, keyRow);
            }

            KERNEL_LOG(KERNEL_INFO, "gather_mm_scatter_aiv: block=%u rows=%u sorted",
                       blockIdx, myRows);

            // ----- Step 3: Set atomic mode for scatter writes -----
            AscendC::SetAtomicAdd<float>();

            // ----- Step 4: Aggregate by target -----
            // Load the first row into aggBuf as the accumulation start.
            uint32_t prevTarget = idxLocal.GetValue(0);
            uint32_t firstOrigRow = rowPermBuf.GetValue(0);
            uint32_t uniqueTargets = (myRows > 0) ? 1 : 0;

            {
                auto firstRow = inQueue.AllocTensor<float>();
                AscendC::DataCopyPad(firstRow, cWsGm[static_cast<uint64_t>(rLo + firstOrigRow) * n],
                                     loadParams, padParams);
                inQueue.EnQue<float>(firstRow);
                firstRow = inQueue.DeQue<float>();
                AscendC::DataCopy(aggBuf, firstRow, ubElements);
                inQueue.FreeTensor(firstRow);
            }

            for (uint32_t si = 1; si < myRows; ++si) {
                uint32_t target = idxLocal.GetValue(si);
                uint32_t origRowIdx = rowPermBuf.GetValue(si);

                // Load next row (MTE2) while previous row is processed (Vector).
                auto nextRow = inQueue.AllocTensor<float>();
                AscendC::DataCopyPad(nextRow, cWsGm[static_cast<uint64_t>(rLo + origRowIdx) * n],
                                     loadParams, padParams);
                inQueue.EnQue<float>(nextRow);
                nextRow = inQueue.DeQue<float>();

                if (target == prevTarget) {
                    // Same target: accumulate into aggBuf.
                    AscendC::Add(aggBuf, aggBuf, nextRow, static_cast<int32_t>(n));
                } else {
                    // Target changed: flush aggBuf to GM via outQueue, then start new aggregation.
                    // Use outQueue for MTE3 store synchronization (proven mechanism).
                    {
                        auto outBuf = outQueue.AllocTensor<float>();
                        AscendC::DataCopy(outBuf, aggBuf, ubElements);
                        outQueue.EnQue<float>(outBuf);
                        auto outReady = outQueue.DeQue<float>();
                        AscendC::DataCopyPad(cGm[static_cast<uint64_t>(prevTarget) * n],
                                             outReady, storeParams);
                        outQueue.FreeTensor(outReady);
                    }
                    AscendC::DataCopy(aggBuf, nextRow, ubElements);
                    prevTarget = target;
                    ++uniqueTargets;
                }

                inQueue.FreeTensor(nextRow);
            }

            // ----- Step 5: Write the last accumulated aggBuf -----
            if (myRows > 0) {
                auto outBuf = outQueue.AllocTensor<float>();
                AscendC::DataCopy(outBuf, aggBuf, ubElements);
                outQueue.EnQue<float>(outBuf);
                auto outReady = outQueue.DeQue<float>();
                AscendC::DataCopyPad(cGm[static_cast<uint64_t>(prevTarget) * n],
                                     outReady, storeParams);
                outQueue.FreeTensor(outReady);
            }

            // ----- Step 6: Clear atomic mode -----
            AscendC::SetAtomicNone();

            KERNEL_LOG(KERNEL_INFO, "gather_mm_scatter_aiv: block=%u rows=%u unique_targets=%u",
                       blockIdx, myRows, uniqueTargets);
        }
    }
}

// ===== Kernel2: GatherMMScatter (B 2D, outer product) =====
// AIC: per-group Cube GEMM (A^T @ B) -> C[idxCGroup] direct write (unique writer, non-atomic).
// AIV: returns early.
extern "C" __global__ __aicore__ void gather_mm_k2(
    GM_ADDR aData,
    GM_ADDR bData,
    GM_ADDR cData,
    GM_ADDR seglenData,
    GM_ADDR groupOffsetsData,
    GM_ADDR idxAPermData,
    GM_ADDR idxBPermData,
    GM_ADDR idxCGroupData,
    uint32_t m,
    uint32_t k,
    uint32_t n,
    uint32_t numGroups,
    uint32_t nTile)
{
    AscendC::TPipe pipe;
    if ASCEND_IS_AIV {
        return;
    }

    if ASCEND_IS_AIC {
        GatherMmK2CubeProcessor proc;
        proc.Init(aData, bData, cData, seglenData, groupOffsetsData,
                  idxAPermData, idxBPermData, idxCGroupData,
                  m, k, n, numGroups, nTile, &pipe);
        proc.Process();
    }
}
