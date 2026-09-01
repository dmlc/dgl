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

        // UB buffer for one row (padded to 32B for DataCopy alignment).
        // inQueue + outQueue each allocate 2 buffers of ubElements floats.
        // Check that the total UB allocation fits within the 192KB AIV UB budget.
        AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueue;
        AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueue;
        constexpr uint32_t UB_AVAILABLE = 192 * 1024;
        uint32_t ubElements = n;
        if (ubElements < 8) ubElements = 8;
        // Round up to 8 for alignment.
        ubElements = (ubElements + 7) / 8 * 8;
        // 4 buffers (inQueue*2 + outQueue*2) + idxBuf (up to m/40*4)
        uint32_t ubAlloc = 4 * ubElements * sizeof(float) + 256;
        if (ubAlloc > UB_AVAILABLE) {
            KERNEL_LOG(KERNEL_ERROR, "gather_mm_scatter_aiv: UB allocation %u exceeds %u",
                       ubAlloc, UB_AVAILABLE);
            return;
        }
        pipe.InitBuffer(inQueue, 2, ubElements * sizeof(float));
        pipe.InitBuffer(outQueue, 2, ubElements * sizeof(float));

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t rowsPerCore = (m + blockNum - 1) / blockNum;
        uint32_t rLo = blockIdx * rowsPerCore;
        uint32_t rHi = rLo + rowsPerCore;
        if (rHi > m) rHi = m;
        if (rLo >= rHi) return;

        // Cache this core's scatter indices into UB via DataCopyPad (coherent DMA).
        // Scalar GM reads (GlobalTensor::GetValue) are non-coherent in AIV context;
        // reading from UB ensures coherent and fast access.
        uint32_t myRows = rHi - rLo;
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

        if (isScatter) {
            AscendC::SetAtomicAdd<float>();
        }

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

            // Load C_ws[i * n] -> UB (DataCopyPad handles non-aligned n).
            AscendC::LocalTensor<float> rowLocal = inQueue.AllocTensor<float>();
            AscendC::DataCopyPad(rowLocal, cWsGm[static_cast<uint64_t>(i) * n], loadParams, padParams);
            inQueue.EnQue<float>(rowLocal);
            AscendC::LocalTensor<float> rowIn = inQueue.DeQue<float>();

            // Copy row to outQueue (UB->UB). The outQueue double-buffer provides the
            // MTE3-store -> next-MTE2-load synchronization: a slot is only reused two
            // iterations later, by which time the prior store has completed. Copying
            // ubElements (32B-aligned padded count) instead of n avoids the DataCopy
            // 32B-alignment requirement for DataCopy; n not divisible by 8
            // requires padded copy width.
            AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
            AscendC::DataCopy(outLocal, rowIn, ubElements);
            outQueue.EnQue<float>(outLocal);
            AscendC::LocalTensor<float> outReady = outQueue.DeQue<float>();

            // Store UB -> C[target * n] (atomic add if isScatter).
            AscendC::DataCopyPad(cGm[static_cast<uint64_t>(target) * n], outReady, storeParams);

            outQueue.FreeTensor(outReady);
            inQueue.FreeTensor(rowIn);
        }

        if (isScatter) {
            AscendC::SetAtomicNone();
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
