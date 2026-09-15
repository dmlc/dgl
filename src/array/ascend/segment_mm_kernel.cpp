#include "segment_mm_cube_processor.h"

// SegmentMM / SegmentMMBackwardB device kernels (Cube(AIC)+MemBase-vector(AIV) mix).

extern "C" __global__ __aicore__ void segment_mm(
    GM_ADDR aData,
    GM_ADDR bData,
    GM_ADDR cData,
    uint32_t m,
    uint32_t k,
    uint32_t n,
    uint32_t mGrid,
    uint32_t nGrid,
    uint32_t nTile,
    uint32_t kTile,
    uint32_t bTrans)
{
    AscendC::TPipe pipe;
    if ASCEND_IS_AIV {
        return;
    }

    if ASCEND_IS_AIC {
        SegmentMmCubeProcessor proc;
        proc.Init(aData, bData, cData, m, k, n, mGrid, nGrid, nTile, kTile, bTrans, &pipe);
        proc.Process();
    }
}

extern "C" __global__ __aicore__ void segment_mm_backward_b(
    GM_ADDR aData,
    GM_ADDR dCData,
    GM_ADDR dBData,
    uint32_t k,
    uint32_t n,
    uint32_t m,
    uint32_t mGrid,
    uint32_t nGrid,
    uint32_t nTile,
    uint32_t kTile)
{
    AscendC::TPipe pipe;
    if ASCEND_IS_AIV {
        return;
    }

    if ASCEND_IS_AIC {
        SegmentMmBwdBCubeProcessor proc;
        proc.Init(aData, dCData, dBData, k, n, m, mGrid, nGrid, nTile, kTile, &pipe);
        proc.Process();
    }
}
