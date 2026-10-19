#include "bspmm_sum_vector_processor.h"
#include "bspmm_sum_cube_processor.h"

extern "C" __global__ __aicore__ void bspmm_sum(
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
    uint32_t columnToEdgeLength,
    uint32_t batchCount)
{
    AscendC::TPipe pipe;
    if ASCEND_IS_AIV {
        AivVectorProcessor vectorProcessor;
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
            batchCount,
            nonZeroCount,
            vectorWindowCount,
            &pipe);
        vectorProcessor.Process();
    }

    if ASCEND_IS_AIC {
        AicCubeProcessor cubeProcessor;
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
            batchCount,
            totalTcBlocks,
            cubeWindowCount,
            columnToEdgeLength,
            &pipe);
        cubeProcessor.Process();
    }
}
