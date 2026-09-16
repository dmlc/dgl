// ============================================================================
// SDDMM dot kernel for Ascend (COO format, u_dot_v scenario)
// ============================================================================
// Math: for each edge e=(u,v): out[e] = sum_k(lhs[u][k] * rhs[v][k])
//
// Entry point: sddmm_dot_coo (registered via extern "C" __global__ __aicore__)
// Host side:   aclrtlaunch_sddmm_dot_coo() declared in sddmm.cc
// ============================================================================

#include <cstdint>
#include "kernel_operator.h"

// ============================================================================
// Tiling struct (shared between kernel and host via GM_ADDR)
// ============================================================================
constexpr uint32_t UB_RESERVED_SDDMM = 2 * 1024;
constexpr uint32_t MAX_BATCH_SDDMM = 255;
constexpr uint32_t DTYPE_FP32_SDDMM = 0;
constexpr uint32_t DTYPE_FP16_SDDMM = 1;
constexpr uint32_t ALIGN_BYTES_SDDMM = 32;
constexpr uint32_t MAX_REDUCE_MASK_FLOAT_SDDMM = 64;

struct SddmmTilingData {
    uint32_t numEdges;
    uint32_t featDim;
    uint32_t blockDim;
    uint32_t edgesPerCore;
    uint32_t batchSize;
    uint32_t dtype;
    uint32_t featDimAligned;
    uint32_t ubSize;
};

// ============================================================================
// Kernel class
// ============================================================================
class KernelSddmm {
public:
    __aicore__ inline KernelSddmm(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR lhs, GM_ADDR rhs, GM_ADDR row, GM_ADDR col, GM_ADDR out,
                                const __gm__ SddmmTilingData* tiling)
    {
        tiling_ = tiling;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        dtype_ = tiling->dtype;

        startEdge_ = blockIdx * tiling->edgesPerCore;
        endEdge_ = (blockIdx + 1) * tiling->edgesPerCore;
        if (endEdge_ > tiling->numEdges) {
            endEdge_ = tiling->numEdges;
        }

        if (dtype_ == DTYPE_FP32_SDDMM) {
            lhsGm.SetGlobalBuffer((__gm__ float*)lhs);
            rhsGm.SetGlobalBuffer((__gm__ float*)rhs);
            outGm.SetGlobalBuffer((__gm__ float*)out);
        } else {
            lhsHalfGm.SetGlobalBuffer((__gm__ half*)lhs);
            rhsHalfGm.SetGlobalBuffer((__gm__ half*)rhs);
            outHalfGm.SetGlobalBuffer((__gm__ half*)out);
        }
        rowGm.SetGlobalBuffer((__gm__ int32_t*)row);
        colGm.SetGlobalBuffer((__gm__ int32_t*)col);

        featDim_ = tiling->featDim;
        batchSize_ = tiling->batchSize;
        featDimAligned_ = tiling->featDimAligned;

        uint32_t featAlign = featDimAligned_;

        pipe_->InitBuffer(indexSrcQueue, 1, batchSize_ * sizeof(int32_t));
        pipe_->InitBuffer(indexDstQueue, 1, batchSize_ * sizeof(int32_t));

        uint32_t featBufSize = (dtype_ == DTYPE_FP32_SDDMM)
            ? (featAlign * sizeof(float))
            : (((featDim_ * sizeof(half) + ALIGN_BYTES_SDDMM - 1) / ALIGN_BYTES_SDDMM * ALIGN_BYTES_SDDMM));
        pipe_->InitBuffer(lhsQueue, 2, featBufSize);
        pipe_->InitBuffer(rhsQueue, 2, featBufSize);

        pipe_->InitBuffer(mulF32BatchBuf, batchSize_ * featAlign * sizeof(float));
        if (dtype_ == DTYPE_FP16_SDDMM) {
            pipe_->InitBuffer(lhsF32Buf, featAlign * sizeof(float));
            pipe_->InitBuffer(rhsF32Buf, featAlign * sizeof(float));
        }
        pipe_->InitBuffer(partSumBuf, batchSize_ * sizeof(float));
        if (dtype_ == DTYPE_FP16_SDDMM) {
            pipe_->InitBuffer(outF32Buf, batchSize_ * sizeof(float));
        }
        uint32_t outElemSize = (dtype_ == DTYPE_FP32_SDDMM) ? sizeof(float) : sizeof(half);
        pipe_->InitBuffer(outQueue, 1, batchSize_ * outElemSize);
    }

    __aicore__ inline void Process()
    {
        if (startEdge_ >= endEdge_) { return; }
        if (dtype_ == DTYPE_FP32_SDDMM) {
            ProcessFp32();
        } else {
            ProcessFp16();
        }
    }

private:
    __aicore__ inline void ProcessFp32()
    {
        for (uint32_t batchStart = startEdge_; batchStart < endEdge_; batchStart += batchSize_) {
            uint32_t actualBatch = batchSize_;
            if (batchStart + actualBatch > endEdge_) { actualBatch = endEdge_ - batchStart; }

            AscendC::LocalTensor<int32_t> srcIdxLocal = indexSrcQueue.AllocTensor<int32_t>();
            AscendC::LocalTensor<int32_t> dstIdxLocal = indexDstQueue.AllocTensor<int32_t>();
            AscendC::DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(actualBatch * sizeof(int32_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int32_t> idxPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(srcIdxLocal, rowGm[batchStart], idxCopyParams, idxPadParams);
            AscendC::DataCopyPad(dstIdxLocal, colGm[batchStart], idxCopyParams, idxPadParams);
            indexSrcQueue.EnQue(srcIdxLocal);
            indexDstQueue.EnQue(dstIdxLocal);
            AscendC::LocalTensor<int32_t> srcIdx = indexSrcQueue.DeQue<int32_t>();
            AscendC::LocalTensor<int32_t> dstIdx = indexDstQueue.DeQue<int32_t>();

            AscendC::LocalTensor<float> mulBatch = mulF32BatchBuf.Get<float>();
            AscendC::LocalTensor<float> lhsCur = lhsQueue.AllocTensor<float>();
            AscendC::LocalTensor<float> rhsCur = rhsQueue.AllocTensor<float>();
            {
                uint32_t u0 = static_cast<uint32_t>(srcIdx.GetValue(0));
                uint32_t v0 = static_cast<uint32_t>(dstIdx.GetValue(0));
                AscendC::DataCopyExtParams featParams{1, static_cast<uint32_t>(featDim_ * sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<float> featPad{false, 0, 0, 0.0f};
                AscendC::DataCopyPad(lhsCur, lhsGm[u0 * featDim_], featParams, featPad);
                AscendC::DataCopyPad(rhsCur, rhsGm[v0 * featDim_], featParams, featPad);
                lhsQueue.EnQue(lhsCur);
                rhsQueue.EnQue(rhsCur);
                lhsCur = lhsQueue.DeQue<float>();
                rhsCur = rhsQueue.DeQue<float>();
            }
            for (uint32_t i = 0; i < actualBatch; i++) {
                if (i + 1 < actualBatch) {
                    uint32_t uN = static_cast<uint32_t>(srcIdx.GetValue(i + 1));
                    uint32_t vN = static_cast<uint32_t>(dstIdx.GetValue(i + 1));
                    AscendC::LocalTensor<float> lhsNxt = lhsQueue.AllocTensor<float>();
                    AscendC::LocalTensor<float> rhsNxt = rhsQueue.AllocTensor<float>();
                    AscendC::DataCopyExtParams featParams{1, static_cast<uint32_t>(featDim_ * sizeof(float)), 0, 0, 0};
                    AscendC::DataCopyPadExtParams<float> featPad{false, 0, 0, 0.0f};
                    AscendC::DataCopyPad(lhsNxt, lhsGm[uN * featDim_], featParams, featPad);
                    AscendC::DataCopyPad(rhsNxt, rhsGm[vN * featDim_], featParams, featPad);
                    lhsQueue.EnQue(lhsNxt);
                    rhsQueue.EnQue(rhsNxt);
                }
                AscendC::LocalTensor<float> mulDst = mulBatch[i * featDimAligned_];
                AscendC::Mul<float>(mulDst, lhsCur, rhsCur, featDim_);
                lhsQueue.FreeTensor(lhsCur);
                rhsQueue.FreeTensor(rhsCur);
                if (i + 1 < actualBatch) {
                    lhsCur = lhsQueue.DeQue<float>();
                    rhsCur = rhsQueue.DeQue<float>();
                }
            }
            AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
            BatchReduceSum(outLocal, mulBatch, actualBatch);
            outQueue.EnQue<float>(outLocal);
            AscendC::LocalTensor<float> outResult = outQueue.DeQue<float>();
            AscendC::DataCopyExtParams outParams{1, static_cast<uint32_t>(actualBatch * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(outGm[batchStart], outResult, outParams);
            outQueue.FreeTensor(outResult);
            indexSrcQueue.FreeTensor(srcIdx);
            indexDstQueue.FreeTensor(dstIdx);
        }
    }

    __aicore__ inline void ProcessFp16()
    {
        for (uint32_t batchStart = startEdge_; batchStart < endEdge_; batchStart += batchSize_) {
            uint32_t actualBatch = batchSize_;
            if (batchStart + actualBatch > endEdge_) { actualBatch = endEdge_ - batchStart; }

            AscendC::LocalTensor<int32_t> srcIdxLocal = indexSrcQueue.AllocTensor<int32_t>();
            AscendC::LocalTensor<int32_t> dstIdxLocal = indexDstQueue.AllocTensor<int32_t>();
            AscendC::DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(actualBatch * sizeof(int32_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int32_t> idxPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(srcIdxLocal, rowGm[batchStart], idxCopyParams, idxPadParams);
            AscendC::DataCopyPad(dstIdxLocal, colGm[batchStart], idxCopyParams, idxPadParams);
            indexSrcQueue.EnQue(srcIdxLocal);
            indexDstQueue.EnQue(dstIdxLocal);
            AscendC::LocalTensor<int32_t> srcIdx = indexSrcQueue.DeQue<int32_t>();
            AscendC::LocalTensor<int32_t> dstIdx = indexDstQueue.DeQue<int32_t>();

            AscendC::LocalTensor<float> mulBatch = mulF32BatchBuf.Get<float>();
            AscendC::LocalTensor<half> lhsHalfCur = lhsQueue.AllocTensor<half>();
            AscendC::LocalTensor<half> rhsHalfCur = rhsQueue.AllocTensor<half>();
            {
                uint32_t u0 = static_cast<uint32_t>(srcIdx.GetValue(0));
                uint32_t v0 = static_cast<uint32_t>(dstIdx.GetValue(0));
                AscendC::DataCopyExtParams featParams{1, static_cast<uint32_t>(featDim_ * sizeof(half)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<half> featPad{false, 0, 0, half(0)};
                AscendC::DataCopyPad(lhsHalfCur, lhsHalfGm[u0 * featDim_], featParams, featPad);
                AscendC::DataCopyPad(rhsHalfCur, rhsHalfGm[v0 * featDim_], featParams, featPad);
                lhsQueue.EnQue(lhsHalfCur);
                rhsQueue.EnQue(rhsHalfCur);
                lhsHalfCur = lhsQueue.DeQue<half>();
                rhsHalfCur = rhsQueue.DeQue<half>();
            }
            for (uint32_t i = 0; i < actualBatch; i++) {
                if (i + 1 < actualBatch) {
                    uint32_t uN = static_cast<uint32_t>(srcIdx.GetValue(i + 1));
                    uint32_t vN = static_cast<uint32_t>(dstIdx.GetValue(i + 1));
                    AscendC::LocalTensor<half> lhsHalfNxt = lhsQueue.AllocTensor<half>();
                    AscendC::LocalTensor<half> rhsHalfNxt = rhsQueue.AllocTensor<half>();
                    AscendC::DataCopyExtParams featParams{1, static_cast<uint32_t>(featDim_ * sizeof(half)), 0, 0, 0};
                    AscendC::DataCopyPadExtParams<half> featPad{false, 0, 0, half(0)};
                    AscendC::DataCopyPad(lhsHalfNxt, lhsHalfGm[uN * featDim_], featParams, featPad);
                    AscendC::DataCopyPad(rhsHalfNxt, rhsHalfGm[vN * featDim_], featParams, featPad);
                    lhsQueue.EnQue(lhsHalfNxt);
                    rhsQueue.EnQue(rhsHalfNxt);
                }
                AscendC::LocalTensor<float> lhsF32 = lhsF32Buf.Get<float>();
                AscendC::LocalTensor<float> rhsF32 = rhsF32Buf.Get<float>();
                AscendC::Cast<float, half>(lhsF32, lhsHalfCur, AscendC::RoundMode::CAST_NONE, featDim_);
                AscendC::Cast<float, half>(rhsF32, rhsHalfCur, AscendC::RoundMode::CAST_NONE, featDim_);
                AscendC::LocalTensor<float> mulDst = mulBatch[i * featDimAligned_];
                AscendC::Mul<float>(mulDst, lhsF32, rhsF32, featDim_);
                lhsQueue.FreeTensor(lhsHalfCur);
                rhsQueue.FreeTensor(rhsHalfCur);
                if (i + 1 < actualBatch) {
                    lhsHalfCur = lhsQueue.DeQue<half>();
                    rhsHalfCur = rhsQueue.DeQue<half>();
                }
            }
            AscendC::LocalTensor<float> outF32 = outF32Buf.Get<float>();
            BatchReduceSum(outF32, mulBatch, actualBatch);
            AscendC::LocalTensor<half> outLocal = outQueue.AllocTensor<half>();
            AscendC::Cast<half, float>(outLocal, outF32, AscendC::RoundMode::CAST_ROUND, actualBatch);
            outQueue.EnQue<half>(outLocal);
            AscendC::LocalTensor<half> outResult = outQueue.DeQue<half>();
            AscendC::DataCopyExtParams outParams{1, static_cast<uint32_t>(actualBatch * sizeof(half)), 0, 0, 0};
            AscendC::DataCopyPad(outHalfGm[batchStart], outResult, outParams);
            outQueue.FreeTensor(outResult);
            indexSrcQueue.FreeTensor(srcIdx);
            indexDstQueue.FreeTensor(dstIdx);
        }
    }

    __aicore__ inline void BatchReduceSum(AscendC::LocalTensor<float>& dst,
                                           AscendC::LocalTensor<float>& src,
                                           uint32_t batchCount)
    {
        if (featDim_ <= MAX_REDUCE_MASK_FLOAT_SDDMM) {
            AscendC::WholeReduceSum<float, true>(
                dst, src, static_cast<int32_t>(featDim_),
                static_cast<int32_t>(batchCount),
                1, 1, static_cast<int32_t>(featDimAligned_ / 8));
        } else {
            uint32_t offset = 0;
            uint32_t remaining = featDim_;
            bool firstChunk = true;
            while (remaining > 0) {
                uint32_t chunkLen = (remaining > MAX_REDUCE_MASK_FLOAT_SDDMM)
                    ? MAX_REDUCE_MASK_FLOAT_SDDMM : remaining;
                AscendC::LocalTensor<float> srcChunk = src[offset];
                if (firstChunk) {
                    AscendC::WholeReduceSum<float, true>(
                        dst, srcChunk, static_cast<int32_t>(chunkLen),
                        static_cast<int32_t>(batchCount),
                        1, 1, static_cast<int32_t>(featDimAligned_ / 8));
                    firstChunk = false;
                } else {
                    AscendC::LocalTensor<float> partSum = partSumBuf.Get<float>();
                    AscendC::WholeReduceSum<float, true>(
                        partSum, srcChunk, static_cast<int32_t>(chunkLen),
                        static_cast<int32_t>(batchCount),
                        1, 1, static_cast<int32_t>(featDimAligned_ / 8));
                    AscendC::Add<float>(dst, dst, partSum, batchCount);
                }
                offset += chunkLen;
                remaining -= chunkLen;
            }
        }
    }

private:
    AscendC::TPipe* pipe_;
    const __gm__ SddmmTilingData* tiling_;
    AscendC::GlobalTensor<float> lhsGm;
    AscendC::GlobalTensor<float> rhsGm;
    AscendC::GlobalTensor<int32_t> rowGm;
    AscendC::GlobalTensor<int32_t> colGm;
    AscendC::GlobalTensor<float> outGm;
    AscendC::GlobalTensor<half> lhsHalfGm;
    AscendC::GlobalTensor<half> rhsHalfGm;
    AscendC::GlobalTensor<half> outHalfGm;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> indexSrcQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> indexDstQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> lhsQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> rhsQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mulF32BatchBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> partSumBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outF32Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lhsF32Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rhsF32Buf;
    uint32_t startEdge_ = 0;
    uint32_t endEdge_ = 0;
    uint32_t featDim_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t dtype_ = 0;
    uint32_t featDimAligned_ = 0;
};

// ============================================================================
// Kernel entry point
// ============================================================================
extern "C" __global__ __aicore__ void sddmm_dot_coo(
    GM_ADDR lhs, GM_ADDR rhs, GM_ADDR row, GM_ADDR col,
    GM_ADDR out, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    KernelSddmm op(&pipe);
    op.Init(lhs, rhs, row, col, out, (__gm__ SddmmTilingData*)tiling);
    op.Process();
}
