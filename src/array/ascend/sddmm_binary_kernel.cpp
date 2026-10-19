// ============================================================================
// Ascend C Kernel 实现 - sddmm_binary (两次 gather + element-wise binary)
// ============================================================================
//
// 数学公式 (DESIGN.md §1.1):
//   for e in 0..nnz-1:
//       out[e, :] = lhsFeat[indexLhs[e], :] ⊕ rhsFeat[indexRhs[e], :]
//   其中 ⊕ ∈ {add, sub, mul, div}
//   等价: out = lhsFeat[indexLhs] ⊕ rhsFeat[indexRhs]  (PyTorch advanced indexing)
//
// 实现说明 (DESIGN.md §1.3 数据流):
//   1. 使用 aclrtlaunch_* 启动模式（参考 sddmm_copy_lhs）
//   2. 流水线: MTE2→V→MTE3（有 V 计算，区别于 sddmm_copy_lhs 的 MTE2→MTE3）
//   3. 同步: TQue<VECIN> 自动同步 MTE2→V，TQue<VECOUT> 自动同步 V→MTE3
//      无需手动 SetFlag/WaitFlag（V 是天然中间消费者/生产者）
//      区别于 sddmm_copy_lhs 的 MTE2→MTE3 直接依赖（需手动 SetFlag/WaitFlag）
//   4. Double Buffer (num=2): lhsQueue/rhsQueue/outQueue 交替使用
//   5. FP16 路径: Cast h→f → binary<float> → Cast f→h（升精度运算，DESIGN.md §2.4.2）
//   6. Div: 使用不传 DivConfig 的原型（910B 不支持 config 原型，DESIGN.md §1.2.1）
//   7. 非对齐处理: DataCopyPad 统一处理 featDim 非 32B 对齐
//   8. UB_SIZE 不硬编码: 通过 TilingData.ubSize 从 Host 侧传入
// ============================================================================
#include "kernel_operator.h"
#include "sddmm_binary_tiling.h"
using namespace AscendC;
class KernelSddmmBinary {
public:
    __aicore__ inline KernelSddmmBinary(TPipe* pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR lhsFeat, GM_ADDR rhsFeat,
                                 GM_ADDR indexLhs, GM_ADDR indexRhs,
                                 GM_ADDR out, const __gm__ SddmmBinaryTilingData* tiling)
    {
        tiling_ = tiling;
        uint32_t blockIdx = GetBlockIdx();
        dtype_ = tiling->dtype;
        op_ = tiling->op;
        // DESIGN.md §2.1: 多核切分 - 每核处理的边范围
        startEdge_ = blockIdx * tiling->edgesPerCore;
        endEdge_ = (blockIdx + 1) * tiling->edgesPerCore;
        if (endEdge_ > tiling->nnz) {
            endEdge_ = tiling->nnz;
        }
        // 设置 GlobalBuffer（根据 dtype 选择 float 或 half）
        if (dtype_ == BINARY_DTYPE_FP32) {
            lhsFeatGm.SetGlobalBuffer((__gm__ float*)lhsFeat);
            rhsFeatGm.SetGlobalBuffer((__gm__ float*)rhsFeat);
            outGm.SetGlobalBuffer((__gm__ float*)out);
        } else {
            lhsFeatHalfGm.SetGlobalBuffer((__gm__ half*)lhsFeat);
            rhsFeatHalfGm.SetGlobalBuffer((__gm__ half*)rhsFeat);
            outHalfGm.SetGlobalBuffer((__gm__ half*)out);
        }
        indexLhsGm.SetGlobalBuffer((__gm__ int32_t*)indexLhs);
        indexRhsGm.SetGlobalBuffer((__gm__ int32_t*)indexRhs);
        featDim_ = tiling->featDim;
        batchSize_ = tiling->batchSize;
        featDimAligned_ = tiling->featDimAligned;
        featDimAlignedF32_ = tiling->featDimAlignedF32;
        // DESIGN.md §1.5 Buffer 规划
        uint32_t typeSize = (dtype_ == BINARY_DTYPE_FP32) ? sizeof(float) : sizeof(half);
        // idxQueue: 批量边索引（单 buffer, num=1）
        pipe_->InitBuffer(idxLhsQueue, 1, batchSize_ * sizeof(int32_t));
        pipe_->InitBuffer(idxRhsQueue, 1, batchSize_ * sizeof(int32_t));
        // 特征行 Queue: TQue Double Buffer (num=2)
        pipe_->InitBuffer(lhsQueue, 2, featDimAligned_ * typeSize);
        pipe_->InitBuffer(rhsQueue, 2, featDimAligned_ * typeSize);
        // 输出 Queue: TQue<VECOUT> Double Buffer (num=2)
        pipe_->InitBuffer(outQueue, 2, featDimAligned_ * typeSize);
        // FP16 路径额外: 3 个 TBuf<VECCALC> float 中间 buffer
        if (dtype_ == BINARY_DTYPE_FP16) {
            pipe_->InitBuffer(lhsF32Buf, featDimAlignedF32_ * sizeof(float));
            pipe_->InitBuffer(rhsF32Buf, featDimAlignedF32_ * sizeof(float));
            pipe_->InitBuffer(outF32Buf, featDimAlignedF32_ * sizeof(float));
        }
    }
    __aicore__ inline void Process()
    {
        // DESIGN.md §2.3: 空闲核早退守卫
        if (startEdge_ >= endEdge_) {
            return;
        }
        if (dtype_ == BINARY_DTYPE_FP32) {
            // FP32 路径: MTE2→V→MTE3，直接 binary<float>
            ProcessImpl<float>(lhsFeatGm, rhsFeatGm, outGm);
        } else {
            // FP16 路径: MTE2(half)→V(Cast→binary→Cast)→MTE3(half)
            ProcessImplFp16<half>(lhsFeatHalfGm, rhsFeatHalfGm, outHalfGm);
        }
    }
private:
    // ============================================================================
    // DispatchBinary - 根据 op 分发 Add/Sub/Mul/Div（DESIGN.md §2.4）
    // ============================================================================
    template <typename T>
    __aicore__ inline void DispatchBinary(LocalTensor<T>& dst,
                                            const LocalTensor<T>& src0,
                                            const LocalTensor<T>& src1,
                                            uint32_t count)
    {
        switch (op_) {
            case BINARY_OP_ADD: Add<T>(dst, src0, src1, count); break;
            case BINARY_OP_SUB: Sub<T>(dst, src0, src1, count); break;
            case BINARY_OP_MUL: Mul<T>(dst, src0, src1, count); break;
            case BINARY_OP_DIV: Div<T>(dst, src0, src1, count); break;  // 不传 DivConfig（910B）
            default: break;
        }
    }
    // ============================================================================
    // FP32 路径: MTE2→V→MTE3 流水线，TQue 自动同步（DESIGN.md §2.4.1）
    // ============================================================================
    template <typename T>
    __aicore__ inline void ProcessImpl(GlobalTensor<T>& lhsGmT,
                                        GlobalTensor<T>& rhsGmT,
                                        GlobalTensor<T>& outGmT)
    {
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPadExtParams<int32_t> idxPad{false, 0, 0, 0};
        // DESIGN.md §1.4: blockLen 用 featDim*sizeof(T)（有效字节长度）
        DataCopyExtParams rowParams{1, static_cast<uint32_t>(featDim_ * sizeof(T)), 0, 0, 0};
        for (uint32_t batchStart = startEdge_; batchStart < endEdge_; batchStart += batchSize_) {
            uint32_t actualBatch = batchSize_;
            if (batchStart + actualBatch > endEdge_) {
                actualBatch = endEdge_ - batchStart;
            }
            // 1. 批量加载索引 (MTE2 → S, TQue EnQue/DeQue 自动同步)
            DataCopyExtParams idxParams{1, static_cast<uint32_t>(actualBatch * sizeof(int32_t)), 0, 0, 0};
            LocalTensor<int32_t> idxLhs = idxLhsQueue.AllocTensor<int32_t>();
            DataCopyPad(idxLhs, indexLhsGm[batchStart], idxParams, idxPad);
            idxLhsQueue.EnQue(idxLhs);
            idxLhs = idxLhsQueue.DeQue<int32_t>();
            LocalTensor<int32_t> idxRhs = idxRhsQueue.AllocTensor<int32_t>();
            DataCopyPad(idxRhs, indexRhsGm[batchStart], idxParams, idxPad);
            idxRhsQueue.EnQue(idxRhs);
            idxRhs = idxRhsQueue.DeQue<int32_t>();
            // 2. 逐边 Gather + Compute + Store
            for (uint32_t i = 0; i < actualBatch; i++) {
                uint32_t lhsIdx = static_cast<uint32_t>(idxLhs.GetValue(i));
                uint32_t rhsIdx = static_cast<uint32_t>(idxRhs.GetValue(i));
                // MTE2: gather lhs row (TQue<VECIN> EnQue 自动同步 MTE2→V)
                LocalTensor<T> lhs = lhsQueue.AllocTensor<T>();
                DataCopyPad(lhs, lhsGmT[lhsIdx * featDim_], rowParams, pad);
                lhsQueue.EnQue(lhs);
                // MTE2: gather rhs row
                LocalTensor<T> rhs = rhsQueue.AllocTensor<T>();
                DataCopyPad(rhs, rhsGmT[rhsIdx * featDim_], rowParams, pad);
                rhsQueue.EnQue(rhs);
                // V: compute binary (TQue<VECIN> DeQue 自动同步 MTE2→V)
                LocalTensor<T> lhsReady = lhsQueue.DeQue<T>();
                LocalTensor<T> rhsReady = rhsQueue.DeQue<T>();
                LocalTensor<T> out = outQueue.AllocTensor<T>();
                // DESIGN.md §1.4: count 用 featDim（有效长度，非对齐长度）
                DispatchBinary<T>(out, lhsReady, rhsReady, featDim_);
                // TQue<VECOUT> EnQue 自动同步 V→MTE3
                outQueue.EnQue(out);
                // MTE3: store output (TQue<VECOUT> DeQue 自动同步 V→MTE3)
                LocalTensor<T> outReady = outQueue.DeQue<T>();
                DataCopyPad(outGmT[(batchStart + i) * featDim_], outReady, rowParams);
                lhsQueue.FreeTensor(lhsReady);
                rhsQueue.FreeTensor(rhsReady);
                outQueue.FreeTensor(outReady);
            }
            idxLhsQueue.FreeTensor(idxLhs);
            idxRhsQueue.FreeTensor(idxRhs);
        }
    }
    // ============================================================================
    // FP16 路径: MTE2(half)→V(Cast→binary→Cast)→MTE3(half)（DESIGN.md §2.4.2）
    // ============================================================================
    // FP16 需要升精度到 float 运算，特别是 div 需要高精度
    // 流程: gather half → Cast h→f → binary<float> → Cast f→h → store half
    // ============================================================================
    template <typename T>
    __aicore__ inline void ProcessImplFp16(GlobalTensor<T>& lhsGmT,
                                            GlobalTensor<T>& rhsGmT,
                                            GlobalTensor<T>& outGmT)
    {
        // T 应为 half；此处保持模板形式以复用 GlobalTensor 类型推导
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPadExtParams<int32_t> idxPad{false, 0, 0, 0};
        DataCopyExtParams rowParams{1, static_cast<uint32_t>(featDim_ * sizeof(T)), 0, 0, 0};
        // FP32 中间 buffer (VECCALC, TBuf)
        LocalTensor<float> lhsF32 = lhsF32Buf.Get<float>();
        LocalTensor<float> rhsF32 = rhsF32Buf.Get<float>();
        LocalTensor<float> outF32 = outF32Buf.Get<float>();
        for (uint32_t batchStart = startEdge_; batchStart < endEdge_; batchStart += batchSize_) {
            uint32_t actualBatch = batchSize_;
            if (batchStart + actualBatch > endEdge_) {
                actualBatch = endEdge_ - batchStart;
            }
            // 1. 批量加载索引
            DataCopyExtParams idxParams{1, static_cast<uint32_t>(actualBatch * sizeof(int32_t)), 0, 0, 0};
            LocalTensor<int32_t> idxLhs = idxLhsQueue.AllocTensor<int32_t>();
            DataCopyPad(idxLhs, indexLhsGm[batchStart], idxParams, idxPad);
            idxLhsQueue.EnQue(idxLhs);
            idxLhs = idxLhsQueue.DeQue<int32_t>();
            LocalTensor<int32_t> idxRhs = idxRhsQueue.AllocTensor<int32_t>();
            DataCopyPad(idxRhs, indexRhsGm[batchStart], idxParams, idxPad);
            idxRhsQueue.EnQue(idxRhs);
            idxRhs = idxRhsQueue.DeQue<int32_t>();
            // 2. 逐边 Gather + Cast + Compute + Cast + Store
            for (uint32_t i = 0; i < actualBatch; i++) {
                uint32_t lhsIdx = static_cast<uint32_t>(idxLhs.GetValue(i));
                uint32_t rhsIdx = static_cast<uint32_t>(idxRhs.GetValue(i));
                // MTE2: gather lhs row (half)
                LocalTensor<T> lhs = lhsQueue.AllocTensor<T>();
                DataCopyPad(lhs, lhsGmT[lhsIdx * featDim_], rowParams, pad);
                lhsQueue.EnQue(lhs);
                // MTE2: gather rhs row (half)
                LocalTensor<T> rhs = rhsQueue.AllocTensor<T>();
                DataCopyPad(rhs, rhsGmT[rhsIdx * featDim_], rowParams, pad);
                rhsQueue.EnQue(rhs);
                // V: Cast half→float + binary + Cast float→half
                LocalTensor<T> lhsReady = lhsQueue.DeQue<T>();
                LocalTensor<T> rhsReady = rhsQueue.DeQue<T>();
                // DESIGN.md §2.7: h→f 用 CAST_NONE（无损）
                Cast<float, T>(lhsF32, lhsReady, RoundMode::CAST_NONE, featDim_);
                Cast<float, T>(rhsF32, rhsReady, RoundMode::CAST_NONE, featDim_);
                // binary 在 float 精度下运算
                DispatchBinary<float>(outF32, lhsF32, rhsF32, featDim_);
                // DESIGN.md §2.7: f→h 用 CAST_ROUND（四舍五入，精度最优）
                LocalTensor<T> out = outQueue.AllocTensor<T>();
                Cast<T, float>(out, outF32, RoundMode::CAST_ROUND, featDim_);
                outQueue.EnQue(out);
                // MTE3: store output (half)
                LocalTensor<T> outReady = outQueue.DeQue<T>();
                DataCopyPad(outGmT[(batchStart + i) * featDim_], outReady, rowParams);
                lhsQueue.FreeTensor(lhsReady);
                rhsQueue.FreeTensor(rhsReady);
                outQueue.FreeTensor(outReady);
            }
            idxLhsQueue.FreeTensor(idxLhs);
            idxRhsQueue.FreeTensor(idxRhs);
        }
    }
private:
    TPipe* pipe_;
    const __gm__ SddmmBinaryTilingData* tiling_;
    // GlobalTensor（FP32/FP16 双版本，根据 dtype 选择）
    GlobalTensor<float> lhsFeatGm;
    GlobalTensor<float> rhsFeatGm;
    GlobalTensor<half> lhsFeatHalfGm;
    GlobalTensor<half> rhsFeatHalfGm;
    GlobalTensor<int32_t> indexLhsGm;
    GlobalTensor<int32_t> indexRhsGm;
    GlobalTensor<float> outGm;
    GlobalTensor<half> outHalfGm;
    // TQue<VECIN>: MTE2 搬运缓冲区
    TQue<TPosition::VECIN, 1> idxLhsQueue;     // lhs 索引批量加载（单 buffer）
    TQue<TPosition::VECIN, 1> idxRhsQueue;     // rhs 索引批量加载（单 buffer）
    TQue<TPosition::VECIN, 2> lhsQueue;        // lhs 特征行（Double Buffer）
    TQue<TPosition::VECIN, 2> rhsQueue;        // rhs 特征行（Double Buffer）
    // TQue<VECOUT>: MTE3 输出缓冲区（Double Buffer）
    TQue<TPosition::VECOUT, 2> outQueue;       // 输出特征行
    // TBuf<VECCALC>: FP16 路径 float 中间 buffer
    TBuf<TPosition::VECCALC> lhsF32Buf;        // lhs 升精度 float
    TBuf<TPosition::VECCALC> rhsF32Buf;        // rhs 升精度 float
    TBuf<TPosition::VECCALC> outF32Buf;        // 运算结果 float
    uint32_t startEdge_ = 0;
    uint32_t endEdge_ = 0;
    uint32_t featDim_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t dtype_ = 0;
    uint32_t op_ = 0;
    uint32_t featDimAligned_ = 0;
    uint32_t featDimAlignedF32_ = 0;
};
// ============================================================================
// 修正 FP16 路径分发：ProcessImpl<half> 调用 ProcessImplFp16
// 通过特化避免 if constexpr 依赖（保持与 DESIGN.md 伪代码结构一致）
// ============================================================================
namespace {
// 由于 ProcessImpl 是模板成员函数，无法在类外特化，这里通过 Process 中的 if 分发处理
// Process() 中 dtype_ == BINARY_DTYPE_FP16 时调用 ProcessImplFp16<half>
}
extern "C" __global__ __aicore__ void sddmm_binary_kernel(
    GM_ADDR lhsFeat, GM_ADDR rhsFeat,
    GM_ADDR indexLhs, GM_ADDR indexRhs,
    GM_ADDR out, GM_ADDR tiling)
{
    TPipe pipe;
    KernelSddmmBinary op(&pipe);
    op.Init(lhsFeat, rhsFeat, indexLhs, indexRhs, out, (__gm__ SddmmBinaryTilingData*)tiling);
    op.Process();
}
