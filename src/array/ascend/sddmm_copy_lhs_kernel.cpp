// ============================================================================
// Ascend C Kernel 实现 - sddmm_copy_lhs (Gather 按索引 gather 特征行)
// ============================================================================
//
// 数学公式 (DESIGN.md §1.1, 纯 Gather 无归约):
//   for i in 0..nnz-1:
//       out[i, :] = feat[index[i], :]
//   等价: torch.index_select(feat, 0, index)
//
// 实现说明:
//   1. 使用 aclrtlaunch_* 启动模式（参考 spmm/sddmm）
//   2. 纯数据搬运（MTE2→MTE3），无 V pipeline 计算:
//      使用 TQue<VECIN> 管理 MTE2 搬运缓冲区（满足 API 最佳实践：MTE 缓冲区用 TQue）
//      TQue EnQue/DeQue 提供 MTE2→V 同步和 buffer 轮转管理
//      额外使用 SetFlag/WaitFlag<MTE2_MTE3> + <MTE3_MTE2> 显式同步 MTE2↔MTE3
//      （TQue EnQue/DeQue 仅同步 MTE2→V，不覆盖 MTE2→MTE3 直接依赖）
//   3. SetFlag/WaitFlag 使用 immediate Set→Wait 模式（Set 紧接 Wait）:
//      参考 Ascend C 内部代码 dav_c220/kernel_operator_sync_impl.h 的同步模式:
//        SetFlag<HardEvent::MTE3_MTE2>(evt);
//        WaitFlag<HardEvent::MTE3_MTE2>(evt);
//      Set 插入源 pipe 队列（MTE3），Wait 插入目标 pipe 队列（MTE2），
//      形成跨流水同步点。Event ID 通过 AllocEventID 分配（唯一、持久）。
//   4. 不使用 PipeBarrier<PIPE_ALL>（仅调试用，性能差）
//   5. Double Buffer (num=2): 两个 buffer 交替使用，MTE3 顺序处理确保 buffer 复用安全
//   6. API 黑名单合规: 不使用 GlobalTensor::GetValue/SetValue
//   7. 非对齐处理: DataCopyPad 统一处理 feat_dim 非 32B 对齐
//   8. UB_SIZE 不硬编码: 通过 TilingData.ubSize 从 Host 侧传入
// ============================================================================
#include "kernel_operator.h"
#include "sddmm_copy_lhs_tiling.h"
class KernelSddmmCopyLhs {
public:
    __aicore__ inline KernelSddmmCopyLhs(AscendC::TPipe* pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR feat, GM_ADDR index, GM_ADDR out,
                                 const __gm__ SddmmCopyLhsTilingData* tiling)
    {
        tiling_ = tiling;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        dtype_ = tiling->dtype;
        startEdge_ = blockIdx * tiling->edgesPerCore;
        endEdge_ = (blockIdx + 1) * tiling->edgesPerCore;
        if (endEdge_ > tiling->nnz) {
            endEdge_ = tiling->nnz;
        }
        if (dtype_ == DTYPE_FP32) {
            featGm.SetGlobalBuffer((__gm__ float*)feat);
            outGm.SetGlobalBuffer((__gm__ float*)out);
        } else {
            featHalfGm.SetGlobalBuffer((__gm__ half*)feat);
            outHalfGm.SetGlobalBuffer((__gm__ half*)out);
        }
        indexGm.SetGlobalBuffer((__gm__ int32_t*)index);
        featDim_ = tiling->featDim;
        batchSize_ = tiling->batchSize;
        featDimAligned_ = tiling->featDimAligned;
        // idxQueue: 批量边索引
        pipe_->InitBuffer(idxQueue, 1, batchSize_ * sizeof(int32_t));
        // featQueue: TQue<VECIN> Double Buffer (num=2)
        uint32_t typeSize = (dtype_ == DTYPE_FP32) ? sizeof(float) : sizeof(half);
        pipe_->InitBuffer(featQueue, 2, featDimAligned_ * typeSize);
        // 分配 Event ID（MTE2↔MTE3 跨流水同步）
        evtM2M3_ = static_cast<int32_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE2_MTE3>());
        evtM3M2_ = static_cast<int32_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
    }
    __aicore__ inline void Process()
    {
        if (startEdge_ >= endEdge_) {
            return;
        }
        if (dtype_ == DTYPE_FP32) {
            ProcessImpl<float>(featGm, outGm);
        } else {
            ProcessImpl<half>(featHalfGm, outHalfGm);
        }
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(evtM2M3_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(evtM3M2_);
    }
private:
    // ============================================================================
    // 核心模板：FP32/FP16 共用，T=float 或 T=half
    // 数据流：MTE2(GM→UB) → MTE3(UB→GM)，无 V 计算
    // 同步策略：
    //   - TQue<VECIN> AllocTensor/EnQue/DeQue/FreeTensor: buffer 轮转管理
    //   - SetFlag/WaitFlag<MTE2_MTE3>: MTE3 等待 MTE2 完成（数据就绪）
    //   - SetFlag/WaitFlag<MTE3_MTE2>: MTE2 等待 MTE3 完成（buffer 复用安全）
    //   - Set/Wait 使用 immediate 模式（Set 紧接 Wait），参考 Ascend C 内部同步实现
    // ============================================================================
    template <typename T>
    __aicore__ inline void ProcessImpl(AscendC::GlobalTensor<T>& featGmT,
                                        AscendC::GlobalTensor<T>& outGmT)
    {
        AscendC::DataCopyPadExtParams<T> featPad{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPadExtParams<int32_t> idxPad{false, 0, 0, 0};
        AscendC::DataCopyExtParams featParams{1, static_cast<uint32_t>(featDim_ * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyExtParams outParams{1, static_cast<uint32_t>(featDim_ * sizeof(T)), 0, 0, 0};
        for (uint32_t batchStart = startEdge_; batchStart < endEdge_; batchStart += batchSize_) {
            uint32_t actualBatch = batchSize_;
            if (batchStart + actualBatch > endEdge_) {
                actualBatch = endEdge_ - batchStart;
            }
            // 1. 批量加载索引到 UB（MTE2，TQue EnQue/DeQue 同步 MTE2→S）
            AscendC::LocalTensor<int32_t> idxLocal = idxQueue.AllocTensor<int32_t>();
            AscendC::DataCopyExtParams idxParams{1, static_cast<uint32_t>(actualBatch * sizeof(int32_t)), 0, 0, 0};
            AscendC::DataCopyPad(idxLocal, indexGm[batchStart], idxParams, idxPad);
            idxQueue.EnQue(idxLocal);
            idxLocal = idxQueue.DeQue<int32_t>();
            // 2. 逐边 Gather
            for (uint32_t i = 0; i < actualBatch; i++) {
                // MTE3→MTE2 同步（buffer 复用安全）：immediate Set→Wait
                // Set 插入 MTE3 队列（前一轮 MTE3 完成后设 flag）
                // Wait 插入 MTE2 队列（MTE2 等待 flag 后才复用 buffer）
                // MTE3 顺序处理，等待前一轮 MTE3 完成即确保所有更早 MTE3 也完成
                if (i > 0) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtM3M2_);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtM3M2_);
                }
                // MTE2: 加载当前边特征 GM→UB
                uint32_t idx_i = static_cast<uint32_t>(idxLocal.GetValue(i));
                AscendC::LocalTensor<T> feat = featQueue.AllocTensor<T>();
                AscendC::DataCopyPad(feat, featGmT[idx_i * featDim_], featParams, featPad);
                featQueue.EnQue(feat);
                AscendC::LocalTensor<T> featReady = featQueue.DeQue<T>();
                // MTE2→MTE3 同步（数据就绪）：immediate Set→Wait
                // Set 插入 MTE2 队列（MTE2 DataCopyPad 完成后设 flag）
                // Wait 插入 MTE3 队列（MTE3 等待 flag 后才读取 buffer）
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtM2M3_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtM2M3_);
                // MTE3: 存储当前边特征 UB→GM
                AscendC::DataCopyPad(outGmT[(batchStart + i) * featDim_], featReady, outParams);
                featQueue.FreeTensor(featReady);
            }
            idxQueue.FreeTensor(idxLocal);
        }
    }
private:
    AscendC::TPipe* pipe_;
    const __gm__ SddmmCopyLhsTilingData* tiling_;
    AscendC::GlobalTensor<float> featGm;
    AscendC::GlobalTensor<half> featHalfGm;
    AscendC::GlobalTensor<int32_t> indexGm;
    AscendC::GlobalTensor<float> outGm;
    AscendC::GlobalTensor<half> outHalfGm;
    // TQue<VECIN>: MTE2 搬运缓冲区（替代 TBuf<VECCALC>，满足 API 最佳实践）
    AscendC::TQue<AscendC::TPosition::VECIN, 1> idxQueue;     // 批量边索引
    AscendC::TQue<AscendC::TPosition::VECIN, 2> featQueue;    // 特征行（Double Buffer）
    // Event ID（MTE2↔MTE3 跨流水同步）
    int32_t evtM2M3_;  // MTE2→MTE3 同步
    int32_t evtM3M2_;  // MTE3→MTE2 同步（buffer 复用）
    uint32_t startEdge_ = 0;
    uint32_t endEdge_ = 0;
    uint32_t featDim_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t dtype_ = 0;
    uint32_t featDimAligned_ = 0;
};
extern "C" __global__ __aicore__ void sddmm_copy_lhs_kernel(GM_ADDR feat, GM_ADDR index,
                                                              GM_ADDR out, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    KernelSddmmCopyLhs op(&pipe);
    op.Init(feat, index, out, (__gm__ SddmmCopyLhsTilingData*)tiling);
    op.Process();
}
