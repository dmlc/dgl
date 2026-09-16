// ============================================================================
// Ascend C Kernel 实现 - edge_softmax (分段 softmax，沿入边维度归约)
// ============================================================================
//
// 对应 DESIGN.md §1.2 API 映射、§1.5 Buffer 规划、§2.4 伪代码
//
// 数学公式 (DESIGN.md §1.1):
//   Forward:  对每个目标节点 v 的入边段 [indptr[v], indptr[v+1]) 做 max-stable softmax
//             max_val[h] = max(efeat[i, h])
//             out[i, h] = exp(efeat[i, h] - max_val[h]) / sum(exp(efeat[i, h] - max_val[h]))
//   Backward: grad_efeat[i, h] = out[i, h] * (grad_out[i, h] - dot[h])
//             dot[h] = sum(grad_out[i, h] * out[i, h])
//
// 架构（DESIGN.md §2.1）:
//   - 段并行：每核处理连续目标节点区间，每段独立计算 softmax
//   - 纯向量计算：KERNEL_TYPE_AIV_ONLY
//   - 多核切分：blockDim = min(num_nodes, coreNum)
//
// 实现说明:
//   1. 使用 aclrtlaunch_* 启动模式（参考 spmm/sddmm）
//   2. 双模式：FullLoad（degree ≤ maxBatch，3-pass in-place）/ RowSplit（degree > maxBatch）
//   3. 双精度：FP32 原生计算；FP16 升精度到 FP32（Pattern::RA ReduceSum A2 只支持 float）
//   4. 双分支：AR（num_heads==1，Level 2 API + Adds/Muls）/ ARA（num_heads>1，Pattern::RA + Sub/Div）
//   5. degree=0 守卫：段为空，直接 continue（无数据可写）
//   6. V→MTE3 同步：Pass3 Div/Mul 写入 outQueue(VECOUT)，EnQue/DeQue 同步输出
// ============================================================================

#include "kernel_operator.h"
#include "edge_softmax_tiling.h"

// ============================================================================
// Kernel 类 - edge_softmax 计算逻辑
// ============================================================================
class KernelEdgeSoftmax {
public:
    __aicore__ inline KernelEdgeSoftmax(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR efeat, GM_ADDR indptr, GM_ADDR out,
                                GM_ADDR gradOut, GM_ADDR gradEfeat,
                                const __gm__ EdgeSoftmaxTilingData* tiling)
    {
        // 声明 AIV-only 任务类型（DESIGN.md §1.2）

        tiling_ = tiling;
        uint32_t blockIdx = AscendC::GetBlockIdx();

        dtype_ = tiling->dtype;
        mode_ = tiling->mode;
        numHeads_ = tiling->numHeads;
        isAR_ = (numHeads_ == 1);

        // DESIGN.md §2.1: 多核切分
        startNode_ = blockIdx * tiling->rowsPerCore;
        endNode_ = (blockIdx + 1) * tiling->rowsPerCore;
        if (endNode_ > tiling->numNodes) {
            endNode_ = tiling->numNodes;
        }

        // Global Tensor 设置
        if (dtype_ == DTYPE_FP32) {
            efeatGm.SetGlobalBuffer((__gm__ float*)efeat);
            outGm.SetGlobalBuffer((__gm__ float*)out);
            gradOutGm.SetGlobalBuffer((__gm__ float*)gradOut);
            gradEfeatGm.SetGlobalBuffer((__gm__ float*)gradEfeat);
        } else {
            efeatHalfGm.SetGlobalBuffer((__gm__ half*)efeat);
            outHalfGm.SetGlobalBuffer((__gm__ half*)out);
            gradOutHalfGm.SetGlobalBuffer((__gm__ half*)gradOut);
            gradEfeatHalfGm.SetGlobalBuffer((__gm__ half*)gradEfeat);
        }
        indptrGm.SetGlobalBuffer((__gm__ int32_t*)indptr);

        alignedColsF_ = tiling->numHeadsAlignedF;
        alignedColsH_ = tiling->numHeadsAlignedH;
        maxBatch_ = tiling->maxBatch;
        // FP16 路径 half buffer 是 [batch, alignedColsH] 布局（32B 对齐），
        // Cast half→float 连续转换整个 32B 块，故 FP16 float 计算空间也用 alignedColsH 列数。
        // FP32 路径直接用 alignedColsF。AR 模式 1D 不涉及列数。
        alignedCols_ = (dtype_ == DTYPE_FP16) ? alignedColsH_ : alignedColsF_;
        // AR 模式 1D 数据按 sizeof(float) 元素；ARA 模式 2D 按 alignedCols_ 元素
        elemPerRow_ = isAR_ ? 1 : alignedCols_;

        // ============================================================================
        // UB Buffer 初始化（DESIGN.md §1.5）
        // ============================================================================
        // indptrQueue: (rowsPerCore + 1) * sizeof(int32_t)，尾核按 actualRows 加载
        uint32_t actualRows = (endNode_ > startNode_) ? (endNode_ - startNode_) : 0;
        uint32_t indptrBufSize = (actualRows + 1) * sizeof(int32_t);
        if (indptrBufSize < 32) { indptrBufSize = 32; }  // 最小对齐
        pipe_->InitBuffer(indptrQueue, 1, indptrBufSize);

        // efeatQueue(VECIN, double buffer): maxBatch * elemPerRow * sizeof(float)
        //   FP32: 直接加载 FP32；FP16: 加载 FP16 到 efeatHalfQueue，Cast 到 efeatQueue(FP32)
        uint32_t efeatBufSize = maxBatch_ * elemPerRow_ * sizeof(float);
        pipe_->InitBuffer(efeatQueue, 2, efeatBufSize);

        // outQueue(VECOUT, double buffer): 输出同步
        pipe_->InitBuffer(outQueue, 2, efeatBufSize);

        if (dtype_ == DTYPE_FP16) {
            // FP16 输入/输出队列
            uint32_t halfElemPerRow = isAR_ ? 1 : alignedColsH_;
            uint32_t efeatHalfBufSize = maxBatch_ * halfElemPerRow * sizeof(half);
            pipe_->InitBuffer(efeatHalfQueue, 2, efeatHalfBufSize);
            pipe_->InitBuffer(outHalfQueue, 2, efeatHalfBufSize);
        }

        if (mode_ == MODE_BACKWARD) {
            // backward 额外需要 out 输入队列（forward 输出作为输入）
            pipe_->InitBuffer(outInQueue, 1, efeatBufSize);
            // backward gradOut 队列（复用 efeatQueue 语义，单独声明 gradOutQueue）
            pipe_->InitBuffer(gradOutQueue, 2, efeatBufSize);
            if (dtype_ == DTYPE_FP16) {
                uint32_t halfElemPerRow = isAR_ ? 1 : alignedColsH_;
                uint32_t halfBufSize = maxBatch_ * halfElemPerRow * sizeof(half);
                pipe_->InitBuffer(outInHalfQueue, 1, halfBufSize);
                pipe_->InitBuffer(gradOutHalfQueue, 2, halfBufSize);
            }
        }

        // VECCALC buffers
        uint32_t scalarBufSize = isAR_ ? 32 : (alignedCols_ * sizeof(float));
        pipe_->InitBuffer(maxValBuf, scalarBufSize);       // forward: max_val
        pipe_->InitBuffer(sumExpBuf, scalarBufSize);       // forward: sum_exp
        pipe_->InitBuffer(dotBuf, scalarBufSize);          // backward: dot
        pipe_->InitBuffer(chunkResultBuf, scalarBufSize);  // chunk 归约结果
        pipe_->InitBuffer(tmpBuf, TMP_BUF_SIZE);           // Reduce tmpBuf
    }

    __aicore__ inline void Process()
    {
        if (startNode_ >= endNode_) {
            return;
        }
        if (mode_ == MODE_FORWARD) {
            ProcessForward();
        } else {
            ProcessBackward();
        }
    }

private:
    // ============================================================================
    // 加载 indptr 到 UB
    // ============================================================================
    __aicore__ inline AscendC::LocalTensor<int32_t> LoadIndptr()
    {
        uint32_t actualRows = endNode_ - startNode_;
        AscendC::LocalTensor<int32_t> indptrLocal = indptrQueue.AllocTensor<int32_t>();
        AscendC::DataCopyExtParams indptrParams{1, static_cast<uint32_t>((actualRows + 1) * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> indptrPad{false, 0, 0, 0};
        AscendC::DataCopyPad(indptrLocal, indptrGm[startNode_], indptrParams, indptrPad);
        indptrQueue.EnQue(indptrLocal);
        return indptrQueue.DeQue<int32_t>();
    }

    // ============================================================================
    // Forward 主流程（DESIGN.md §2.4.1/2.4.2/2.4.3）
    // ============================================================================
    __aicore__ inline void ProcessForward()
    {
        AscendC::LocalTensor<int32_t> indptrLocal = LoadIndptr();

        for (uint32_t v = startNode_; v < endNode_; v++) {
            uint32_t rowStart = static_cast<uint32_t>(indptrLocal.GetValue(v - startNode_));
            uint32_t rowEnd = static_cast<uint32_t>(indptrLocal.GetValue(v - startNode_ + 1));
            uint32_t degree = rowEnd - rowStart;

            if (degree == 0) {
                continue;  // 孤立节点，段为空无需写
            }

            if (degree <= maxBatch_) {
                ForwardFullLoad(rowStart, degree);
            } else {
                ForwardRowSplit(rowStart, degree);
            }
        }
        indptrQueue.FreeTensor(indptrLocal);
    }

    // ============================================================================
    // Forward FullLoad（degree ≤ maxBatch，整段驻留 UB，3-pass in-place）
    // ============================================================================
    __aicore__ inline void ForwardFullLoad(uint32_t rowStart, uint32_t degree)
    {
        if (dtype_ == DTYPE_FP32) {
            if (isAR_) {
                ForwardFullLoadArFp32(rowStart, degree);
            } else {
                ForwardFullLoadAraFp32(rowStart, degree);
            }
        } else {
            if (isAR_) {
                ForwardFullLoadArFp16(rowStart, degree);
            } else {
                ForwardFullLoadAraFp16(rowStart, degree);
            }
        }
    }

    // ============================================================================
    // Forward RowSplit（degree > maxBatch，分批加载，3-pass 每遍重新加载）
    // ============================================================================
    __aicore__ inline void ForwardRowSplit(uint32_t rowStart, uint32_t degree)
    {
        if (dtype_ == DTYPE_FP32) {
            if (isAR_) {
                ForwardRowSplitArFp32(rowStart, degree);
            } else {
                ForwardRowSplitAraFp32(rowStart, degree);
            }
        } else {
            if (isAR_) {
                ForwardRowSplitArFp16(rowStart, degree);
            } else {
                ForwardRowSplitAraFp16(rowStart, degree);
            }
        }
    }

    // ============================================================================
    // Forward ARA FullLoad FP32（DESIGN.md §2.4.1）
    // ============================================================================
    __aicore__ inline void ForwardFullLoadAraFp32(uint32_t rowStart, uint32_t degree)
    {
        // 加载 efeat [degree, alignedColsF]
        AscendC::LocalTensor<float> efeatLocal = LoadEfeatFp32(rowStart, degree);

        AscendC::LocalTensor<float> maxValLocal = maxValBuf.Get<float>();
        AscendC::LocalTensor<float> sumExpLocal = sumExpBuf.Get<float>();
        AscendC::LocalTensor<float> chunkLocal = chunkResultBuf.Get<float>();
        AscendC::LocalTensor<uint8_t> tmpUint8 = tmpBuf.Get<uint8_t>();

        // Pass 1: 找 max
        AscendC::Duplicate<float>(maxValLocal, -__builtin_huge_valf(), alignedCols_);
        uint32_t srcShape[2] = {degree, alignedCols_};
        AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA>(chunkLocal, efeatLocal, tmpUint8, srcShape, true);
        AscendC::Max<float>(maxValLocal, maxValLocal, chunkLocal, alignedCols_);

        // Pass 2: exp + sum (in-place efeatLocal → exp(efeat-max))
        AscendC::Duplicate<float>(sumExpLocal, 0.0f, alignedCols_);
        BroadcastSubAra(efeatLocal, efeatLocal, maxValLocal, degree);
        AscendC::Exp<float>(efeatLocal, efeatLocal, degree * alignedCols_);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA>(chunkLocal, efeatLocal, tmpUint8, srcShape, true);
        AscendC::Add<float>(sumExpLocal, sumExpLocal, chunkLocal, alignedCols_);

        // Pass 3: normalize → 写入 outQueue(VECOUT) 实现 V→MTE3 同步
        AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        BroadcastDivAra(outLocal, efeatLocal, sumExpLocal, degree);
        OutputFp32(outLocal, rowStart, degree);

        efeatQueue.FreeTensor(efeatLocal);
    }

    // ============================================================================
    // Forward AR FullLoad FP32（DESIGN.md §2.4.2）
    // ============================================================================
    __aicore__ inline void ForwardFullLoadArFp32(uint32_t rowStart, uint32_t degree)
    {
        AscendC::LocalTensor<float> efeatLocal = LoadEfeatFp32(rowStart, degree);

        AscendC::LocalTensor<float> scalarLocal = maxValBuf.Get<float>();
        AscendC::LocalTensor<float> tmpFloat = tmpBuf.Get<float>();  // Level 2 tmp 类型为 LocalTensor<T>

        // Pass 1: max
        AscendC::ReduceMax<float>(scalarLocal, efeatLocal, tmpFloat, static_cast<int32_t>(degree), false);
        float maxVal = scalarLocal.GetValue(0);

        // Pass 2: exp + sum (in-place)
        AscendC::Adds<float>(efeatLocal, efeatLocal, -maxVal, degree);
        AscendC::Exp<float>(efeatLocal, efeatLocal, degree);
        AscendC::ReduceSum<float, true>(scalarLocal, efeatLocal, tmpFloat, static_cast<int32_t>(degree));
        float sumExp = scalarLocal.GetValue(0);

        // Pass 3: normalize → outQueue(VECOUT)
        AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        AscendC::Muls<float>(outLocal, efeatLocal, 1.0f / sumExp, degree);
        OutputFp32(outLocal, rowStart, degree);

        efeatQueue.FreeTensor(efeatLocal);
    }

    // ============================================================================
    // Forward ARA RowSplit FP32（DESIGN.md §2.4.3）
    // ============================================================================
    __aicore__ inline void ForwardRowSplitAraFp32(uint32_t rowStart, uint32_t degree)
    {
        AscendC::LocalTensor<float> maxValLocal = maxValBuf.Get<float>();
        AscendC::LocalTensor<float> sumExpLocal = sumExpBuf.Get<float>();
        AscendC::LocalTensor<float> chunkLocal = chunkResultBuf.Get<float>();
        AscendC::LocalTensor<uint8_t> tmpUint8 = tmpBuf.Get<uint8_t>();

        // Pass 1: 逐批找 max
        AscendC::Duplicate<float>(maxValLocal, -__builtin_huge_valf(), alignedCols_);
        for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
            uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
            AscendC::LocalTensor<float> efeatLocal = LoadEfeatFp32(bs, batch);
            uint32_t srcShape[2] = {batch, alignedCols_};
            AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA>(chunkLocal, efeatLocal, tmpUint8, srcShape, true);
            AscendC::Max<float>(maxValLocal, maxValLocal, chunkLocal, alignedCols_);
            efeatQueue.FreeTensor(efeatLocal);
        }

        // Pass 2: 逐批 exp + sum
        AscendC::Duplicate<float>(sumExpLocal, 0.0f, alignedCols_);
        for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
            uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
            AscendC::LocalTensor<float> efeatLocal = LoadEfeatFp32(bs, batch);
            BroadcastSubAra(efeatLocal, efeatLocal, maxValLocal, batch);
            AscendC::Exp<float>(efeatLocal, efeatLocal, batch * alignedCols_);
            uint32_t srcShape[2] = {batch, alignedCols_};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA>(chunkLocal, efeatLocal, tmpUint8, srcShape, true);
            AscendC::Add<float>(sumExpLocal, sumExpLocal, chunkLocal, alignedCols_);
            efeatQueue.FreeTensor(efeatLocal);
        }

        // Pass 3: 逐批 normalize → 输出
        for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
            uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
            AscendC::LocalTensor<float> efeatLocal = LoadEfeatFp32(bs, batch);
            BroadcastSubAra(efeatLocal, efeatLocal, maxValLocal, batch);
            AscendC::Exp<float>(efeatLocal, efeatLocal, batch * alignedCols_);
            AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
            BroadcastDivAra(outLocal, efeatLocal, sumExpLocal, batch);
            OutputFp32(outLocal, bs, batch);
            efeatQueue.FreeTensor(efeatLocal);
        }
    }

    // ============================================================================
    // Forward AR RowSplit FP32
    // ============================================================================
    __aicore__ inline void ForwardRowSplitArFp32(uint32_t rowStart, uint32_t degree)
    {
        AscendC::LocalTensor<float> scalarLocal = maxValBuf.Get<float>();
        AscendC::LocalTensor<float> tmpFloat = tmpBuf.Get<float>();
        AscendC::LocalTensor<float> maxValLocal = sumExpBuf.Get<float>();  // 复用 sumExpBuf 存 maxVal 标量

        // Pass 1: 逐批找 max（标量合并）
        AscendC::Duplicate<float>(maxValLocal, -__builtin_huge_valf(), 8);  // 最小对齐
        float globalMax = -__builtin_huge_valf();
        for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
            uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
            AscendC::LocalTensor<float> efeatLocal = LoadEfeatFp32(bs, batch);
            AscendC::ReduceMax<float>(scalarLocal, efeatLocal, tmpFloat, static_cast<int32_t>(batch), false);
            float chunkMax = scalarLocal.GetValue(0);
            if (chunkMax > globalMax) { globalMax = chunkMax; }
            efeatQueue.FreeTensor(efeatLocal);
        }

        // Pass 2: 逐批 exp + sum
        float globalSum = 0.0f;
        for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
            uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
            AscendC::LocalTensor<float> efeatLocal = LoadEfeatFp32(bs, batch);
            AscendC::Adds<float>(efeatLocal, efeatLocal, -globalMax, batch);
            AscendC::Exp<float>(efeatLocal, efeatLocal, batch);
            AscendC::ReduceSum<float, true>(scalarLocal, efeatLocal, tmpFloat, static_cast<int32_t>(batch));
            globalSum += scalarLocal.GetValue(0);
            efeatQueue.FreeTensor(efeatLocal);
        }

        // Pass 3: 逐批 normalize → 输出
        float invSum = 1.0f / globalSum;
        for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
            uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
            AscendC::LocalTensor<float> efeatLocal = LoadEfeatFp32(bs, batch);
            AscendC::Adds<float>(efeatLocal, efeatLocal, -globalMax, batch);
            AscendC::Exp<float>(efeatLocal, efeatLocal, batch);
            AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
            AscendC::Muls<float>(outLocal, efeatLocal, invSum, batch);
            OutputFp32(outLocal, bs, batch);
            efeatQueue.FreeTensor(efeatLocal);
        }
    }

    // ============================================================================
    // Backward 主流程（DESIGN.md §2.4.4）
    // ============================================================================
    __aicore__ inline void ProcessBackward()
    {
        AscendC::LocalTensor<int32_t> indptrLocal = LoadIndptr();

        for (uint32_t v = startNode_; v < endNode_; v++) {
            uint32_t rowStart = static_cast<uint32_t>(indptrLocal.GetValue(v - startNode_));
            uint32_t rowEnd = static_cast<uint32_t>(indptrLocal.GetValue(v - startNode_ + 1));
            uint32_t degree = rowEnd - rowStart;

            if (degree == 0) {
                continue;
            }

            if (degree <= maxBatch_) {
                BackwardFullLoad(rowStart, degree);
            } else {
                BackwardRowSplit(rowStart, degree);
            }
        }
        indptrQueue.FreeTensor(indptrLocal);
    }

    __aicore__ inline void BackwardFullLoad(uint32_t rowStart, uint32_t degree)
    {
        if (dtype_ == DTYPE_FP32) {
            if (isAR_) {
                BackwardFullLoadArFp32(rowStart, degree);
            } else {
                BackwardFullLoadAraFp32(rowStart, degree);
            }
        } else {
            if (isAR_) {
                BackwardFullLoadArFp16(rowStart, degree);
            } else {
                BackwardFullLoadAraFp16(rowStart, degree);
            }
        }
    }

    __aicore__ inline void BackwardRowSplit(uint32_t rowStart, uint32_t degree)
    {
        if (dtype_ == DTYPE_FP32) {
            if (isAR_) {
                BackwardRowSplitArFp32(rowStart, degree);
            } else {
                BackwardRowSplitAraFp32(rowStart, degree);
            }
        } else {
            if (isAR_) {
                BackwardRowSplitArFp16(rowStart, degree);
            } else {
                BackwardRowSplitAraFp16(rowStart, degree);
            }
        }
    }

    // ============================================================================
    // Backward ARA FullLoad FP32（DESIGN.md §2.4.4）
    // ============================================================================
    __aicore__ inline void BackwardFullLoadAraFp32(uint32_t rowStart, uint32_t degree)
    {
        AscendC::LocalTensor<float> dotLocal = dotBuf.Get<float>();

        // Pass 1: dot = sum(grad_out * out)
        // 使用逐行顺序累加（sequential Add）替代 Pattern::RA ReduceSum，
        // 匹配 numpy 的顺序求和（degree ≤ 128 时 numpy 使用 sequential sum），
        // 消除归约顺序差异导致的 dot 精度误差（修复 T14 精度问题）。
        AscendC::Duplicate<float>(dotLocal, 0.0f, alignedCols_);
        AscendC::LocalTensor<float> gradOutLocal = LoadGradOutFp32(rowStart, degree);
        for (uint32_t i = 0; i < degree; i++) {
            AscendC::Add<float>(dotLocal, dotLocal, gradOutLocal[i * alignedCols_], alignedCols_);
        }
        gradOutQueue.FreeTensor(gradOutLocal);

        // Pass 2: grad_efeat = sds - out * dot (DGL interface: gradOut=sds=out*grad_out)
        AscendC::LocalTensor<float> gradOutLocal2 = LoadGradOutFp32(rowStart, degree);
        AscendC::LocalTensor<float> outLocal2 = LoadOutInFp32(rowStart, degree);
        BroadcastMulAra(outLocal2, outLocal2, dotLocal, degree);
        AscendC::LocalTensor<float> gradEfeatLocal = outQueue.AllocTensor<float>();
        AscendC::Sub<float>(gradEfeatLocal, gradOutLocal2, outLocal2, degree * alignedCols_);
        OutputFp32Grad(gradEfeatLocal, rowStart, degree);
        gradOutQueue.FreeTensor(gradOutLocal2);
        outInQueue.FreeTensor(outLocal2);
    }

    // ============================================================================
    // Backward AR FullLoad FP32
    // ============================================================================
    __aicore__ inline void BackwardFullLoadArFp32(uint32_t rowStart, uint32_t degree)
    {
        AscendC::LocalTensor<float> scalarLocal = dotBuf.Get<float>();
        AscendC::LocalTensor<float> tmpFloat = tmpBuf.Get<float>();

        // Pass 1: dot = sum(grad_out * out)
        AscendC::LocalTensor<float> gradOutLocal = LoadGradOutFp32(rowStart, degree);
        AscendC::ReduceSum<float, true>(scalarLocal, gradOutLocal, tmpFloat, static_cast<int32_t>(degree));
        float dot = scalarLocal.GetValue(0);
        gradOutQueue.FreeTensor(gradOutLocal);

        // Pass 2: grad_efeat = sds - out * dot (DGL interface)
        AscendC::LocalTensor<float> gradOutLocal2 = LoadGradOutFp32(rowStart, degree);
        AscendC::LocalTensor<float> outLocal2 = LoadOutInFp32(rowStart, degree);
        AscendC::Muls<float>(outLocal2, outLocal2, dot, degree);
        AscendC::LocalTensor<float> gradEfeatLocal = outQueue.AllocTensor<float>();
        AscendC::Sub<float>(gradEfeatLocal, gradOutLocal2, outLocal2, degree);
        OutputFp32Grad(gradEfeatLocal, rowStart, degree);
        gradOutQueue.FreeTensor(gradOutLocal2);
        outInQueue.FreeTensor(outLocal2);
    }

    // ============================================================================
    // Backward ARA RowSplit FP32
    // ============================================================================
    __aicore__ inline void BackwardRowSplitAraFp32(uint32_t rowStart, uint32_t degree)
    {
        AscendC::LocalTensor<float> dotLocal = dotBuf.Get<float>();

        // Pass 1: dot = sum(grad_out * out)
        // 逐行顺序累加（sequential Add），匹配 numpy 顺序求和，消除归约顺序差异
        AscendC::Duplicate<float>(dotLocal, 0.0f, alignedCols_);
        for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
            uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
            AscendC::LocalTensor<float> gradOutLocal = LoadGradOutFp32(bs, batch);
            for (uint32_t i = 0; i < batch; i++) {
                AscendC::Add<float>(dotLocal, dotLocal, gradOutLocal[i * alignedCols_], alignedCols_);
            }
            gradOutQueue.FreeTensor(gradOutLocal);
        }

        // Pass 2: grad_efeat = out * (grad_out - dot) → 逐批输出
        for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
            uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
            AscendC::LocalTensor<float> gradOutLocal = LoadGradOutFp32(bs, batch);
            AscendC::LocalTensor<float> outLocal = LoadOutInFp32(bs, batch);
            BroadcastMulAra(outLocal, outLocal, dotLocal, batch);
            AscendC::LocalTensor<float> gradEfeatLocal = outQueue.AllocTensor<float>();
            AscendC::Sub<float>(gradEfeatLocal, gradOutLocal, outLocal, batch * alignedCols_);
            OutputFp32Grad(gradEfeatLocal, bs, batch);
            gradOutQueue.FreeTensor(gradOutLocal);
            outInQueue.FreeTensor(outLocal);
        }
    }

    // ============================================================================
    // Backward AR RowSplit FP32
    // ============================================================================
    __aicore__ inline void BackwardRowSplitArFp32(uint32_t rowStart, uint32_t degree)
    {
        AscendC::LocalTensor<float> scalarLocal = dotBuf.Get<float>();
        AscendC::LocalTensor<float> tmpFloat = tmpBuf.Get<float>();

        // Pass 1: dot
        float dot = 0.0f;
        for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
            uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
            AscendC::LocalTensor<float> gradOutLocal = LoadGradOutFp32(bs, batch);
            AscendC::ReduceSum<float, true>(scalarLocal, gradOutLocal, tmpFloat, static_cast<int32_t>(batch));
            dot += scalarLocal.GetValue(0);
            gradOutQueue.FreeTensor(gradOutLocal);
        }

        // Pass 2: grad_efeat = out * (grad_out - dot)
        for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
            uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
            AscendC::LocalTensor<float> gradOutLocal = LoadGradOutFp32(bs, batch);
            AscendC::LocalTensor<float> outLocal = LoadOutInFp32(bs, batch);
            AscendC::Muls<float>(outLocal, outLocal, dot, batch);
            AscendC::LocalTensor<float> gradEfeatLocal = outQueue.AllocTensor<float>();
            AscendC::Sub<float>(gradEfeatLocal, gradOutLocal, outLocal, batch);
            OutputFp32Grad(gradEfeatLocal, bs, batch);
            gradOutQueue.FreeTensor(gradOutLocal);
            outInQueue.FreeTensor(outLocal);
        }
    }

    // ============================================================================
    // FP16 分支 wrapper（load half → Cast FP32 → 调用 FP32 逻辑 → Cast half → 输出）
    // ============================================================================
    __aicore__ inline void ForwardFullLoadAraFp16(uint32_t rowStart, uint32_t degree)
    {
        ForwardFp16Core(rowStart, degree, true /*isFullLoad*/, true /*isAra*/);
    }
    __aicore__ inline void ForwardFullLoadArFp16(uint32_t rowStart, uint32_t degree)
    {
        ForwardFp16Core(rowStart, degree, true /*isFullLoad*/, false /*isAra*/);
    }
    __aicore__ inline void ForwardRowSplitAraFp16(uint32_t rowStart, uint32_t degree)
    {
        ForwardFp16RowSplit(rowStart, degree, true /*isAra*/);
    }
    __aicore__ inline void ForwardRowSplitArFp16(uint32_t rowStart, uint32_t degree)
    {
        ForwardFp16RowSplit(rowStart, degree, false /*isAra*/);
    }

    // FP16 FullLoad: 整段加载 half → Cast FP32 → FP32 FullLoad 逻辑 → Cast half 输出
    __aicore__ inline void ForwardFp16Core(uint32_t rowStart, uint32_t degree, bool isFullLoad, bool isAra)
    {
        // 加载 FP16 → Cast FP32 到 efeatQueue
        AscendC::LocalTensor<float> efeatLocal = LoadEfeatFp16ToFp32(rowStart, degree);

        if (isAra) {
            AscendC::LocalTensor<float> maxValLocal = maxValBuf.Get<float>();
            AscendC::LocalTensor<float> sumExpLocal = sumExpBuf.Get<float>();
            AscendC::LocalTensor<float> chunkLocal = chunkResultBuf.Get<float>();
            AscendC::LocalTensor<uint8_t> tmpUint8 = tmpBuf.Get<uint8_t>();

            AscendC::Duplicate<float>(maxValLocal, -__builtin_huge_valf(), alignedCols_);
            uint32_t srcShape[2] = {degree, alignedCols_};
            AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA>(chunkLocal, efeatLocal, tmpUint8, srcShape, true);
            AscendC::Max<float>(maxValLocal, maxValLocal, chunkLocal, alignedCols_);

            AscendC::Duplicate<float>(sumExpLocal, 0.0f, alignedCols_);
            BroadcastSubAra(efeatLocal, efeatLocal, maxValLocal, degree);
            AscendC::Exp<float>(efeatLocal, efeatLocal, degree * alignedCols_);
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA>(chunkLocal, efeatLocal, tmpUint8, srcShape, true);
            AscendC::Add<float>(sumExpLocal, sumExpLocal, chunkLocal, alignedCols_);

            AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
            BroadcastDivAra(outLocal, efeatLocal, sumExpLocal, degree);
            OutputFp16(outLocal, rowStart, degree);
        } else {
            AscendC::LocalTensor<float> scalarLocal = maxValBuf.Get<float>();
            AscendC::LocalTensor<float> tmpFloat = tmpBuf.Get<float>();
            AscendC::ReduceMax<float>(scalarLocal, efeatLocal, tmpFloat, static_cast<int32_t>(degree), false);
            float maxVal = scalarLocal.GetValue(0);
            AscendC::Adds<float>(efeatLocal, efeatLocal, -maxVal, degree);
            AscendC::Exp<float>(efeatLocal, efeatLocal, degree);
            AscendC::ReduceSum<float, true>(scalarLocal, efeatLocal, tmpFloat, static_cast<int32_t>(degree));
            float sumExp = scalarLocal.GetValue(0);
            AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
            AscendC::Muls<float>(outLocal, efeatLocal, 1.0f / sumExp, degree);
            OutputFp16(outLocal, rowStart, degree);
        }
        efeatQueue.FreeTensor(efeatLocal);
    }

    // FP16 RowSplit: 逐批加载 half → Cast FP32 → FP32 RowSplit 逻辑 → Cast half 输出
    __aicore__ inline void ForwardFp16RowSplit(uint32_t rowStart, uint32_t degree, bool isAra)
    {
        if (isAra) {
            AscendC::LocalTensor<float> maxValLocal = maxValBuf.Get<float>();
            AscendC::LocalTensor<float> sumExpLocal = sumExpBuf.Get<float>();
            AscendC::LocalTensor<float> chunkLocal = chunkResultBuf.Get<float>();
            AscendC::LocalTensor<uint8_t> tmpUint8 = tmpBuf.Get<uint8_t>();

            AscendC::Duplicate<float>(maxValLocal, -__builtin_huge_valf(), alignedCols_);
            for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
                uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
                AscendC::LocalTensor<float> efeatLocal = LoadEfeatFp16ToFp32(bs, batch);
                uint32_t srcShape[2] = {batch, alignedCols_};
                AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA>(chunkLocal, efeatLocal, tmpUint8, srcShape, true);
                AscendC::Max<float>(maxValLocal, maxValLocal, chunkLocal, alignedCols_);
                efeatQueue.FreeTensor(efeatLocal);
            }
            AscendC::Duplicate<float>(sumExpLocal, 0.0f, alignedCols_);
            for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
                uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
                AscendC::LocalTensor<float> efeatLocal = LoadEfeatFp16ToFp32(bs, batch);
                BroadcastSubAra(efeatLocal, efeatLocal, maxValLocal, batch);
                AscendC::Exp<float>(efeatLocal, efeatLocal, batch * alignedCols_);
                uint32_t srcShape[2] = {batch, alignedCols_};
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA>(chunkLocal, efeatLocal, tmpUint8, srcShape, true);
                AscendC::Add<float>(sumExpLocal, sumExpLocal, chunkLocal, alignedCols_);
                efeatQueue.FreeTensor(efeatLocal);
            }
            for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
                uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
                AscendC::LocalTensor<float> efeatLocal = LoadEfeatFp16ToFp32(bs, batch);
                BroadcastSubAra(efeatLocal, efeatLocal, maxValLocal, batch);
                AscendC::Exp<float>(efeatLocal, efeatLocal, batch * alignedCols_);
                AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
                BroadcastDivAra(outLocal, efeatLocal, sumExpLocal, batch);
                OutputFp16(outLocal, bs, batch);
                efeatQueue.FreeTensor(efeatLocal);
            }
        } else {
            AscendC::LocalTensor<float> scalarLocal = maxValBuf.Get<float>();
            AscendC::LocalTensor<float> tmpFloat = tmpBuf.Get<float>();
            float globalMax = -__builtin_huge_valf();
            for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
                uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
                AscendC::LocalTensor<float> efeatLocal = LoadEfeatFp16ToFp32(bs, batch);
                AscendC::ReduceMax<float>(scalarLocal, efeatLocal, tmpFloat, static_cast<int32_t>(batch), false);
                float chunkMax = scalarLocal.GetValue(0);
                if (chunkMax > globalMax) { globalMax = chunkMax; }
                efeatQueue.FreeTensor(efeatLocal);
            }
            float globalSum = 0.0f;
            for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
                uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
                AscendC::LocalTensor<float> efeatLocal = LoadEfeatFp16ToFp32(bs, batch);
                AscendC::Adds<float>(efeatLocal, efeatLocal, -globalMax, batch);
                AscendC::Exp<float>(efeatLocal, efeatLocal, batch);
                AscendC::ReduceSum<float, true>(scalarLocal, efeatLocal, tmpFloat, static_cast<int32_t>(batch));
                globalSum += scalarLocal.GetValue(0);
                efeatQueue.FreeTensor(efeatLocal);
            }
            float invSum = 1.0f / globalSum;
            for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
                uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
                AscendC::LocalTensor<float> efeatLocal = LoadEfeatFp16ToFp32(bs, batch);
                AscendC::Adds<float>(efeatLocal, efeatLocal, -globalMax, batch);
                AscendC::Exp<float>(efeatLocal, efeatLocal, batch);
                AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
                AscendC::Muls<float>(outLocal, efeatLocal, invSum, batch);
                OutputFp16(outLocal, bs, batch);
                efeatQueue.FreeTensor(efeatLocal);
            }
        }
    }

    // FP16 Backward wrappers
    __aicore__ inline void BackwardFullLoadAraFp16(uint32_t rowStart, uint32_t degree)
    {
        BackwardFp16Core(rowStart, degree, true /*isFullLoad*/, true /*isAra*/);
    }
    __aicore__ inline void BackwardFullLoadArFp16(uint32_t rowStart, uint32_t degree)
    {
        BackwardFp16Core(rowStart, degree, true /*isFullLoad*/, false /*isAra*/);
    }
    __aicore__ inline void BackwardRowSplitAraFp16(uint32_t rowStart, uint32_t degree)
    {
        BackwardFp16RowSplit(rowStart, degree, true /*isAra*/);
    }
    __aicore__ inline void BackwardRowSplitArFp16(uint32_t rowStart, uint32_t degree)
    {
        BackwardFp16RowSplit(rowStart, degree, false /*isAra*/);
    }

    __aicore__ inline void BackwardFp16Core(uint32_t rowStart, uint32_t degree, bool isFullLoad, bool isAra)
    {
        if (isAra) {
            AscendC::LocalTensor<float> dotLocal = dotBuf.Get<float>();

            // 逐行顺序累加（sequential Add），匹配 numpy 顺序求和，消除归约顺序差异
            AscendC::Duplicate<float>(dotLocal, 0.0f, alignedCols_);
            AscendC::LocalTensor<float> gradOutLocal = LoadGradOutFp16ToFp32(rowStart, degree);
            for (uint32_t i = 0; i < degree; i++) {
                AscendC::Add<float>(dotLocal, dotLocal, gradOutLocal[i * alignedCols_], alignedCols_);
            }
            gradOutQueue.FreeTensor(gradOutLocal);

            AscendC::LocalTensor<float> gradOutLocal2 = LoadGradOutFp16ToFp32(rowStart, degree);
            AscendC::LocalTensor<float> outLocal2 = LoadOutInFp16ToFp32(rowStart, degree);
            BroadcastMulAra(outLocal2, outLocal2, dotLocal, degree);
            AscendC::LocalTensor<float> gradEfeatLocal = outQueue.AllocTensor<float>();
            AscendC::Sub<float>(gradEfeatLocal, gradOutLocal2, outLocal2, degree * alignedCols_);
            OutputFp16Grad(gradEfeatLocal, rowStart, degree);
            gradOutQueue.FreeTensor(gradOutLocal2);
            outInQueue.FreeTensor(outLocal2);
        } else {
            AscendC::LocalTensor<float> scalarLocal = dotBuf.Get<float>();
            AscendC::LocalTensor<float> tmpFloat = tmpBuf.Get<float>();
            AscendC::LocalTensor<float> gradOutLocal = LoadGradOutFp16ToFp32(rowStart, degree);
            AscendC::ReduceSum<float, true>(scalarLocal, gradOutLocal, tmpFloat, static_cast<int32_t>(degree));
            float dot = scalarLocal.GetValue(0);
            gradOutQueue.FreeTensor(gradOutLocal);

            AscendC::LocalTensor<float> gradOutLocal2 = LoadGradOutFp16ToFp32(rowStart, degree);
            AscendC::LocalTensor<float> outLocal2 = LoadOutInFp16ToFp32(rowStart, degree);
            AscendC::Muls<float>(outLocal2, outLocal2, dot, degree);
            AscendC::LocalTensor<float> gradEfeatLocal = outQueue.AllocTensor<float>();
            AscendC::Sub<float>(gradEfeatLocal, gradOutLocal2, outLocal2, degree);
            OutputFp16Grad(gradEfeatLocal, rowStart, degree);
            gradOutQueue.FreeTensor(gradOutLocal2);
            outInQueue.FreeTensor(outLocal2);
        }
    }

    __aicore__ inline void BackwardFp16RowSplit(uint32_t rowStart, uint32_t degree, bool isAra)
    {
        if (isAra) {
            AscendC::LocalTensor<float> dotLocal = dotBuf.Get<float>();
            // 逐行顺序累加（sequential Add），匹配 numpy 顺序求和，消除归约顺序差异
            AscendC::Duplicate<float>(dotLocal, 0.0f, alignedCols_);
            for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
                uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
                AscendC::LocalTensor<float> gradOutLocal = LoadGradOutFp16ToFp32(bs, batch);
                for (uint32_t i = 0; i < batch; i++) {
                    AscendC::Add<float>(dotLocal, dotLocal, gradOutLocal[i * alignedCols_], alignedCols_);
                }
                gradOutQueue.FreeTensor(gradOutLocal);
            }
            for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
                uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
                AscendC::LocalTensor<float> gradOutLocal = LoadGradOutFp16ToFp32(bs, batch);
                AscendC::LocalTensor<float> outLocal = LoadOutInFp16ToFp32(bs, batch);
                BroadcastMulAra(outLocal, outLocal, dotLocal, batch);
                AscendC::LocalTensor<float> gradEfeatLocal = outQueue.AllocTensor<float>();
                AscendC::Sub<float>(gradEfeatLocal, gradOutLocal, outLocal, batch * alignedCols_);
                OutputFp16Grad(gradEfeatLocal, bs, batch);
                gradOutQueue.FreeTensor(gradOutLocal);
                outInQueue.FreeTensor(outLocal);
            }
        } else {
            AscendC::LocalTensor<float> scalarLocal = dotBuf.Get<float>();
            AscendC::LocalTensor<float> tmpFloat = tmpBuf.Get<float>();
            float dot = 0.0f;
            for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
                uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
                AscendC::LocalTensor<float> gradOutLocal = LoadGradOutFp16ToFp32(bs, batch);
                AscendC::ReduceSum<float, true>(scalarLocal, gradOutLocal, tmpFloat, static_cast<int32_t>(batch));
                dot += scalarLocal.GetValue(0);
                gradOutQueue.FreeTensor(gradOutLocal);
            }
            for (uint32_t bs = rowStart; bs < rowStart + degree; bs += maxBatch_) {
                uint32_t batch = (bs + maxBatch_ < rowStart + degree) ? maxBatch_ : (rowStart + degree - bs);
                AscendC::LocalTensor<float> gradOutLocal = LoadGradOutFp16ToFp32(bs, batch);
                AscendC::LocalTensor<float> outLocal = LoadOutInFp16ToFp32(bs, batch);
                AscendC::Muls<float>(outLocal, outLocal, dot, batch);
                AscendC::LocalTensor<float> gradEfeatLocal = outQueue.AllocTensor<float>();
                AscendC::Sub<float>(gradEfeatLocal, gradOutLocal, outLocal, batch);
                OutputFp16Grad(gradEfeatLocal, bs, batch);
                gradOutQueue.FreeTensor(gradOutLocal);
                outInQueue.FreeTensor(outLocal);
            }
        }
    }

    // ============================================================================
    // 数据加载辅助函数
    // ============================================================================
    // FP32 加载 efeat [batch, elemPerRow] 到 efeatQueue(VECIN)
    __aicore__ inline AscendC::LocalTensor<float> LoadEfeatFp32(uint32_t rowStart, uint32_t batch)
    {
        AscendC::LocalTensor<float> efeatLocal = efeatQueue.AllocTensor<float>();
        if (isAR_) {
            // AR: 1D [batch] 个 float
            AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(batch * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(efeatLocal, efeatGm[rowStart], params, pad);
        } else {
            // ARA: 2D [batch, numHeads] 行主序，blockLen=numHeads*4
            AscendC::DataCopyExtParams params{static_cast<uint16_t>(batch),
                static_cast<uint32_t>(numHeads_ * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(efeatLocal, efeatGm[rowStart * numHeads_], params, pad);
        }
        efeatQueue.EnQue(efeatLocal);
        return efeatQueue.DeQue<float>();
    }

    // FP16 加载 efeat half → Cast FP32 到 efeatQueue
    __aicore__ inline AscendC::LocalTensor<float> LoadEfeatFp16ToFp32(uint32_t rowStart, uint32_t batch)
    {
        AscendC::LocalTensor<half> efeatHalf = efeatHalfQueue.AllocTensor<half>();
        if (isAR_) {
            AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(batch * sizeof(half)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<half> pad{false, 0, 0, half(0)};
            AscendC::DataCopyPad(efeatHalf, efeatHalfGm[rowStart], params, pad);
        } else {
            AscendC::DataCopyExtParams params{static_cast<uint16_t>(batch),
                static_cast<uint32_t>(numHeads_ * sizeof(half)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<half> pad{false, 0, 0, half(0)};
            AscendC::DataCopyPad(efeatHalf, efeatHalfGm[rowStart * numHeads_], params, pad);
        }
        efeatHalfQueue.EnQue(efeatHalf);
        efeatHalf = efeatHalfQueue.DeQue<half>();

        AscendC::LocalTensor<float> efeatLocal = efeatQueue.AllocTensor<float>();
        uint32_t castCount = isAR_ ? batch : (batch * alignedCols_);
        AscendC::Cast<float, half>(efeatLocal, efeatHalf, AscendC::RoundMode::CAST_NONE, castCount);
        efeatHalfQueue.FreeTensor(efeatHalf);
        return efeatLocal;  // VECCALC/VECIN FP32，无需 EnQue（Cast 是 V 流水，数据已就绪）
    }

    // FP32 加载 gradOut
    __aicore__ inline AscendC::LocalTensor<float> LoadGradOutFp32(uint32_t rowStart, uint32_t batch)
    {
        AscendC::LocalTensor<float> gradOutLocal = gradOutQueue.AllocTensor<float>();
        if (isAR_) {
            AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(batch * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(gradOutLocal, gradOutGm[rowStart], params, pad);
        } else {
            AscendC::DataCopyExtParams params{static_cast<uint16_t>(batch),
                static_cast<uint32_t>(numHeads_ * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(gradOutLocal, gradOutGm[rowStart * numHeads_], params, pad);
        }
        gradOutQueue.EnQue(gradOutLocal);
        return gradOutQueue.DeQue<float>();
    }

    __aicore__ inline AscendC::LocalTensor<float> LoadGradOutFp16ToFp32(uint32_t rowStart, uint32_t batch)
    {
        AscendC::LocalTensor<half> gradOutHalf = gradOutHalfQueue.AllocTensor<half>();
        if (isAR_) {
            AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(batch * sizeof(half)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<half> pad{false, 0, 0, half(0)};
            AscendC::DataCopyPad(gradOutHalf, gradOutHalfGm[rowStart], params, pad);
        } else {
            AscendC::DataCopyExtParams params{static_cast<uint16_t>(batch),
                static_cast<uint32_t>(numHeads_ * sizeof(half)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<half> pad{false, 0, 0, half(0)};
            AscendC::DataCopyPad(gradOutHalf, gradOutHalfGm[rowStart * numHeads_], params, pad);
        }
        gradOutHalfQueue.EnQue(gradOutHalf);
        gradOutHalf = gradOutHalfQueue.DeQue<half>();

        AscendC::LocalTensor<float> gradOutLocal = gradOutQueue.AllocTensor<float>();
        uint32_t castCount = isAR_ ? batch : (batch * alignedCols_);
        AscendC::Cast<float, half>(gradOutLocal, gradOutHalf, AscendC::RoundMode::CAST_NONE, castCount);
        gradOutHalfQueue.FreeTensor(gradOutHalf);
        return gradOutLocal;
    }

    // FP32 加载 out (backward 输入)
    __aicore__ inline AscendC::LocalTensor<float> LoadOutInFp32(uint32_t rowStart, uint32_t batch)
    {
        AscendC::LocalTensor<float> outLocal = outInQueue.AllocTensor<float>();
        if (isAR_) {
            AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(batch * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(outLocal, outGm[rowStart], params, pad);
        } else {
            AscendC::DataCopyExtParams params{static_cast<uint16_t>(batch),
                static_cast<uint32_t>(numHeads_ * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(outLocal, outGm[rowStart * numHeads_], params, pad);
        }
        outInQueue.EnQue(outLocal);
        return outInQueue.DeQue<float>();
    }

    __aicore__ inline AscendC::LocalTensor<float> LoadOutInFp16ToFp32(uint32_t rowStart, uint32_t batch)
    {
        AscendC::LocalTensor<half> outHalf = outInHalfQueue.AllocTensor<half>();
        if (isAR_) {
            AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(batch * sizeof(half)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<half> pad{false, 0, 0, half(0)};
            AscendC::DataCopyPad(outHalf, outHalfGm[rowStart], params, pad);
        } else {
            AscendC::DataCopyExtParams params{static_cast<uint16_t>(batch),
                static_cast<uint32_t>(numHeads_ * sizeof(half)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<half> pad{false, 0, 0, half(0)};
            AscendC::DataCopyPad(outHalf, outHalfGm[rowStart * numHeads_], params, pad);
        }
        outInHalfQueue.EnQue(outHalf);
        outHalf = outInHalfQueue.DeQue<half>();

        AscendC::LocalTensor<float> outLocal = outInQueue.AllocTensor<float>();
        uint32_t castCount = isAR_ ? batch : (batch * alignedCols_);
        AscendC::Cast<float, half>(outLocal, outHalf, AscendC::RoundMode::CAST_NONE, castCount);
        outInHalfQueue.FreeTensor(outHalf);
        return outLocal;
    }

    // ============================================================================
    // ARA 广播辅助（BinaryRepeatParams src1RepStride=0）
    // ============================================================================
    __aicore__ inline void BroadcastSubAra(AscendC::LocalTensor<float>& dst,
                                            AscendC::LocalTensor<float>& src0,
                                            AscendC::LocalTensor<float>& src1, uint32_t batch)
    {
        uint64_t mask = numHeads_;
        uint8_t repTime = static_cast<uint8_t>(batch);
        AscendC::Sub<float>(dst, src0, src1, mask, repTime,
            {1, 1, 1, static_cast<uint8_t>(alignedCols_ / 8),
             static_cast<uint8_t>(alignedCols_ / 8), 0});
    }

    __aicore__ inline void BroadcastDivAra(AscendC::LocalTensor<float>& dst,
                                            AscendC::LocalTensor<float>& src0,
                                            AscendC::LocalTensor<float>& src1, uint32_t batch)
    {
        uint64_t mask = numHeads_;
        uint8_t repTime = static_cast<uint8_t>(batch);
        AscendC::Div<float>(dst, src0, src1, mask, repTime,
            {1, 1, 1, static_cast<uint8_t>(alignedCols_ / 8),
             static_cast<uint8_t>(alignedCols_ / 8), 0});
    }
    __aicore__ inline void BroadcastMulAra(AscendC::LocalTensor<float>& dst,
                                            AscendC::LocalTensor<float>& src0,
                                            AscendC::LocalTensor<float>& src1, uint32_t batch)
    {
        uint64_t mask = numHeads_;
        uint8_t repTime = static_cast<uint8_t>(batch);
        AscendC::Mul<float>(dst, src0, src1, mask, repTime,
            {1, 1, 1, static_cast<uint8_t>(alignedCols_ / 8),
             static_cast<uint8_t>(alignedCols_ / 8), 0});
    }

    // ============================================================================
    // 输出辅助（VECOUT EnQue/DeQue → DataCopyPad → FreeTensor）
    // ============================================================================
    __aicore__ inline void OutputFp32(AscendC::LocalTensor<float>& outLocal, uint32_t rowStart, uint32_t batch)
    {
        outQueue.EnQue<float>(outLocal);
        AscendC::LocalTensor<float> outResult = outQueue.DeQue<float>();
        if (isAR_) {
            AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(batch * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(outGm[rowStart], outResult, params);
        } else {
            AscendC::DataCopyExtParams params{static_cast<uint16_t>(batch),
                static_cast<uint32_t>(numHeads_ * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(outGm[rowStart * numHeads_], outResult, params);
        }
        outQueue.FreeTensor(outResult);
    }

    __aicore__ inline void OutputFp32Grad(AscendC::LocalTensor<float>& gradLocal, uint32_t rowStart, uint32_t batch)
    {
        outQueue.EnQue<float>(gradLocal);
        AscendC::LocalTensor<float> gradResult = outQueue.DeQue<float>();
        if (isAR_) {
            AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(batch * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(gradEfeatGm[rowStart], gradResult, params);
        } else {
            AscendC::DataCopyExtParams params{static_cast<uint16_t>(batch),
                static_cast<uint32_t>(numHeads_ * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(gradEfeatGm[rowStart * numHeads_], gradResult, params);
        }
        outQueue.FreeTensor(gradResult);
    }

    __aicore__ inline void OutputFp16(AscendC::LocalTensor<float>& outF32, uint32_t rowStart, uint32_t batch)
    {
        // Cast FP32 → FP16 到 outHalfQueue(VECOUT)
        AscendC::LocalTensor<half> outHalf = outHalfQueue.AllocTensor<half>();
        uint32_t castCount = isAR_ ? batch : (batch * alignedCols_);
        AscendC::Cast<half, float>(outHalf, outF32, AscendC::RoundMode::CAST_ROUND, castCount);
        // 先释放 FP32 outQueue
        outQueue.FreeTensor(outF32);
        // outHalfQueue EnQue → DataCopyPad 输出
        outHalfQueue.EnQue<half>(outHalf);
        AscendC::LocalTensor<half> outResult = outHalfQueue.DeQue<half>();
        if (isAR_) {
            AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(batch * sizeof(half)), 0, 0, 0};
            AscendC::DataCopyPad(outHalfGm[rowStart], outResult, params);
        } else {
            AscendC::DataCopyExtParams params{static_cast<uint16_t>(batch),
                static_cast<uint32_t>(numHeads_ * sizeof(half)), 0, 0, 0};
            AscendC::DataCopyPad(outHalfGm[rowStart * numHeads_], outResult, params);
        }
        outHalfQueue.FreeTensor(outResult);
    }

    __aicore__ inline void OutputFp16Grad(AscendC::LocalTensor<float>& gradF32, uint32_t rowStart, uint32_t batch)
    {
        AscendC::LocalTensor<half> gradHalf = outHalfQueue.AllocTensor<half>();
        uint32_t castCount = isAR_ ? batch : (batch * alignedCols_);
        AscendC::Cast<half, float>(gradHalf, gradF32, AscendC::RoundMode::CAST_ROUND, castCount);
        outQueue.FreeTensor(gradF32);
        outHalfQueue.EnQue<half>(gradHalf);
        AscendC::LocalTensor<half> gradResult = outHalfQueue.DeQue<half>();
        if (isAR_) {
            AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(batch * sizeof(half)), 0, 0, 0};
            AscendC::DataCopyPad(gradEfeatHalfGm[rowStart], gradResult, params);
        } else {
            AscendC::DataCopyExtParams params{static_cast<uint16_t>(batch),
                static_cast<uint32_t>(numHeads_ * sizeof(half)), 0, 0, 0};
            AscendC::DataCopyPad(gradEfeatHalfGm[rowStart * numHeads_], gradResult, params);
        }
        outHalfQueue.FreeTensor(gradResult);
    }

private:
    AscendC::TPipe* pipe_;
    const __gm__ EdgeSoftmaxTilingData* tiling_;

    // Global Tensor
    AscendC::GlobalTensor<float> efeatGm;
    AscendC::GlobalTensor<float> outGm;
    AscendC::GlobalTensor<float> gradOutGm;
    AscendC::GlobalTensor<float> gradEfeatGm;
    AscendC::GlobalTensor<half> efeatHalfGm;
    AscendC::GlobalTensor<half> outHalfGm;
    AscendC::GlobalTensor<half> gradOutHalfGm;
    AscendC::GlobalTensor<half> gradEfeatHalfGm;
    AscendC::GlobalTensor<int32_t> indptrGm;

    // UB Queue
    AscendC::TQue<AscendC::TPosition::VECIN, 2> efeatQueue;       // efeat 输入（FP32 计算空间）
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueue;        // 输出同步
    AscendC::TQue<AscendC::TPosition::VECIN, 1> indptrQueue;      // indptr 行指针
    // FP16 专用
    AscendC::TQue<AscendC::TPosition::VECIN, 2> efeatHalfQueue;   // FP16 输入
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outHalfQueue;    // FP16 输出
    // Backward 专用
    AscendC::TQue<AscendC::TPosition::VECIN, 2> gradOutQueue;     // gradOut 输入
    AscendC::TQue<AscendC::TPosition::VECIN, 1> outInQueue;       // out 输入（forward 输出）
    AscendC::TQue<AscendC::TPosition::VECIN, 2> gradOutHalfQueue; // FP16 gradOut
    AscendC::TQue<AscendC::TPosition::VECIN, 1> outInHalfQueue;   // FP16 out 输入

    // VECCALC buffers
    AscendC::TBuf<AscendC::TPosition::VECCALC> maxValBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumExpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dotBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> chunkResultBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;

    uint32_t startNode_ = 0;
    uint32_t endNode_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t alignedColsF_ = 0;
    uint32_t alignedColsH_ = 0;
    uint32_t alignedCols_ = 0;   // ARA 计算实际列数：FP32=alignedColsF, FP16=alignedColsH
    uint32_t maxBatch_ = 0;
    uint32_t elemPerRow_ = 0;
    uint32_t dtype_ = 0;
    uint32_t mode_ = 0;
    bool isAR_ = false;
};

// ============================================================================
// 核函数入口
// ============================================================================
extern "C" __global__ __aicore__ void edge_softmax_kernel(GM_ADDR efeat, GM_ADDR indptr,
                                                           GM_ADDR out, GM_ADDR gradOut,
                                                           GM_ADDR gradEfeat, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    KernelEdgeSoftmax op(&pipe);
    op.Init(efeat, indptr, out, gradOut, gradEfeat, (__gm__ EdgeSoftmaxTilingData*)tiling);
    op.Process();
}
