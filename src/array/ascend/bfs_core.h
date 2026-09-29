// =============================================================================
// BFSNodesFrontiers / BFSEdgesFrontiers 共享 kernel 核心（模板 <T, IS_EDGE>）
// =============================================================================

#include <cstdint>
#include "kernel_operator.h"
#include "kernel_operator_cache_intf.h"
#include "bfs_ws_layout.h"

namespace bfscore {

// 平衡分区（ceil 形式，保证 F<C 时仍有核分到工作）
__aicore__ inline int64_t BsPartLo(int64_t i, int64_t n, int32_t c) { return (i * n + c - 1) / c; }
__aicore__ inline int64_t BsPartHi(int64_t i, int64_t n, int32_t c) { return ((i + 1) * n + c - 1) / c; }                // pSrc 特值：frag 在 source 槽（parity0 末位）

__aicore__ inline int64_t BsMinI64(int64_t a, int64_t b) { return a < b ? a : b; }
__aicore__ inline int64_t BsAlign(int64_t v, int64_t a) { return (v + a - 1) / a * a; }
__aicore__ inline int32_t BsDivUp(int32_t a, int32_t b) { return (a + b - 1) / b; }

// ---------------- device 侧算式（__aicore__；与 host 版 Hs* 严格同式） ----------------
__aicore__ inline int32_t BfsRowStrideInt32(int32_t cores)
{
    return (int32_t)BsAlign((int64_t)cores * 4, 32) / 4;
}

__aicore__ inline int32_t BfsRowStrideJ(int32_t numChunksMax)
{
    return (int32_t)BsAlign((int64_t)numChunksMax * 4, 32) / 4;
}

__aicore__ inline int32_t BsChunkElems(int64_t elemSize) { return (elemSize == 4) ? 1024 : 512; }

__aicore__ inline BfsWs BfsWorkspaceLayout(int64_t numRows, int64_t nnz, int32_t cores,
                                           int32_t numChunksMax, int64_t elemSize)
{
    BfsWs w;
    int32_t rs = BfsRowStrideInt32(cores);
    int64_t off = 0;
    w.visTag = off;  off += BsAlign(numRows * 4, WS_ALIGN);
    w.syncGm = off;  off += BsAlign((int64_t)cores * 32, WS_ALIGN);
    w.errflag = off; off += BsAlign((int64_t)cores * 32, WS_ALIGN);
    w.segByCore = off; off += BsAlign((int64_t)cores * 32, WS_ALIGN);
    w.cntDom = off;  off += BsAlign((int64_t)numChunksMax * rs * 4, WS_ALIGN);
    w.cntConfD = off; off += BsAlign((int64_t)cores * BfsRowStrideJ(numChunksMax) * 4, WS_ALIGN);
    w.cntConfT = off; off += BsAlign((int64_t)numChunksMax * rs * 4, WS_ALIGN);
    w.candRec = off; off += BsAlign(nnz * REC_INTS * 4, WS_ALIGN);
    w.confRec = off; off += BsAlign((nnz + (int64_t)numChunksMax * cores) * REC_INTS * 4, WS_ALIGN);
    w.frNodes = off; off += BsAlign(numRows * 4, WS_ALIGN);
    w.stgIds = off;  off += BsAlign((int64_t)numChunksMax * BsChunkElems(elemSize) * elemSize, WS_ALIGN);
    w.stgFr = off;   off += BsAlign((int64_t)numChunksMax * BsChunkElems(elemSize) * 4, WS_ALIGN);
    w.fragIds = off; off += BsAlign(64, WS_ALIGN);   // 两槽 fragIds[parity]
    w.fragFr = off;  off += BsAlign(64, WS_ALIGN);   // 两槽 fragFr[parity]
    w.total = off;
    return w;
}

// ---------------------------------------------------------------------------
// Kernel 主体
// ---------------------------------------------------------------------------
template <typename T, bool IS_EDGE>
class KernelBfs {
public:
    constexpr static int32_t CHUNK = (sizeof(T) == 4) ? 1024 : 512;   // 方案 §4
    constexpr static int32_t ELEMS_PER_SECTOR = 32 / (int32_t)sizeof(T);
    constexpr static int32_t TRANS_JB = 128;                          // P2.5 转置块宽

    __aicore__ inline KernelBfs() {}

    __aicore__ inline void Init(GM_ADDR indptr, GM_ADDR indices, GM_ADDR eid, GM_ADDR source,
                                GM_ADDR ids, GM_ADDR sections, GM_ADDR ws, int64_t countOff,
                                int64_t numRows, int64_t nnz, int64_t S, int64_t hasEid,
                                int64_t tilingKey, int64_t numBlocks, int64_t numChunksMax)
    {
        ipGm_.SetGlobalBuffer((__gm__ T *)indptr);
        idxGm_.SetGlobalBuffer((__gm__ T *)indices);
        if (IS_EDGE && hasEid) eidGm_.SetGlobalBuffer((__gm__ T *)eid);
        srcGm_.SetGlobalBuffer((__gm__ T *)source);
        idsGm_.SetGlobalBuffer((__gm__ T *)ids);
        secGm_.SetGlobalBuffer((__gm__ int64_t *)sections);

        N_ = numRows; NNZ_ = nnz; S_ = S; hasEid_ = hasEid;
        key_ = (int32_t)tilingKey;
        C_ = (int32_t)numBlocks;
        numChunksMax_ = (int32_t)numChunksMax;
        CH_ = CHUNK;
        rowStride_ = BfsRowStrideInt32(C_);
        rowStrideJ_ = BfsRowStrideJ(numChunksMax_);
        blk_ = (int32_t)AscendC::GetBlockIdx();

        wsBfs_ = BfsWorkspaceLayout(numRows, nnz, C_, numChunksMax, sizeof(T));
        wsGm_.SetGlobalBuffer((__gm__ int8_t *)ws);
        visGm_.SetGlobalBuffer((__gm__ int32_t *)(ws + wsBfs_.visTag));
        syncGmT_.SetGlobalBuffer((__gm__ int32_t *)(ws + wsBfs_.syncGm));
        errGm_.SetGlobalBuffer((__gm__ int32_t *)(ws + wsBfs_.errflag));
        segGm_.SetGlobalBuffer((__gm__ int32_t *)(ws + wsBfs_.segByCore));
        cntDomGm_.SetGlobalBuffer((__gm__ int32_t *)(ws + wsBfs_.cntDom));
        cntConfDGm_.SetGlobalBuffer((__gm__ int32_t *)(ws + wsBfs_.cntConfD));
        cntConfTGm_.SetGlobalBuffer((__gm__ int32_t *)(ws + wsBfs_.cntConfT));
        candGm_.SetGlobalBuffer((__gm__ int32_t *)(ws + wsBfs_.candRec));
        confGm_.SetGlobalBuffer((__gm__ int32_t *)(ws + wsBfs_.confRec));
        frNodesGm_.SetGlobalBuffer((__gm__ int32_t *)(ws + wsBfs_.frNodes));
        fragIdsGm_.SetGlobalBuffer((__gm__ int32_t *)(ws + wsBfs_.fragIds));
        fragFrGm_.SetGlobalBuffer((__gm__ int32_t *)(ws + wsBfs_.fragFr));
        stgIdsGm_.SetGlobalBuffer((__gm__ T *)(ws + wsBfs_.stgIds));
        stgIds32Gm_.SetGlobalBuffer((__gm__ int32_t *)(ws + wsBfs_.stgIds));
        stgFrGm_.SetGlobalBuffer((__gm__ int32_t *)(ws + wsBfs_.stgFr));
        // [集成计数槽] ws 尾部追加 16B 计数槽（host 分配 wsBytes+16），block0 退出写 idsLen/secLen
        countOff_ = countOff;
        if (countOff_ >= 0) countGm_.SetGlobalBuffer((__gm__ int64_t *)(ws + countOff_));

        domLo_ = DomBound(blk_);
        domHi_ = (blk_ + 1 == C_) ? BsAlign(N_ * 4, 32) / 4 : DomBound(blk_ + 1);
    }

    __aicore__ inline void Process()
    {
        AllocUb();
        // ---- 每核 errflag 槽自净（标量写 + dcci、host 可见；3.4 第 3 轮修复终版）----
        // 关键：本核槽的最终值必须由**本核 kernel 在 launch 之后**确定，而不是依赖 host 侧
        // 预清零的可见性/时序（实测：host 预清零与回读之间不保证按预期落地——合法输入会读到
        // `at::empty` 的陈旧残留 → 误报非类别码；非法输入的类别码也可能被后到的预清零抹掉
        // → raised=false）。故 Key0/Key1/Key2 三条路径统一在入口先标量写 0 + dcci；
        // 随后若校验失败，同核 SetErr(1..4) 以标量写 + DataCacheCleanAndInvalid 覆盖
        // （同核同址 in-order；SetErr 本体注释见下）。
        SetErr(0);
        if (key_ == 2 && BFS_ENABLE_KEY2_MULTICORE) {
            ProcessMultiCore();
        } else {
            // 保底：key=2 关闭多核路径后（numBlocks=40 仍由 tiling 下发），仅 block0 执行
            // 串行（Key1 GM 标量语义，已验证正确），其余核自净 errflag 后直接返回
            // （串行无跨核交互、无屏障，40 核并发跑串行会互相踩输出，故必须单核）。
            if (blk_ == 0) ProcessSerial(key_ == 0);
        }
        // ---- 退出：写计数槽（idsLen/secLen，供 host 精确长度截断；block0 单写者）----
        // 仿 SetErr 的跨 agent（kernel→host）可见性模式：标量写 GM + DataCacheCleanAndInvalid
        // + PipeBarrier。计数槽位于 ws 尾部（countOff_ = wsBytes），32B 对齐，
        // SINGLE_CACHE_LINE dcci 刷整 32B 扇区（槽 16B + 尾部 16B 填充）。
        if (blk_ == 0 && countOff_ >= 0) {
            countGm_.SetValue(0, idsOutLen_);
            countGm_.SetValue(1, secOutLen_);
            AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(countGm_[0]);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    // ------------------------- UB 资源 -------------------------
    __aicore__ inline void AllocUb()
    {
        pipe_.InitBuffer(syncUb_, 8 * C_ * sizeof(int32_t));
        pipe_.InitBuffer(miscUb_, 512 * sizeof(int32_t));
        pipe_.InitBuffer(alignUb_, 256 * sizeof(int32_t));  // 除 32B 扇区 scratch 外，P1/P2.5 需写 C_×(int32) 大行（≤ GetCoreNumAiv）
        // 行窗口容量 = max(CHUNK, SRC_WIN)：P0/P1 行段分窗用 CHUNK，源层扫描（Key0/Key1/Key2 三处）
        // 固定 2048 元素分窗 —— 取二者较大值，否则 SRC_WIN > CHUNK+16 时 DataCopyPad 越界写 UB。
        pipe_.InitBuffer(lineIdxUb_, (int32_t)((CHUNK > SRC_WIN ? CHUNK : SRC_WIN) + 16) * sizeof(T));
        if (IS_EDGE) pipe_.InitBuffer(lineEidUb_, CHUNK * sizeof(T));
        pipe_.InitBuffer(frontWinUb_, 2048 * sizeof(int32_t));
        syncUbLoc_ = syncUb_.Get<int32_t>();
        miscLoc_ = miscUb_.Get<int32_t>();
        alignLoc_ = alignUb_.Get<int32_t>();
        lineIdxLoc_ = lineIdxUb_.template Get<T>();
        frontWinLoc_ = frontWinUb_.Get<int32_t>();
        if (IS_EDGE) lineEidLoc_ = lineEidUb_.template Get<T>();
        if (key_ == 2) {
            pipe_.InitBuffer(candStageUb_, CHUNK * REC_INTS * sizeof(int32_t));
            pipe_.InitBuffer(confStageUb_, CHUNK * REC_INTS * sizeof(int32_t));
            pipe_.InitBuffer(sliceUb_, (CHUNK + 64) * REC_INTS * sizeof(int32_t));
            pipe_.InitBuffer(outUb_, CHUNK * REC_INTS * sizeof(int32_t));
            pipe_.InitBuffer(transUb_, TRANS_JB * 64 * sizeof(int32_t));
            pipe_.InitBuffer(metaUb_, 256 * sizeof(int32_t));
            pipe_.InitBuffer(misc2Ub_, 128 * sizeof(int32_t));
            candStageLoc_ = candStageUb_.Get<int32_t>();
            confStageLoc_ = confStageUb_.Get<int32_t>();
            sliceLoc_ = sliceUb_.Get<int32_t>();
            outLoc_ = outUb_.Get<int32_t>();
            transLoc_ = transUb_.Get<int32_t>();
            metaLoc_ = metaUb_.Get<int32_t>();
            miscLoc2Loc_ = misc2Ub_.Get<int32_t>();
        } else if (key_ == 0) {
            // Key0：UB 常驻工作集（预算 ≤96KB 由 host 侧 calc 校验，方案 §7）
            pipe_.InitBuffer(k0IpUb_, (N_ + 1) * sizeof(T));
            pipe_.InitBuffer(k0IdxUb_, NNZ_ * sizeof(T));
            if (IS_EDGE && hasEid_) pipe_.InitBuffer(k0EidUb_, NNZ_ * sizeof(T));
            pipe_.InitBuffer(k0VisUb_, N_ * sizeof(int32_t));
            k0IpLoc_ = k0IpUb_.template Get<T>();
            k0IdxLoc_ = k0IdxUb_.template Get<T>();
            if (IS_EDGE && hasEid_) k0EidLoc_ = k0EidUb_.template Get<T>();
            k0VisLoc_ = k0VisUb_.Get<int32_t>();
            Gm2Ub<T>(k0IpLoc_, ipGm_, 0, (int32_t)((N_ + 1) * (int64_t)sizeof(T)));
            if (NNZ_ > 0) Gm2Ub<T>(k0IdxLoc_, idxGm_, 0, (int32_t)(NNZ_ * (int64_t)sizeof(T)));
            if (IS_EDGE && hasEid_ && NNZ_ > 0)
                Gm2Ub<T>(k0EidLoc_, eidGm_, 0, (int32_t)(NNZ_ * (int64_t)sizeof(T)));
        }
    }

    // ------------------------- 通用小工具 -------------------------
    // MTE 写 → 后续标量读 的同核一致性刷新（s12m2 实证：ENTIRE_DATA_CACHE 有效）
    __aicore__ inline void DcciEntire()
    {
        AscendC::PipeBarrier<PIPE_ALL>();  // 先等全部 MTE 写落地（实测定向 barrier 不可靠）
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
            AscendC::DcciDst::CACHELINE_ALL>(visGm_[0]);
    }

    __aicore__ inline void Barrier()
    {
        // [FIX-mc2] 跨核 MTE 读可见性修复（WB-K2-01~04 上板实测 E 抖动 ±10 / 逐核 myDeg 偏差；
        //  WB-K2-05/06 深尾层因各核 W 发散 → barrier 计数不齐 → SyncAll deadlock）。
        //  根因：原实现在 SyncAll **之后**才 dcci —— SyncAll 末尾的 flag 写（MTE3）先于本核
        //  数据 flush 被他核看到，读核可能在写核数据落 GM 前过屏障，读到陈旧扇区。
        //  修复：把「排空本核全部 MTE 写 + ENTIRE_DATA_CACHE clean+invalidate（DcciEntire 口径，
        //  s12m2 实证 ENTIRE 有效）」移到 SyncAll **之前**，保证「本核写落 GM → flag → 他核读」
        //  的确定性序；屏障后再补 PipeBarrier<PIPE_ALL> 兜底读侧。
        DcciEntire();
        AscendC::SyncAll(syncGmT_, syncUbLoc_, (int32_t)C_);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void SetErr(int32_t cls)
    {
        // errflag 的消费者是 host（`bfs_check_errflag` 经 aclrtMemcpy D2H 回读），
        // 属「跨 agent 共享数据」。
        //
        // 3.4 上板第 3 轮修复（终版）：errflag 槽用 **标量写 GM + DataCacheCleanAndInvalid**。
        // 依据（simt-precision-debug / DCache 一致性）：
        //   「Scalar 单元写 GM 后必须 DataCacheCleanAndInvalid 刷新 Cache，保证 Cache 一致性」
        //   跨 agent（kernel→host）与跨核同属 DCache 一致性场景。
        // 实测迭代（01:2x~01:3x）：
        //   · 标量写 GM（§4.1 原版）无 dcci → host 读不到（陈旧值）→ L2-04 等 raised=false；
        //   · MTE3 写 + dcci(ENTIRE_DATA_CACHE/CACHELINE_ALL) → 部分用例误报残留类别码；
        //   · MTE3 写 + dcci(SINGLE_CACHE_LINE/CACHELINE_OUT) → 仍不可靠（MTE3 写不经过
        //     Scalar DCache，dcci 刷不到；且 host 首读早于 L2 落 GM）；
        //   · 标量写 + dcci(SINGLE_CACHE_LINE/CACHELINE_OUT)（本版）→ 与官方 HCCL/msg 模式一致，
        //     写经 Scalar DCache、dcci 落 GM、host 首读即可见。
        // [对照已验证] MTE 版本下 WB-K2-01 仍 FAIL → Key2 multi-chunk 缺陷与本 SetErr 修改无关。
        for (int32_t k = 0; k < 8; ++k) errGm_.SetValue((int64_t)blk_ * 8 + k, cls);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(errGm_[blk_ * 8]);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 读 GM → UB（MTE2，dst 32B 对齐），cnt 为字节数
    template <typename DT>
    __aicore__ inline void Gm2Ub(AscendC::LocalTensor<DT> dst, const AscendC::GlobalTensor<DT> &src,
                                 int64_t srcOff, int32_t bytes)
    {
        if (bytes <= 0) return;
        AscendC::DataCopyExtParams cp{(uint16_t)1, (uint32_t)bytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT> rp{false, 0, 0, 0};
        AscendC::DataCopyPad(dst, src[srcOff], cp, rp);
        // 实测（3.1 穿刺+调试）：PipeBarrier<PIPE_MTE2> 不能可靠保证后续标量 UB 读可见，
        // 按 api-pipeline.md 调试指引以 PIPE_ALL 换正确性；性能影响留 CP4 校准。
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 写 UB → GM（MTE3，src 须 32B 对齐），cnt 为字节数
    template <typename DT>
    __aicore__ inline void Ub2Gm(const AscendC::GlobalTensor<DT> &dst, int64_t dstOff,
                                 const AscendC::LocalTensor<DT> &src, int32_t bytes)
    {
        if (bytes <= 0) return;
        AscendC::DataCopyExtParams cp{(uint16_t)1, (uint32_t)bytes, 0, 0, 0};
        AscendC::DataCopyPad(dst[dstOff], src, cp);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    // visTag 零初始化：跨核对齐分区（各核整 32B 扇区、互不重叠，穿刺④约束）
    // [CP5 #1] FillZeroPartition 已删除：Key0/Key2 清零均改用内联 Duplicate/分区写
    //（Key0 L511、Key2 序言 L622-625），该函数无调用点；再启用多核路径无需恢复。
    // 从 UB 标量读 T（Key0 走 UB，Key1/2 走 GM；idx 为 T 元素下标）
    __aicore__ inline T RdIp(int64_t i) const
    {
        if (key_ == 0) return k0IpLoc_.GetValue((int32_t)i);  // Key0 限 UB 预算内（≤96KB）
        return ipGm_.GetValue(i);
    }
    __aicore__ inline T RdIdx(int64_t i) const
    {
        if (key_ == 0) return k0IdxLoc_.GetValue((int32_t)i);
        return idxGm_.GetValue(i);
    }
    __aicore__ inline T RdEid(int64_t i) const
    {
        if (!hasEid_) return (T)i;
        if (key_ == 0) return k0EidLoc_.GetValue((int32_t)i);
        return eidGm_.GetValue(i);
    }
    __aicore__ inline int32_t RdVis(int64_t v) const
    {
        if (key_ == 0) return k0VisLoc_.GetValue((int32_t)v);
        return visGm_.GetValue(v);
    }
    __aicore__ inline void WrVis(int64_t v, int32_t val)
    {
        if (key_ == 0) k0VisLoc_.SetValue((int32_t)v, val);
        else visGm_.SetValue(v, val);
    }

    // ---- 多核（Key2）visTag 读写：一律 MTE 32B 扇区（穿刺②：标量 GetValue/SetValue 跨屏障
    //      不可靠——同核自写自读仅 tight-loop 内成立，跨 B3/B4/B5 屏障后标量读回陈旧值，
    //      导致层间去重失效；且 GlobalTensor::SetValue/GetValue 属 API 黑名单）。
    //      visTag 写者为 v 域 owner（单写者），域界 8-int32 对齐 → 32B 扇区不跨域，RMW 安全。 ----
    __aicore__ inline int32_t RdVisMc(int64_t v)
    {
        int64_t secBase = (v >> 3) << 3;   // 32B 扇区起始（8 int32）
        Gm2Ub<int32_t>(metaLoc_, visGm_, secBase, 32);
        return metaLoc_.GetValue((int32_t)(v - secBase));
    }
    __aicore__ inline void WrVisMc(int64_t v, int32_t val)
    {
        int64_t secBase = (v >> 3) << 3;
        Gm2Ub<int32_t>(metaLoc_, visGm_, secBase, 32);
        metaLoc_.SetValue((int32_t)(v - secBase), val);
        Ub2Gm<int32_t>(visGm_, secBase, metaLoc_, 32);
    }

    // ---- 微层单 owner（O1）：core0 串行复刻 CPU 伪代码，产出本层节点（CPU 首-hit 序）
    //      到 stgIds/stgFr（chunk 槽位化，ReadStgByPos 同口径），返回 W。其余核不调用。
    //      父节点读走 MTE 32B 扇区（上一层层输出为跨核写）；visTag 读写走 MTE（RdVisMc/WrVisMc）。
    __aicore__ inline int64_t MicroLayerBfs(int64_t frBase, int64_t frontLen, int64_t frontStart, int32_t curLayer)
    {
        int64_t nw = 0;
        // [FIX-O1] 防御性最大迭代保护（WB-K2-05/06 深尾 O1 分支曾 NPU 150s hang）：
        // 上界 = 层内边数上界（本分支前置条件 E≤TINY_E，正确数据边迭代总数=E）的 4 倍
        // + 父节点数裕量。正确输入永不触发；仅当跨核读异常（如前沿/indptr 读到异常大
        // 度数节点）导致循环失控时截断：输出已知正确前缀并 SetErr(5)，保证不 hang，
        // host 侧按错误类 5 抛异常快速失败（不允许无限循环）。
        const int64_t iterGuard = (int64_t)TINY_E * 4 + (int64_t)TINY_F * 2;
        int64_t nIter = 0;
        for (int64_t fi = 0; fi < frontLen; ++fi) {
            int64_t u;
            if (IS_EDGE) {
                int64_t abs = frBase + fi;
                int64_t secBase = (abs >> 3) << 3;
                Gm2Ub<int32_t>(metaLoc_, frNodesGm_, secBase, 32);
                u = (int64_t)metaLoc_.GetValue((int32_t)(abs - secBase));
            } else {
                int64_t secElems = 32 / (int64_t)sizeof(T);
                int64_t abs = frontStart + fi;
                int64_t secBase = abs / secElems * secElems;
                Gm2Ub<T>(lineIdxLoc_[CHUNK], idsGm_, secBase, 32);
                u = (int64_t)lineIdxLoc_.GetValue((int32_t)(CHUNK + (abs - secBase)));
            }
            if (u < 0 || u >= N_) continue;
            int64_t rs = (int64_t)ipGm_.GetValue(u);
            int64_t re = (int64_t)ipGm_.GetValue(u + 1);
            for (int64_t idx = rs; idx < re; ++idx) {
                if (++nIter > iterGuard) { SetErr(5); break; }
                int64_t v = (int64_t)idxGm_.GetValue(idx);
                if (v < 0 || v >= N_) { SetErr(4); continue; }
                if (RdVisMc(v) != 0) continue;
                WrVisMc(v, curLayer + 1);
                int32_t slot = (int32_t)(nw % CH_);
                if (!IS_EDGE) {
                    lineIdxLoc_.SetValue(slot, (T)v);
                } else {
                    lineIdxLoc_.SetValue(slot, RdEid(idx));
                    frontWinLoc_.SetValue(slot, (int32_t)v);
                }
                ++nw;
                if (nw % CH_ == 0) {
                    int64_t j = nw / CH_ - 1;
                    Ub2Gm<T>(stgIdsGm_, j * CH_, lineIdxLoc_, (int32_t)(CH_ * (int64_t)sizeof(T)));
                    if (IS_EDGE) Ub2Gm<int32_t>(stgFrGm_, j * CH_, frontWinLoc_, (int32_t)(CH_ * 4));
                }
            }
            if (nIter > iterGuard) break;
        }
        if (nw % CH_ != 0) {
            int64_t j = nw / CH_;
            int64_t rem = nw % CH_;
            int64_t secElems = 32 / (int64_t)sizeof(T);
            int64_t rem2 = (rem + secElems - 1) / secElems * secElems;
            for (int64_t m = rem; m < rem2; ++m) lineIdxLoc_.SetValue((int32_t)m, (T)0);
            Ub2Gm<T>(stgIdsGm_, j * CH_, lineIdxLoc_, (int32_t)(rem2 * (int64_t)sizeof(T)));
            if (IS_EDGE) {
                int64_t rem8 = (rem + 7) / 8 * 8;
                for (int64_t m = rem; m < rem8; ++m) frontWinLoc_.SetValue((int32_t)m, 0);
                Ub2Gm<int32_t>(stgFrGm_, j * CH_, frontWinLoc_, (int32_t)(rem8 * 4));
            }
        }
        return nw;
    }

    // ------------------------- 串行路径（Key0 / Key1） -------------------------
    __aicore__ inline void ProcessSerial(bool ubResident)
    {
        (void)ubResident;
        // prologue 校验：indptr 非降 + 末值=nnz（单核全查）
        for (int64_t i = 0; i < N_; ++i) {
            if (RdIp(i) > RdIp(i + 1)) { SetErr(1); break; }
        }
        if (N_ > 0 && RdIp(N_) != (T)NNZ_) SetErr(2);
        // visTag 清零
        if (key_ == 0) {
            for (int64_t i = 0; i < N_; i += 2048) {
                int32_t cnt = (int32_t)BsMinI64(N_ - i, (int64_t)2048);
                AscendC::Duplicate(k0VisLoc_[i], (int32_t)0, cnt);
                AscendC::PipeBarrier<PIPE_V>();
            }
        } else {
            // Key1：visTag 标量清零（标量自写自读为 s4 实证可靠模式；
            // MTE 写→标量读即使 ENTIRE dcci 仍实测不可靠，性能留 CP4 优化）
            for (int64_t i = 0; i < N_; ++i) visGm_.SetValue(i, 0);
        }
        // 源层
        for (int64_t i = 0; i < S_; ++i) {
            int64_t u = (int64_t)srcGm_.GetValue(i);
            if (u < 0 || u >= N_) { SetErr(3); continue; }
            WrVis(u, 1);
        }
        int64_t T_ = 0;      // ids 写入游标
        int64_t frTotal = 0; // edges 版 frontier 总数
        int64_t secIdx = 0;
        if (!IS_EDGE && S_ > 0) {
            // ids[0..S) = source 拷贝（MTE 批量，O3）
            for (int64_t off = 0; off < S_; off += SRC_WIN) {
                int64_t cnt = BsMinI64(S_ - off, (int64_t)SRC_WIN);
                Gm2Ub<T>(lineIdxLoc_, srcGm_, off, (int32_t)(cnt * (int64_t)sizeof(T)));
                AscendC::PipeBarrier<PIPE_ALL>();  // MTE2→MTE3 交接（跨队列汇聚点，非热路径）
                Ub2Gm<T>(idsGm_, off, lineIdxLoc_, (int32_t)(cnt * (int64_t)sizeof(T)));
            }
            T_ = S_;
            secGm_.SetValue(secIdx++, (int64_t)S_);
        }
        if (IS_EDGE && S_ > 0) {
            // edges 版源层：源节点写入 frNodes 作为第 1 层 frontier（不写 ids、无 sections 首项）
            for (int64_t i = 0; i < S_; ++i) {
                int64_t u = (int64_t)srcGm_.GetValue(i);
                frNodesGm_.SetValue(i, (int32_t)u);
            }
            frTotal = S_;
        }
        int64_t frontStart = 0, frontLen = S_;
        int32_t curLayer = 1;
        while (frontLen > 0) {
            int64_t W = 0;
            int64_t frBase = frTotal;  // 本层父在 frontier 数组中的起点（edges）
            for (int64_t fi = 0; fi < frontLen; ++fi) {
                int64_t u;
                if (IS_EDGE) u = frNodesGm_.GetValue(frontStart + fi);
                else u = (int64_t)idsGm_.GetValue(frontStart + fi);
                if (u < 0 || u >= N_) continue;
                int64_t rs = (int64_t)RdIp(u), re = (int64_t)RdIp(u + 1);
                int64_t deg = re - rs;
                bool mte = (deg >= D0_ROW_MTE);   // O2：行段 MTE 混合搬运
                // [FIX-mc3] 行段按 CH_ 分窗 MTE 载入（度数可 > CHUNK，如 lollipop 交汇节点
                // deg=50002）：原实现只拷 min(deg,CHUNK) 元素却循环读 deg 个 → UB 越界读
                // → vector core exception（WB-K2-05/06 保底退避串行后实测暴露）。与
                // ProcessMultiCore P1 同款分窗，读 offsets 用 CSR 偏移、定序用层内 pos。
                for (int64_t offBase = 0; offBase < deg; offBase += CH_) {
                    int64_t offLen = BsMinI64(deg - offBase, (int64_t)CH_);
                    if (mte) {
                        Gm2Ub<T>(lineIdxLoc_, idxGm_, rs + offBase, (int32_t)(offLen * (int64_t)sizeof(T)));
                        if (IS_EDGE && hasEid_)
                            Gm2Ub<T>(lineEidLoc_, eidGm_, rs + offBase, (int32_t)(offLen * (int64_t)sizeof(T)));
                    }
                    for (int64_t off = offBase; off < offBase + offLen; ++off) {
                        int64_t idx = rs + off;
                        int64_t v;
                        if (mte) v = (int64_t)lineIdxLoc_.GetValue((int32_t)(off - offBase));
                        else v = (int64_t)RdIdx(idx);
                        if (v < 0 || v >= N_) { SetErr(4); continue; }
                        if (RdVis(v) == 0) {
                            WrVis(v, curLayer + 1);
                            if (IS_EDGE) {
                                T ev = RdEid(idx);
                                idsGm_.SetValue(T_++, ev);
                                frNodesGm_.SetValue(frTotal + W, (int32_t)v);
                            } else {
                                idsGm_.SetValue(T_++, (T)v);
                            }
                            ++W;
                        }
                    }
                }
            }
            if (W > 0) secGm_.SetValue(secIdx++, (int64_t)W);
            if (IS_EDGE) frontStart = frBase;
            else frontStart = T_ - W;
            frontLen = W;
            frTotal += W;
            ++curLayer;
        }
        // [集成计数槽] 保存精确输出长度（host 回读截断用）
        idsOutLen_ = T_;
        secOutLen_ = secIdx;
    }

    // ------------------------- 多核分层路径（Key2，方案 R2 备选：无原子 owner-去重） -------------------------
    // 域界按 32B 扇区对齐（8 int32），保证 visTag 的 MTE 分区清零/标量写严格单写者（穿刺④）
    __aicore__ inline int64_t DomBound(int32_t d) const
    {
        return BsAlign((int64_t)d * N_ / C_ * 4, 32) / 4;
    }
    __aicore__ inline int32_t DomOf(int64_t v) const
    {
        int32_t d = (int32_t)((int64_t)v * C_ / N_);
        if (d >= C_) d = C_ - 1;
        while (d > 0 && DomBound(d) > v) --d;
        while (d + 1 < C_ && DomBound(d + 1) <= v) ++d;
        return d;
    }

    __aicore__ inline void ProcessMultiCore()
    {
        // ---- 序言：visTag 分区清零 + indptr 分段校验 ----
        // 注：syncGm 屏障计数器由 plugin 侧 host 整块预清零（穿刺②：kernel 序言标量清零对屏障 MTE 读不可见）；
        //     errflag 本核槽的自净已在 Process() 入口统一完成（Key0/Key1/Key2 共用），此处不再重复。
        {
            // visTag 域内清零（MTE 32B 扇区写，域界 8-int32 对齐 → 整扇区单写者）
            AscendC::Duplicate(alignLoc_, (int32_t)0, (int32_t)8);
            AscendC::PipeBarrier<PIPE_V>();
            for (int64_t off = domLo_; off < domHi_; off += 8) {
                Ub2Gm<int32_t>(visGm_, off, alignLoc_, 32);
            }
        }
        {
            int64_t lo = BsPartLo(blk_, N_, C_), hi = BsPartHi(blk_, N_, C_);
            for (int64_t i = lo; i < hi; ++i) {
                if (ipGm_.GetValue(i) > ipGm_.GetValue(i + 1)) { SetErr(1); break; }
            }
            if (blk_ + 1 == C_ && N_ > 0 && ipGm_.GetValue(N_) != (T)NNZ_) SetErr(2);
        }
        Barrier();  // B_init

        // ---- 源层 ----
        int64_t idsTotal = 0, frTotal = 0, secIdx = 0;
        if (IS_EDGE) {
            // 源层标记：全源扫描，域内单写者标记（MTE）；越界 SetErr(3)
            for (int64_t off = 0; off < S_; off += SRC_WIN) {
                int64_t cnt = BsMinI64(S_ - off, (int64_t)SRC_WIN);
                Gm2Ub<T>(lineIdxLoc_, srcGm_, off, (int32_t)(cnt * (int64_t)sizeof(T)));
                for (int64_t k = 0; k < cnt; ++k) {
                    int64_t u = (int64_t)lineIdxLoc_.GetValue((int32_t)k);
                    if (u < 0 || u >= N_) { SetErr(3); continue; }
                    if (u >= domLo_ && u < domHi_) WrVisMc(u, 1);
                }
            }
            // 源节点写进 frNodes 作为第 1 层 frontier（32B 扇区对齐分区单写者，穿刺④）
            if (S_ > 0) {
                int64_t fLo = BsAlign((int64_t)blk_ * S_ * 4, 32) / 4;
                int64_t fHi = (blk_ + 1 == C_) ? S_
                    : BsAlign((int64_t)(blk_ + 1) * S_ * 4, 32) / 4;
                for (int64_t base = fLo; base < fHi; base += CH_) {
                    int64_t n = BsMinI64(fHi - base, (int64_t)CH_);
                    Gm2Ub<T>(lineIdxLoc_, srcGm_, base, (int32_t)(n * (int64_t)sizeof(T)));
                    for (int64_t k = 0; k < n; ++k)
                        frontWinLoc_.SetValue((int32_t)k, (int32_t)((int64_t)lineIdxLoc_.GetValue((int32_t)k)));
                    Ub2Gm<int32_t>(frNodesGm_, base, frontWinLoc_, (int32_t)(n * 4));
                }
                if (blk_ == 0 && S_ % ELEMS_PER_SECTOR != 0) {
                    // 源层尾扇区 pending frag → fragFr（层 1 头合并接力初值）
                    int32_t eps = ELEMS_PER_SECTOR;
                    int64_t secStart = S_ - S_ % eps;
                    int32_t fc = (int32_t)(S_ - secStart);
                    Gm2Ub<T>(lineIdxLoc_, srcGm_, secStart, (int32_t)(fc * (int64_t)sizeof(T)));
                    alignLoc_.SetValue(0, fc);
                    for (int32_t k = 0; k < fc; ++k)
                        alignLoc_.SetValue(1 + k, (int32_t)((int64_t)lineIdxLoc_.GetValue(k)));
                    Ub2Gm<int32_t>(fragFrGm_, 0, alignLoc_, 32);
                }
                frTotal = S_;
            }
        } else {
            // nodes：ids[0..S)=source 对齐分区拷贝 + 域内标记
            // 源层标记：全源扫描，域内单写者标记（MTE）；越界 SetErr(3)
            for (int64_t off = 0; off < S_; off += SRC_WIN) {
                int64_t cnt = BsMinI64(S_ - off, (int64_t)SRC_WIN);
                Gm2Ub<T>(lineIdxLoc_, srcGm_, off, (int32_t)(cnt * (int64_t)sizeof(T)));
                for (int64_t k = 0; k < cnt; ++k) {
                    int64_t u = (int64_t)lineIdxLoc_.GetValue((int32_t)k);
                    if (u < 0 || u >= N_) { SetErr(3); continue; }
                    if (u >= domLo_ && u < domHi_) WrVisMc(u, 1);
                }
            }
            if (S_ > 0 && blk_ == 0 && S_ % ELEMS_PER_SECTOR != 0) {
                // 源层尾扇区 pending frag（层 1 头合并的接力初值）
                int32_t eps = ELEMS_PER_SECTOR;
                int64_t secStart = S_ - S_ % eps;
                int32_t fc = (int32_t)(S_ - secStart);
                Gm2Ub<T>(lineIdxLoc_, srcGm_, secStart, (int32_t)(fc * (int64_t)sizeof(T)));
                alignLoc_.SetValue(0, fc);
                for (int32_t k = 0; k < fc; ++k) {
                    T v = lineIdxLoc_.GetValue(k);
                    if (sizeof(T) == 8) {
                        alignLoc_.SetValue(1 + k * 2, (int32_t)((int64_t)v & 0xffffffffll));
                        alignLoc_.SetValue(2 + k * 2, (int32_t)((uint64_t)v >> 32));
                    } else {
                        alignLoc_.SetValue(1 + k, (int32_t)v);
                    }
                }
                Ub2Gm<int32_t>(fragIdsGm_, 0, alignLoc_, 32);
            }
            if (S_ > 0) {
                int64_t bLo = BsAlign((int64_t)blk_ * S_ * (int64_t)sizeof(T), 32) / (int64_t)sizeof(T);
                int64_t bHi = (blk_ + 1 == C_) ? S_
                    : BsAlign((int64_t)(blk_ + 1) * S_ * (int64_t)sizeof(T), 32) / (int64_t)sizeof(T);
                for (int64_t base = bLo; base < bHi; base += CHUNK) {
                    int64_t n = BsMinI64(bHi - base, (int64_t)CHUNK);
                    Gm2Ub<T>(lineIdxLoc_, srcGm_, base, (int32_t)(n * (int64_t)sizeof(T)));
                    Ub2Gm<T>(idsGm_, base, lineIdxLoc_, (int32_t)(n * (int64_t)sizeof(T)));
                }
                if (blk_ == 0) secGm_.SetValue(secIdx++, (int64_t)S_);
            }
            idsTotal = S_;
        }
        int64_t frontStart = 0, frontLen = S_;
        int32_t curLayer = 1;
        int32_t pendSrc = -1;          // 待接力 frag 的来源 chunk（-1 无）
        int32_t pendParity = 0;        // frag 槽层奇偶

        Barrier();  // B_src：源层 ids/frNodes 分区写 → 层1 P0 跨核 MTE 读的可见性（穿刺④）
        while (frontLen > 0) {
            int64_t frBase = frTotal - frontLen;
            // ---- P0：度数前缀 ----
            int64_t segLo = BsPartLo(blk_, frontLen, C_);
            int64_t segHi = BsPartHi(blk_, frontLen, C_);
            int64_t myDeg = 0;
            for (int64_t off = segLo; off < segHi; off += CH_) {
                int64_t cnt = BsMinI64(segHi - off, (int64_t)CH_);
                if (IS_EDGE) {
                    // 跨核读 frNodes：32B 扇区对齐读（非对齐单窗口读跨核不可靠）
                    int64_t abs0 = frBase + off;
                    int64_t lo = abs0 / 8 * 8, skip = abs0 - lo;
                    int64_t rCnt = (cnt + skip + 7) / 8 * 8;   // 对齐 8（32B）
                    Gm2Ub<int32_t>(frontWinLoc_, frNodesGm_, lo, (int32_t)(rCnt * 4));
                    for (int64_t k = 0; k < cnt; ++k) {
                        int64_t u = (int64_t)frontWinLoc_.GetValue((int32_t)(k + skip));
                        if (u >= 0 && u < N_) myDeg += (int64_t)(ipGm_.GetValue(u + 1) - ipGm_.GetValue(u));
                    }
                } else {
                    int64_t sectorElems = 32 / (int64_t)sizeof(T);
                    int64_t abs0 = frontStart + off;
                    int64_t lo = abs0 / sectorElems * sectorElems, skip = abs0 - lo;
                    int64_t rCnt = (cnt + skip + sectorElems - 1) / sectorElems * sectorElems;  // 对齐 32B
                    Gm2Ub<T>(lineIdxLoc_, idsGm_, lo, (int32_t)(rCnt * (int64_t)sizeof(T)));
                    for (int64_t k = 0; k < cnt; ++k) {
                        int64_t u = (int64_t)lineIdxLoc_.GetValue((int32_t)(k + skip));
                        if (u >= 0 && u < N_) myDeg += (int64_t)(ipGm_.GetValue(u + 1) - ipGm_.GetValue(u));
                    }
                }
            }
            for (int32_t z = 0; z < 8; ++z) alignLoc_.SetValue(z, 0);
            alignLoc_.SetValue(0, (int32_t)myDeg);
            Ub2Gm<int32_t>(segGm_, (int64_t)blk_ * 8, alignLoc_, 32);
            Barrier();  // B1
            Gm2Ub<int32_t>(miscLoc_, segGm_, 0, C_ * 8 * 4);   // 每核 32B 块，lane0 为值
            int64_t E = 0, myStreamOff = 0;
            for (int32_t k = 0; k < C_; ++k) {
                int64_t d = (int64_t)miscLoc_.GetValue(k * 8);
                if (k < blk_) myStreamOff += d;
                E += d;
            }
            if (E == 0) break;  // 空层：无输出、无 sections，结束
            int32_t numChunks = (int32_t)BsDivUp((int32_t)E, CH_);
            int32_t parity = curLayer & 1;

            // ---- O1 微层单 owner：上一层 W≤TINY_F 且层边≤TINY_E → core0 串行、其余核仅 1 屏障 ----
            // [FIX-mc3] 保底：O1 分支默认关闭（BFS_ENABLE_O1_MICROLAYER=false），回退通用路径
            //（P1/P2/P3），避免深尾 O1 分支运行时 hang 风险（WB-K2-05/06）。
            if (BFS_ENABLE_O1_MICROLAYER && frontLen <= TINY_F && E <= TINY_E) {
                if (blk_ == 0) {
                    int64_t Wtiny = MicroLayerBfs(frBase, frontLen, frontStart, curLayer);
                    int32_t numChunksTiny = BsDivUp((int32_t)Wtiny, CH_);
                    if (numChunksTiny < 1) numChunksTiny = 1;
                    // 写 cntConfT[j][0]（chunk 计数，供 ReadStgByPos 前缀定位）
                    for (int32_t j = 0; j < numChunksTiny; ++j) {
                        int64_t cnt = BsMinI64((int64_t)(j + 1) * CH_, Wtiny) - (int64_t)j * CH_;
                        for (int32_t d = 0; d < rowStride_; ++d) alignLoc_.SetValue(d, 0);
                        alignLoc_.SetValue(0, (int32_t)cnt);
                        Ub2Gm<int32_t>(cntConfTGm_, (int64_t)j * rowStride_, alignLoc_, rowStride_ * 4);
                    }
                    // 扇区压缩写出 ids / frNodes（复用 P3b 辅助，core0 单写者走全扇区）
                    curLayer_ = curLayer;
                    int32_t eps = ELEMS_PER_SECTOR;
                    int64_t L0 = idsTotal, L1 = idsTotal + Wtiny;
                    for (int64_t s = (L0 + eps - 1) / eps; s < L1 / eps; ++s)
                        CompactSector(s, L0, Wtiny, numChunksTiny, false);
                    if (L0 % eps != 0) CompactHeadSector(L0, Wtiny, numChunksTiny, false);
                    if (L1 % eps != 0) StashTailFrag(L1, Wtiny, numChunksTiny, false);
                    if (IS_EDGE) {
                        int64_t F0 = frTotal, F1 = frTotal + Wtiny;
                        for (int64_t s = (F0 + eps - 1) / eps; s < F1 / eps; ++s)
                            CompactSector(s, F0, Wtiny, numChunksTiny, true);
                        if (F0 % eps != 0) CompactHeadSector(F0, Wtiny, numChunksTiny, true);
                        if (F1 % eps != 0) StashTailFrag(F1, Wtiny, numChunksTiny, true);
                    }
                    if (Wtiny > 0) secGm_.SetValue(secIdx, (int64_t)Wtiny);
                    // 广播 W 到 segGm_[0]（P0 数据已消费，复用）
                    for (int32_t d = 0; d < 8; ++d) alignLoc_.SetValue(d, 0);
                    alignLoc_.SetValue(0, (int32_t)Wtiny);
                    Ub2Gm<int32_t>(segGm_, 0, alignLoc_, 32);
                }
                Barrier();  // B_micro（其余核仅此一次屏障）
                Gm2Ub<int32_t>(miscLoc_, segGm_, 0, 32);
                int64_t W = (int64_t)miscLoc_.GetValue(0);
                if (W > 0) ++secIdx;
                idsTotal += W;
                frTotal += W;
                if (IS_EDGE) frontStart = frTotal - W;
                else frontStart = idsTotal - W;
                frontLen = W;
                ++curLayer;
                if (W == 0) break;
                continue;
            }

            // ---- P1：全 frontier 单遍扫描，边按 pos 所属 chunk 归属（chunk owner 单写者） ----
            {
                int64_t nodeCur = 0, streamOff = 0;
                int64_t winBase = 0, winLen = BsMinI64(frontLen, (int64_t)2048);
                if (IS_EDGE && winLen > 0)
                    Gm2Ub<int32_t>(frontWinLoc_, frNodesGm_, frBase, (int32_t)(winLen * 4));
                int64_t curChunk = -1;          // 当前落桶 chunk（-1 无）
                int64_t runStart = 0;           // 当前 staging 连续段起点（chunk 内）
                int32_t stageCnt = 0;
                for (int32_t d = 0; d < C_; ++d) miscLoc_.SetValue(32 + d, 0);
                while (nodeCur < frontLen && streamOff < E) {
                    // 父节点：nodes 走 MTE 单元素读（标量读 MTE 写数据实测陈旧，穿刺 m0）；edges 走 frontWin 窗口
                    int64_t u;
                    if (IS_EDGE) {
                        u = (int64_t)frontWinLoc_.GetValue((int32_t)(nodeCur - winBase));
                    } else {
                        // 跨核读上一层 P3b 写的 ids：须 32B 扇区对齐读（单元素 4/8B 跨核读实测不可靠，穿刺 m3/m4）
                        int64_t sectorElems = 32 / (int64_t)sizeof(T);
                        int64_t absPos = frontStart + nodeCur;
                        int64_t secBase = absPos / sectorElems * sectorElems;
                        Gm2Ub<T>(lineIdxLoc_, idsGm_, secBase, 32);
                        u = (int64_t)lineIdxLoc_.GetValue((int32_t)(absPos - secBase));
                    }
                    if (u < 0 || u >= N_) { ++nodeCur; continue; }  // 非法父节点（源越界等错误路径）计 0 边
                    int64_t csrBase = (int64_t)ipGm_.GetValue(u);            // indptr[u]
                    int64_t deg = (int64_t)ipGm_.GetValue(u + 1) - csrBase;  // 行度数
                    bool mte = (deg >= D0_ROW_MTE);                          // O2：行段 MTE 混合搬运
                    // 行段按 CH_ 分窗 MTE 载入（度数可 > CHUNK），读 offsets 用 CSR 偏移、定序用层内 pos
                    for (int64_t iBase = 0; iBase < deg; iBase += CH_) {
                        int64_t iLen = BsMinI64(deg - iBase, (int64_t)CH_);
                        if (mte) {
                            Gm2Ub<T>(lineIdxLoc_, idxGm_, csrBase + iBase, (int32_t)(iLen * (int64_t)sizeof(T)));
                            if (IS_EDGE && hasEid_)
                                Gm2Ub<T>(lineEidLoc_, eidGm_, csrBase + iBase, (int32_t)(iLen * (int64_t)sizeof(T)));
                        }
                        for (int64_t ii = 0; ii < iLen; ++ii) {
                            int64_t i = iBase + ii;
                            int64_t pos = streamOff + i;                     // 层内串行位次（唯一去重/定序键）
                            int64_t j = pos / CH_;
                            if (j % C_ != blk_) continue;                    // 非本核 chunk
                            if (j != curChunk) {
                                // flush 上一 chunk 连续段
                                if (stageCnt > 0) {
                                    int32_t pbytes = (int32_t)((stageCnt / 2) * 2 * REC_INTS * 4);
                                    Ub2Gm<int32_t>(candGm_, runStart * REC_INTS, candStageLoc_, pbytes);
                                    if (stageCnt % 2 == 1) {
                                        for (int32_t r = 0; r < REC_INTS; ++r)
                                            alignLoc_.SetValue(r, candStageLoc_.GetValue((stageCnt - 1) * REC_INTS + r));
                                        for (int32_t r = REC_INTS; r < 8; ++r) alignLoc_.SetValue(r, 0);
                                        Ub2Gm<int32_t>(candGm_, (runStart + stageCnt - 1) * REC_INTS, alignLoc_, 32);
                                    }
                                    for (int32_t d = 0; d < C_; ++d)
                                        alignLoc_.SetValue(d, miscLoc_.GetValue(32 + d));
                                    for (int32_t d = C_; d < rowStride_; ++d) alignLoc_.SetValue(d, 0);
                                    Ub2Gm<int32_t>(cntDomGm_, (int64_t)curChunk * rowStride_, alignLoc_, rowStride_ * 4);
                                }
                                curChunk = j; runStart = j * CH_; stageCnt = 0;
                                for (int32_t d = 0; d < C_; ++d) miscLoc_.SetValue(32 + d, 0);
                            }
                            int64_t v;
                            if (mte) v = (int64_t)lineIdxLoc_.GetValue((int32_t)ii);
                            else v = (int64_t)idxGm_.GetValue(csrBase + i);
                            if (v < 0 || v >= N_) { SetErr(4); v = -1; }
                            int64_t val;
                            if (!IS_EDGE) val = v;
                            else if (hasEid_) val = (int64_t)(mte ? lineEidLoc_.GetValue((int32_t)ii)
                                                                  : eidGm_.GetValue(csrBase + i));
                            else val = csrBase + i;   // 边 id = CSR 全局偏移 idx（需求 1.3，非 pos）
                            candStageLoc_.SetValue(stageCnt * 4 + 0, (int32_t)v);
                            candStageLoc_.SetValue(stageCnt * 4 + 1, (int32_t)pos);
                            candStageLoc_.SetValue(stageCnt * 4 + 2, (int32_t)(val & 0xffffffffll));
                            candStageLoc_.SetValue(stageCnt * 4 + 3, (int32_t)((uint64_t)val >> 32));
                            ++stageCnt;
                            if (v >= 0) {
                                int32_t dv = DomOf(v);
                                miscLoc_.SetValue(32 + dv, miscLoc_.GetValue(32 + dv) + 1);
                            }
                        }
                    }
                    streamOff += deg;
                    ++nodeCur;
                    if (IS_EDGE && nodeCur < frontLen && nodeCur - winBase >= 2048) {
                        winLen = BsMinI64(frontLen - nodeCur, (int64_t)2048);
                        Gm2Ub<int32_t>(frontWinLoc_, frNodesGm_, frBase + nodeCur, (int32_t)(winLen * 4));
                        winBase = nodeCur;
                    }
                }
                if (stageCnt > 0 && curChunk >= 0) {
                    int32_t pbytes = (int32_t)((stageCnt / 2) * 2 * REC_INTS * 4);
                    Ub2Gm<int32_t>(candGm_, runStart * REC_INTS, candStageLoc_, pbytes);
                    if (stageCnt % 2 == 1) {
                        for (int32_t r = 0; r < REC_INTS; ++r)
                            alignLoc_.SetValue(r, candStageLoc_.GetValue((stageCnt - 1) * REC_INTS + r));
                        for (int32_t r = REC_INTS; r < 8; ++r) alignLoc_.SetValue(r, 0);
                        Ub2Gm<int32_t>(candGm_, (runStart + stageCnt - 1) * REC_INTS, alignLoc_, 32);
                    }
                    for (int32_t d = 0; d < C_; ++d)
                        alignLoc_.SetValue(d, miscLoc_.GetValue(32 + d));
                    for (int32_t d = C_; d < rowStride_; ++d) alignLoc_.SetValue(d, 0);
                    Ub2Gm<int32_t>(cntDomGm_, (int64_t)curChunk * rowStride_, alignLoc_, rowStride_ * 4);
                }
            }
            Barrier();  // B2

            // ---- P2：v 域确认（visTag 域内自写自读；confRec (j,d) 分片单写者） ----
            {
                int64_t rowTotalPrev = 0;   // Σ_{j'<j} Σ_d align2(domHist[j'][d])
                int32_t myD = blk_;              // 域序号 = 块号（域界扇区对齐）
                for (int32_t j = 0; j < numChunks; ++j) {
                    Gm2Ub<int32_t>(miscLoc_, cntDomGm_, (int64_t)j * rowStride_, rowStride_ * 4);
                    int64_t baseMy = rowTotalPrev;
                    for (int32_t d = 0; d < (int32_t)myD && d < C_; ++d) {
                        int64_t capd = miscLoc_.GetValue(d);
                        baseMy += (capd + 1) / 2 * 2;
                    }
                    int64_t sliceLen = BsMinI64((int64_t)j * CH_ + CH_, E) - (int64_t)j * CH_;
                    int32_t confCnt = 0;
                    for (int64_t off = 0; off < sliceLen; off += 512) {
                        int64_t win = BsMinI64(sliceLen - off, (int64_t)512);
                        int64_t win2 = (win + 1) / 2 * 2;   // 对齐偶数（32B）：跨核读须 32B 对齐长度
                        Gm2Ub<int32_t>(sliceLoc_, candGm_, ((int64_t)j * CH_ + off) * REC_INTS,
                                       (int32_t)(win2 * REC_INTS * 4));
                        for (int64_t k = 0; k < win; ++k) {
                            int64_t v = (int64_t)sliceLoc_.GetValue((int32_t)(k * REC_INTS + 0));
                            if (v < domLo_ || v >= domHi_) continue;
                            if (RdVisMc(v) != 0) {
                                continue;
                            }
                            WrVisMc(v, curLayer + 1);
                            if (confCnt < CHUNK) {
                                for (int32_t r = 0; r < REC_INTS; ++r)
                                    confStageLoc_.SetValue(confCnt * REC_INTS + r,
                                        sliceLoc_.GetValue((int32_t)(k * REC_INTS + r)));
                                ++confCnt;
                            }
                        }
                    }
                    // flush 确认记录（baseMy 偶数 → 32B 对齐；(j,d) 分片单写者）
                    if (confCnt > 0) {
                        int32_t pbytes = (int32_t)((confCnt / 2) * 2 * REC_INTS * 4);
                        Ub2Gm<int32_t>(confGm_, baseMy * REC_INTS, confStageLoc_, pbytes);
                        if (confCnt % 2 == 1) {
                            for (int32_t r = 0; r < REC_INTS; ++r)
                                alignLoc_.SetValue(r, confStageLoc_.GetValue((confCnt - 1) * REC_INTS + r));
                            for (int32_t r = REC_INTS; r < 8; ++r) alignLoc_.SetValue(r, 0);
                            Ub2Gm<int32_t>(confGm_, (baseMy + confCnt - 1) * REC_INTS, alignLoc_, 32);
                        }
                    }
                    alignLoc_.SetValue(0, confCnt);
                    Ub2Gm<int32_t>(cntConfDGm_, ((int64_t)myD * rowStrideJ_ + j), alignLoc_, 4);
                    // own row 单写者 + MTE 写→MTE 读（转置读），实证可靠组合
                    for (int32_t d = 0; d < C_; ++d) {
                        int64_t capd = miscLoc_.GetValue(d);
                        rowTotalPrev += (capd + 1) / 2 * 2;
                    }
                }
            }
            Barrier();  // B3

            // ---- P2.5：core0 转置 cntConfD[d][j] → cntConfT[j][d]（行读写皆 MTE 安全） ----
            if (blk_ == 0) {
                for (int32_t jb = 0; jb < numChunks; jb += TRANS_JB) {
                    int32_t jl = (int32_t)BsMinI64((int64_t)TRANS_JB, (int64_t)numChunks - jb);
                    int32_t jl2 = (int32_t)((jl + 7) / 8 * 8);   // 对齐到 8（32B）：跨核读须 32B 对齐长度
                    for (int32_t d = 0; d < C_; ++d) {
                        Gm2Ub<int32_t>(transLoc_[d * TRANS_JB], cntConfDGm_,
                                       (int64_t)d * rowStrideJ_ + jb, jl2 * 4);
                    }
                    for (int32_t k = 0; k < jl; ++k) {
                        for (int32_t d = 0; d < C_; ++d)
                            alignLoc_.SetValue(d, transLoc_.GetValue(d * TRANS_JB + k));
                        for (int32_t d = C_; d < rowStride_; ++d)
                            alignLoc_.SetValue(d, 0);   // 填充到 32B 对齐（rowStride 个 int32）
                        Ub2Gm<int32_t>(cntConfTGm_, (int64_t)(jb + k) * rowStride_, alignLoc_, rowStride_ * 4);
                    }
                }
            }
            Barrier();  // B3.5


            // ---- P3a：own chunks pos-归并 → chunk 槽位化 staging 写出（位置对齐，单写者） ----
            int64_t W = 0, outBase = 0, sliceBasePrev = 0;
            for (int32_t j = 0; j < numChunks; ++j) {
                Gm2Ub<int32_t>(miscLoc_, cntConfTGm_, (int64_t)j * rowStride_, rowStride_ * 4);
                int64_t totJ = 0;
                for (int32_t d = 0; d < C_; ++d) totJ += miscLoc_.GetValue(d);
                W += totJ;
                // cntDom[j] 必须对每个 chunk 都读（sliceBasePrev 偏移累加不依赖 chunk owner）
                Gm2Ub<int32_t>(miscLoc2Loc_, cntDomGm_, (int64_t)j * rowStride_, rowStride_ * 4);
                if (totJ > 0 && j % C_ == blk_) {
                    int64_t sliceBase = sliceBasePrev;
                    int32_t slot = 0, nonEmpty = 0;
                    for (int32_t d = 0; d < C_; ++d) {
                        int64_t confLen = miscLoc_.GetValue(d);      // cntConfT[j][d]：确认记录数（读长/归并用）
                        int64_t cap = miscLoc2Loc_.GetValue(d);      // cntDom[j][d]：候选数（confRec 槽位对齐/偏移用）
                        if (confLen > 0) {
                            int64_t confLen2 = (confLen + 1) / 2 * 2;   // 对齐偶数（32B）
                            Gm2Ub<int32_t>(sliceLoc_[slot * REC_INTS], confGm_,
                                           sliceBase * REC_INTS, (int32_t)(confLen2 * REC_INTS * 4));
                            metaLoc_.SetValue(nonEmpty * 4 + 0, d);
                            metaLoc_.SetValue(nonEmpty * 4 + 1, slot);
                            metaLoc_.SetValue(nonEmpty * 4 + 2, (int32_t)confLen);
                            metaLoc_.SetValue(nonEmpty * 4 + 3, 0);
                            ++nonEmpty;
                        }
                        // sliceLoc_ 槽按确认数对齐偶数累加：既保证 dst 32B 对齐（DataCopyPad UB 侧须对齐），
                        // 又不按候选数 cap 预留（cap 可能远大于 confLen，会导致 UB 越界）。
                        slot += (int32_t)((confLen + 1) / 2 * 2);
                        sliceBase += (cap + 1) / 2 * 2;        // confRec 偏移仍按候选数对齐（confRec 槽位由候选数预留）
                    }
                    int64_t outIdx = 0;
                    for (;;) {
                        int32_t best = -1, bestP = 0;
                        for (int32_t sd = 0; sd < nonEmpty; ++sd) {
                            int32_t cur = metaLoc_.GetValue(sd * 4 + 3);
                            int32_t len = metaLoc_.GetValue(sd * 4 + 2);
                            if (cur < len) {
                                int32_t pp = sliceLoc_.GetValue((metaLoc_.GetValue(sd * 4 + 1) + cur) * REC_INTS + 1);
                                if (best < 0 || pp < bestP) { best = sd; bestP = pp; }
                            }
                        }
                        if (best < 0) break;
                        int32_t src = (metaLoc_.GetValue(best * 4 + 1) + metaLoc_.GetValue(best * 4 + 3)) * REC_INTS;
                        for (int32_t r = 0; r < REC_INTS; ++r)
                            outLoc_.SetValue((int32_t)(outIdx * REC_INTS + r), sliceLoc_.GetValue(src + r));
                        metaLoc_.SetValue(best * 4 + 3, metaLoc_.GetValue(best * 4 + 3) + 1);
                        ++outIdx;
                    }
                    // ids 流 → stgIds[j*CH + m]（chunk 槽位化，32B 对齐单写者）；frontier 流 → stgFr（int32）
                    for (int64_t m = 0; m < totJ; ++m) {
                        T v = (sizeof(T) == 8)
                            ? (T)(((int64_t)outLoc_.GetValue((int32_t)(m * REC_INTS + 3)) << 32) |
                                  (uint32_t)outLoc_.GetValue((int32_t)(m * REC_INTS + 2)))
                            : (T)(IS_EDGE ? outLoc_.GetValue((int32_t)(m * REC_INTS + 2))
                                          : outLoc_.GetValue((int32_t)(m * REC_INTS + 0)));
                        lineIdxLoc_.SetValue((int32_t)m, v);
                        if (IS_EDGE)
                            frontWinLoc_.SetValue((int32_t)m, outLoc_.GetValue((int32_t)(m * REC_INTS + 0)));
                    }
                    // 对齐到 32B（跨核读须 32B 完整扇区写，部分扇区写跨核读不可靠）
                    int64_t sectorElems = 32 / (int64_t)sizeof(T);
                    int64_t totJ2 = (totJ + sectorElems - 1) / sectorElems * sectorElems;
                    for (int64_t m = totJ; m < totJ2; ++m) lineIdxLoc_.SetValue((int32_t)m, (T)0);
                    Ub2Gm<T>(stgIdsGm_, (int64_t)j * CH_, lineIdxLoc_, (int32_t)(totJ2 * (int64_t)sizeof(T)));
                    if (IS_EDGE) {
                        int64_t totJ8 = (totJ + 7) / 8 * 8;
                        for (int64_t m = totJ; m < totJ8; ++m) frontWinLoc_.SetValue((int32_t)m, 0);
                        Ub2Gm<int32_t>(stgFrGm_, (int64_t)j * CH_, frontWinLoc_, (int32_t)(totJ8 * 4));
                    }
                }
                outBase += totJ;
                for (int32_t d = 0; d < C_; ++d) {
                    int64_t lend = miscLoc2Loc_.GetValue(d);
                    sliceBasePrev += (lend + 1) / 2 * 2;
                }
            }
            Barrier();  // B4

            // ---- P3b：单遍压缩 staging → ids/frNodes（扇区单写者）+ sections + pending frag（双槽按层奇偶） ----
            {
                curLayer_ = curLayer;   // 供 frag 双槽 parity 计算（读 parity=(curLayer-1)&1，写 parity=curLayer&1）
                int32_t eps = ELEMS_PER_SECTOR;
                int64_t L0 = idsTotal, L1 = idsTotal + W;
                // 我的全整扇区
                for (int64_t s = (L0 + eps - 1) / eps; s < L1 / eps; ++s) {
                    if (s % C_ != blk_) continue;
                    CompactSector(s, L0, W, numChunks, false);
                }
                // 头部分扇区（L0 非对齐时）：prev frag + staging 头
                if (L0 % eps != 0 && L0 / eps % C_ == blk_)
                    CompactHeadSector(L0, W, numChunks, false);
                // 尾部分扇区 → 单 frag 槽（下一层头合并 / 退出补写用）
                if (L1 % eps != 0 && (L1 - 1) / eps % C_ == blk_)
                    StashTailFrag(L1, W, numChunks, false);
                if (IS_EDGE) {
                    int64_t F0 = frTotal, F1 = frTotal + W;
                    for (int64_t s = (F0 + eps - 1) / eps; s < F1 / eps; ++s) {
                        if (s % C_ != blk_) continue;
                        CompactSector(s, F0, W, numChunks, true);
                    }
                    if (F0 % eps != 0 && F0 / eps % C_ == blk_)
                        CompactHeadSector(F0, W, numChunks, true);
                    if (F1 % eps != 0 && (F1 - 1) / eps % C_ == blk_)
                        StashTailFrag(F1, W, numChunks, true);
                }
                if (blk_ == 0 && W > 0) secGm_.SetValue(secIdx, (int64_t)W);
                if (W > 0) ++secIdx;
                idsTotal += W;
                frTotal += W;
                if (IS_EDGE) frontStart = frTotal - W;
                else frontStart = idsTotal - W;
                frontLen = W;
                ++curLayer;
            }
            Barrier();  // B5
            if (W == 0) break;
        }
        // ---- 退出：pending 尾扇区补写（frag + 0 补位；有效前缀之外为未定义 padding） ----
        {
            int32_t eps = ELEMS_PER_SECTOR;
            if (idsTotal % eps != 0 && (idsTotal - 1) / eps % C_ == blk_)
                FlushPending(idsTotal, false);
            if (IS_EDGE && frTotal % eps != 0 && (frTotal - 1) / eps % C_ == blk_)
                FlushPending(frTotal, true);
        }
        // [集成计数槽] 保存精确输出长度（host 回读截断用）
        idsOutLen_ = idsTotal;
        secOutLen_ = secIdx;
    }

    // ------------------------- 输出压缩辅助 -------------------------
    // 流值写入扇区 scratch（int32 流：槽 k；int64 流：偶偶对 2k/2k+1）
    __aicore__ inline void PutSectorVal(int32_t k, int32_t lo, int32_t hi, bool st)
    {
        if (st || sizeof(T) == 4) {
            alignLoc_.SetValue(k, lo);
        } else {
            alignLoc_.SetValue(2 * k, lo);
            alignLoc_.SetValue(2 * k + 1, hi);
        }
    }
    // 层内位置 p → (chunk j = p/CH, 槽内偏移 p - j*CH)；staging 槽按位置静态对齐
    // 从 staging 读流值到 sector scratch（int32 流槽 k；int64 流 2k/2k+1）
    __aicore__ inline void CompactSector(int64_t s, int64_t L0, int64_t W, int32_t numChunks, bool st)
    {
        int32_t eps = ELEMS_PER_SECTOR;
        int64_t p0 = s * eps - L0, p1 = p0 + eps;   // 层内位置区间
        for (int32_t k = 0; k < eps; ++k) {
            int64_t p = p0 + k;
            int32_t lo, hi = 0;
            ReadStgByPos(p, numChunks, st, lo, hi);
            PutSectorVal(k, lo, hi, st);
        }
        if (st) Ub2Gm<int32_t>(frNodesGm_, s * eps, alignLoc_, eps * 4);
        else Ub2Gm<T>(idsGm_, s * eps, alignLoc_.template ReinterpretCast<T>(), eps * (int32_t)sizeof(T));
    }

    __aicore__ inline void CompactHeadSector(int64_t L0, int64_t W, int32_t numChunks, bool st)
    {
        int32_t eps = ELEMS_PER_SECTOR;
        int64_t s = L0 / eps;
        Gm2Ub<int32_t>(miscLoc2Loc_, st ? fragFrGm_ : fragIdsGm_, ((curLayer_ - 1) & 1) * 8, 32);
        int32_t fc = miscLoc2Loc_.GetValue(0);
        for (int32_t k = 0; k < eps; ++k) {
            int32_t lo = 0, hi = 0;
            if (k < fc) {
                int32_t stride = st ? 1 : (int32_t)sizeof(T) / 4;
                lo = miscLoc2Loc_.GetValue(1 + k * stride);
                if (!st && sizeof(T) == 8) hi = miscLoc2Loc_.GetValue(2 + k * 2);
            } else {
                int64_t p = k - fc;   // 本层值从扇区位 fc 起，层内位置 0 起
                if (p < W) {
                    ReadStgByPos(p, numChunks, st, lo, hi);
                }
            }
            PutSectorVal(k, lo, hi, st);
        }
        if (st) Ub2Gm<int32_t>(frNodesGm_, s * eps, alignLoc_, eps * 4);
        else Ub2Gm<T>(idsGm_, s * eps, alignLoc_.template ReinterpretCast<T>(), eps * (int32_t)sizeof(T));
    }

    __aicore__ inline void StashTailFrag(int64_t L1, int64_t W, int32_t numChunks, bool st)
    {
        int32_t eps = ELEMS_PER_SECTOR;
        int64_t secStart = L1 - L1 % eps;
        int64_t L0 = L1 - W;
        int64_t oldSecStart = -1;
        if (L0 % eps != 0) oldSecStart = L0 - L0 % eps;
        bool sameSector = (oldSecStart >= 0 && secStart == oldSecStart);
        int32_t fc = (int32_t)(sameSector ? (L1 - oldSecStart) : (L1 - secStart));
        int32_t oldFragLen = (int32_t)(sameSector ? (L0 - oldSecStart) : 0);
        int32_t stride = st ? 1 : (int32_t)sizeof(T) / 4;

        alignLoc_.SetValue(0, fc);
        for (int32_t k = 0; k < eps; ++k) {
            int32_t lo = 0, hi = 0;
            if (k < fc) {
                if (k < oldFragLen) {
                    // 旧 frag 段（prev 层尾，读上一层写的 parity 槽）
                    Gm2Ub<int32_t>(miscLoc2Loc_, st ? fragFrGm_ : fragIdsGm_, ((curLayer_ - 1) & 1) * 8, 32);
                    lo = miscLoc2Loc_.GetValue(1 + k * stride);
                    if (!st && sizeof(T) == 8) hi = miscLoc2Loc_.GetValue(2 + k * 2);
                } else {
                    int64_t p = sameSector ? (int64_t)(k - oldFragLen) : (int64_t)(W - fc + k);
                    ReadStgByPos(p, numChunks, st, lo, hi);
                }
                if (st || sizeof(T) == 4) alignLoc_.SetValue(1 + k * stride, lo);
                else { alignLoc_.SetValue(1 + k * 2, lo); alignLoc_.SetValue(2 + k * 2, hi); }
            }
            // 同步构建 sector scratch（tail flush 用）
            if (st) frontWinLoc_.SetValue(k, lo);
            else if (sizeof(T) == 8) lineIdxLoc_.SetValue(k, (T)(((int64_t)hi << 32) | (uint32_t)lo));
            else lineIdxLoc_.SetValue(k, (T)lo);
        }
        lastWrParity_ = curLayer_ & 1;
        Ub2Gm<int32_t>(st ? fragFrGm_ : fragIdsGm_, lastWrParity_ * 8, alignLoc_, 32);
        // 同步 flush 尾扇区到 ids/frNodes（下一层 P0 前沿读取需要完整尾段），但不得超过数组上界
        {
            int64_t arrayElems = st ? N_ : (S_ + N_);   // frNodes 上界 N_；ids 上界 S_+N_
            int64_t avail = arrayElems - secStart;
            if (avail > 0) {
                int32_t welems = (int32_t)BsMinI64(avail, (int64_t)eps);
                if (st) Ub2Gm<int32_t>(frNodesGm_, secStart, frontWinLoc_, welems * 4);
                else Ub2Gm<T>(idsGm_, secStart, lineIdxLoc_, welems * (int32_t)sizeof(T));
            }
        }
    }

    __aicore__ inline void FlushPending(int64_t total, bool st)
    {
        int32_t eps = ELEMS_PER_SECTOR;
        int64_t secStart = total - total % eps;
        Gm2Ub<int32_t>(miscLoc2Loc_, st ? fragFrGm_ : fragIdsGm_, lastWrParity_ * 8, 32);
        int32_t fc = miscLoc2Loc_.GetValue(0);
        int32_t stride = st ? 1 : (int32_t)sizeof(T) / 4;
        for (int32_t k = 0; k < eps; ++k) {
            int32_t lo = 0, hi = 0;
            if (k < fc) {
                lo = miscLoc2Loc_.GetValue(1 + k * stride);
                if (!st && sizeof(T) == 8) hi = miscLoc2Loc_.GetValue(2 + k * 2);
            }
            PutSectorVal(k, lo, hi, st);
        }
        if (st) Ub2Gm<int32_t>(frNodesGm_, secStart, alignLoc_, eps * 4);
        else Ub2Gm<T>(idsGm_, secStart, alignLoc_.template ReinterpretCast<T>(), eps * (int32_t)sizeof(T));
    }

    // staging 读原语：读含目标元素的 32B 扇区（跨核 MTE 读须 32B 对齐，4B 跨核读实测不可靠，
    // 穿刺④约束：跨核共享数据读一律 MTE 读回，且以 32B 扇区为单位），再从 UB 取目标 4B。
    __aicore__ inline int32_t ReadStgFr(int32_t j, int64_t o)
    {
        int64_t abs = (int64_t)j * CH_ + o;
        int64_t secBase = (abs / 8) * 8;                 // 32B 扇区起始（8 int32）
        Gm2Ub<int32_t>(metaLoc_, stgFrGm_, secBase, 32);
        return metaLoc_.GetValue((int32_t)(abs - secBase));
    }
    __aicore__ inline int32_t ReadStgIds32(int32_t j, int64_t o)
    {
        int64_t abs = (int64_t)j * CH_ + o;
        int64_t secBase = (abs / 8) * 8;
        Gm2Ub<int32_t>(metaLoc_, stgIds32Gm_, secBase, 32);
        return metaLoc_.GetValue((int32_t)(abs - secBase));
    }
    __aicore__ inline int64_t ReadStgIds64(int32_t j, int64_t o, bool hi)
    {
        int64_t abs = (int64_t)j * CH_ * 2 + o * 2 + (hi ? 1 : 0);
        int64_t secBase = (abs / 8) * 8;
        Gm2Ub<int32_t>(metaLoc_, stgIds32Gm_, secBase, 32);
        return (int64_t)(uint32_t)metaLoc_.GetValue((int32_t)(abs - secBase));
    }

    // 按层内输出位次 p 读 staging 值：线性扫 cntConfT 前缀定位 (chunk j, rank m)，再读 chunk 槽
    __aicore__ inline void ReadStgByPos(int64_t p, int32_t numChunks, bool st, int32_t &lo, int32_t &hi)
    {
        lo = 0; hi = 0;
        int64_t acc = 0;
        int32_t j = 0;
        int64_t m = 0;
        for (int32_t c = 0; c < numChunks; ++c) {
            Gm2Ub<int32_t>(miscLoc_, cntConfTGm_, (int64_t)c * rowStride_, rowStride_ * 4);
            int64_t totJ = 0;
            for (int32_t d = 0; d < C_; ++d) totJ += miscLoc_.GetValue(d);
            if (p < acc + totJ) { j = c; m = p - acc; break; }
            acc += totJ;
        }
        if (st) lo = ReadStgFr(j, m);
        else if (sizeof(T) == 8) { lo = (int32_t)ReadStgIds64(j, m, false); hi = (int32_t)ReadStgIds64(j, m, true); }
        else lo = (int32_t)ReadStgIds32(j, m);
    }

    // ------------------------- 成员 -------------------------
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> ipGm_, idxGm_, eidGm_, srcGm_, idsGm_;
    AscendC::GlobalTensor<int64_t> secGm_;
    AscendC::GlobalTensor<int8_t> wsGm_;
    AscendC::GlobalTensor<int32_t> visGm_, syncGmT_, errGm_, segGm_;
    AscendC::GlobalTensor<int32_t> cntDomGm_, cntConfDGm_, cntConfTGm_;
    AscendC::GlobalTensor<int32_t> candGm_, confGm_, frNodesGm_;
    AscendC::GlobalTensor<int32_t> fragIdsGm_, fragFrGm_;
    AscendC::GlobalTensor<T> stgIdsGm_;
    AscendC::GlobalTensor<int32_t> stgIds32Gm_, stgFrGm_;
    AscendC::GlobalTensor<int64_t> countGm_;   // [集成计数槽] ws 尾部计数槽（idsLen/secLen）

    AscendC::TBuf<AscendC::TPosition::VECCALC> syncUb_, miscUb_, misc2Ub_, alignUb_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lineIdxUb_, lineEidUb_, frontWinUb_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> candStageUb_, confStageUb_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sliceUb_, outUb_, transUb_, metaUb_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> k0IpUb_, k0IdxUb_, k0EidUb_, k0VisUb_;

    AscendC::LocalTensor<int32_t> syncUbLoc_, miscLoc_, miscLoc2Loc_, alignLoc_;
    AscendC::LocalTensor<int32_t> frontWinLoc_, candStageLoc_, confStageLoc_;
    AscendC::LocalTensor<int32_t> sliceLoc_, outLoc_, transLoc_, metaLoc_;
    AscendC::LocalTensor<T> lineIdxLoc_, lineEidLoc_;
    AscendC::LocalTensor<T> k0IpLoc_, k0IdxLoc_, k0EidLoc_;
    AscendC::LocalTensor<int32_t> k0VisLoc_;

    BfsWs wsBfs_;
    int64_t N_ = 0, NNZ_ = 0, S_ = 0;
    int64_t hasEid_ = 0;
    int32_t key_ = 2, C_ = 1, blk_ = 0, CH_ = 1024, rowStride_ = 1, rowStrideJ_ = 1, numChunksMax_ = 1;
    int64_t domLo_ = 0, domHi_ = 0;
    int32_t curLayer_ = 0;       // 当前层号（1-based，源层=0）；frag 双槽 parity 用
    int32_t lastWrParity_ = 0;   // 最后一层尾扇区写 frag 的槽 parity（退出 FlushPending 读用）
    int64_t countOff_ = -1;      // [集成计数槽] ws 内计数槽偏移（-1 不写）
    int64_t idsOutLen_ = 0;      // [集成计数槽] 输出 ids 精确长度（host 回读截断用）
    int64_t secOutLen_ = 0;      // [集成计数槽] 输出 sections 精确长度
};

}  // namespace bfscore
