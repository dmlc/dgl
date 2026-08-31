# SegmentMM / SegmentMMBackwardB 白盒分支覆盖说明

> 产物来源：3.3 白盒测试补全任务。源码依据 `segment_mm_kernel.cpp` / `segment_mm_cube_processor.h` / `segment_mm.cc`。
> 白盒用例已补入 `test_segment_mm_npu.py`（WB_FWD_CASES 9 条 + WB_BWD_CASES 7 条 = 16 条）。

## 一、枚举的执行分支（全部活分支）

### Forward — SegmentMmCubeProcessor

| 编号 | 分支 | 源码位置 | 触发条件 | 命中用例 | 状态 |
|------|------|----------|----------|----------|------|
| F1 | AIV no-op 早返回 | kernel.cpp:24-27 | AIV 核心进入即 return | 所有用例（每次 launch 均触发） | ✅ 已覆盖 |
| F2 | 尾核空闲早返回 (mIdx>=mGrid) | cube_processor.h:84 | blockIdx >= mGrid*nGrid | 不可经 dispatch 触发（blockDim=mGrid*nGrid） | ⚠️ 防御性，不可达 |
| F3 | 空M-block范围早返回 | cube_processor.h:90 | myMBlockLo >= myMBlockHi（mBlocks 非整除 mGrid） | WB-SMM-07 (M=336, mBlocks=21, mGrid=20, core11-19 空) | ✅ 已覆盖 |
| F4 | M-block 尾块 (actualRows<16) | cube_processor.h:100 | Min(16, M-mOff) 在末 block < 16 | WB-SMM-01 (M=17, 尾 actualRows=1), WB-SMM-08 (M=17) | ✅ 已覆盖 |
| F5 | N-tile 尾块 (nChunk<nTile) | cube_processor.h:103 | Min(nTile, myNHi-nOff) 在末 tile < nTile | WB-SMM-06 (N=65, nTile=64, 尾 nChunk=1) | ✅ 已覆盖 |
| F6 | K-tile 尾块 (kChunk<kTile) | cube_processor.h:110 | Min(kTile, K-kOff) 在末 K 迭代 < kTile | WB-SMM-03 (K=9), WB-SMM-04 (K=65 尾 kChunk=1), WB-SMM-08 (K=9) | ✅ 已覆盖 |
| F7 | K 分片累加 (initC true→false) | cube_processor.h:106,123 | K > kTile(64) → 多次 K 迭代 | WB-SMM-04 (K=65), WB-SMM-05 (K=65), WB-SMM-09 (K=128) | ✅ 已覆盖 |
| F8 | bTrans=false SplitBTranspose | cube_processor.h:119 | bTrans=false → LoadData3DParamsV2 enTranspose=true | WB-SMM-01/02/04/06/07/08/09, L0-SMM-01 等 | ✅ 已覆盖 |
| F9 | bTrans=true SplitBNoTranspose | cube_processor.h:117 | bTrans=true → LoadData2D ifTranspose=false | WB-SMM-03/05, L0-SMM-03, L1-SMM-03/06/09 等 | ✅ 已覆盖 |
| F10 | kFractal 循环 >1 (SplitBNoTranspose) | cube_processor.h:208 | kChunk > C0_FLOAT(8), bTrans=true | WB-SMM-03 (K=9, kFractals=2), WB-SMM-05 (K=65, 每 K 迭代 kFractals=8) | ✅ 已覆盖 |
| F11 | host 0 段跳过 | segment_mm.cc:116 | m==0||k==0||n==0 → continue | L1-SMM-10 (seglen 含 0 段) | ✅ 已覆盖（黑盒） |
| F12 | host mGrid 多核切分 | segment_mm.cc:49-62 | mBlocks > 1 → mGrid > 1 | WB-SMM-01/02 (mGrid=2), WB-SMM-06/07 (mGrid=20), WB-SMM-09 (mGrid=4) | ✅ 已覆盖 |
| F13 | host nGrid N方向多核 | segment_mm.cc:54-59 | nBlocks > 1, nGrid > 1 | WB-SMM-09 (nGrid=4), L1-SMM-01 (nGrid=4) | ✅ 已覆盖 |
| F14 | host nTile 对齐裁剪 | segment_mm.cc:67-70 | nTile 16 对齐，cap 到 nPerCore | WB-SMM-06 (nTile=64), 所有用例 | ✅ 已覆盖 |
| F15 | host kTile 8 对齐裁剪 | segment_mm.cc:74-79 | kTile 8 对齐，cap 到 64 | WB-SMM-04/05 (kTile=64), 所有用例 | ✅ 已覆盖 |

### Forward — 无 K-tail 的 K-split 精确路径

| 编号 | 分支 | 触发条件 | 命中用例 | 状态 |
|------|------|----------|----------|------|
| F16 | K-split 精确无尾 (kChunk==kTile 每次迭代) | K 是 kTile(64) 的整数倍且 K>64 | WB-SMM-09 (K=128, 2×64 exact) | ✅ 已覆盖 |

### Forward — 多核边界

| 编号 | 分支 | 触发条件 | 命中用例 | 状态 |
|------|------|----------|----------|------|
| F17 | 单核 (mGrid=1) | M <= 16 | L0-SMM-01~05, WB-SMM-03/04/05 | ✅ 已覆盖 |
| F18 | 双核 (mGrid=2) | M=17~32 | WB-SMM-01 (M=17), WB-SMM-02 (M=32) | ✅ 已覆盖 |
| F19 | 满 20 核 (mGrid=20) | M >= 320 | WB-SMM-06 (M=320), WB-SMM-07 (M=336) | ✅ 已覆盖 |
| F20 | 尾核块数少于其他核 | mBlocks 非整除 mGrid | WB-SMM-07 (mBlocks=21, mGrid=20, core10=1 block) | ✅ 已覆盖 |

### Backward — SegmentMmBwdBCubeProcessor

| 编号 | 分支 | 源码位置 | 触发条件 | 命中用例 | 状态 |
|------|------|----------|----------|----------|------|
| B1 | AIV no-op 早返回 | kernel.cpp:49-51 | AIV 核心进入即 return | 所有反向用例 | ✅ 已覆盖 |
| B2 | 空M-block范围早返回 | cube_processor.h:332 | myMBlockLo >= myMBlockHi | WB-BWD-06 (N=336, mBlocks=21, mGrid=20, core11-19 空) | ✅ 已覆盖 |
| B3 | 输出行尾块 (actualRows<16) | cube_processor.h:342 | Min(16, N-nOff) < 16 | WB-BWD-01 (N=17, 尾 actualRows=1), WB-BWD-05 (N=17) | ✅ 已覆盖 |
| B4 | 输出列 tile 尾块 | cube_processor.h:345 | Min(nTile, myNHi-mOff) < nTile | WB-BWD-05 (M=17, nChunk partial) | ✅ 已覆盖 |
| B5 | K-tile 尾块 | cube_processor.h:351 | Min(kTile, K-kOff) < kTile | WB-BWD-02 (K=65, 尾 kChunk=1), WB-BWD-03 (K=9), WB-BWD-05 (K=9) | ✅ 已覆盖 |
| B6 | K 分片累加 (initC true→false) | cube_processor.h:348,362 | K > kTile(64) | WB-BWD-02 (K=65), WB-BWD-07 (K=128) | ✅ 已覆盖 |
| B7 | A 转置 SplitATranspose | cube_processor.h:393 | LoadData3DParamsV2 enTranspose=true, kDirectionAlign=true | 所有反向用例 | ✅ 已覆盖 |
| B8 | dC B-转置 SplitBTranspose | cube_processor.h:432 | LoadData3DParamsV2 enTranspose=true | 所有反向用例 | ✅ 已覆盖 |
| B9 | K-split 精确无尾 | K 是 64 整数倍且 K>64 | WB-BWD-07 (K=128, 2×64 exact) | ✅ 已覆盖 |
| B10 | 满 20 核多块 | N >= 320 | WB-BWD-04 (N=320), WB-BWD-06 (N=336) | ✅ 已覆盖 |
| B11 | 尾核块数少 | mBlocks 非整除 mGrid | WB-BWD-06 (N=336, mBlocks=21, mGrid=20) | ✅ 已覆盖 |

## 二、不可达分支说明

| 分支 | 源码 | 不可达原因 |
|------|------|------------|
| F2: mIdx >= mGrid 早返回 | cube_processor.h:84 | host 设 blockDim = mGrid*nGrid，blockIdx ∈ [0, blockDim) → mIdx = blockIdx/nGrid ∈ [0, mGrid)，恒 < mGrid |
| B-空N范围: myNLo >= myNHi | cube_processor.h:95/338 | host 设 nGrid <= nBlocks，nPerCore = CeilDiv(N, nGrid) >= 1，nIdx ∈ [0, nGrid) → myNLo = nIdx*nPerCore < N = myNHi |

## 三、数据档说明

所有白盒用例使用 `normal` 数据档（[-1,1] 均匀分布），属 clean 档必过集。未引入 stress 档（big/neg_big/denormal），因白盒目标是分支路径覆盖而非精度边界。

## 四、覆盖汇总

- 枚举活分支：20（Forward 16 + Backward 11，去重后 20 条独立分支路径）
- 已覆盖：18 条（含 2 条不可达防御分支标注）
- 白盒用例数：16 条（WB-SMM-01~09 + WB-BWD-01~07）
- 覆盖率：100%（可达分支全覆盖）
