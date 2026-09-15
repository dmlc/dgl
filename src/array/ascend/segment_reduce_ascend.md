# Ascend SegmentReduce 算子实现技术说明

> 本文说明 `dgl-ascend` 中 SegmentReduce 算子的实现架构，重点覆盖 CPU/NPU 分工、传参与关键技术点。

## 1. 算子概述

SegmentReduce 按段对特征矩阵做规约，输入是连续存储的特征和段偏移数组：

```
out[seg, :] = Reduce( feat[offsets[seg] : offsets[seg+1], :] )
```

支持 `reduce = sum / mean / max / min`，数据类型为 `float32`，全部走 **AIV 纯向量路径**。

与 SpMM 的区别：SpMM 基于 CSR 图结构做不规则 gather，SegmentReduce 的数据在内存中是连续的，因此不需要窗口化或稠密块构造，实现更为直接。

---

## 2. 总体架构

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         DGL / Python 调用层                         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              CPU 入口：SegmentReduceAscend (segment_reduce.cc)      │
│                                                                     │
│  1. 参数校验 + 解析规模                                             │
│  2. 按 offsets 做负载均衡，生成 segment_split                       │
│  3. 构造 tiling 参数                                                │
│  4. 传回 Device，launch NPU Kernel                                  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    NPU Kernel（AIV only）                           │
│                                                                     │
│  每个核处理一段 segments：                                          │
│    逐 segment 分批搬入连续特征 → 向量 Add/Max/Min → 写回            │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
                          output 写回
```

CPU 做负载均衡和参数打包，NPU 做连续数据搬运和向量规约。

---

## 3. CPU 做了什么

### 3.1 负载均衡（BuildSegmentSplit）

CPU 的核心工作是按段的 item 数做负载均衡：

1. 将 `offsets` 从 Device 拷回 Host
2. 以 `offsets` 末尾值（总 item 数）为总工作量
3. 对 `block_dim`（最多 40 核）做前缀和二分，找到每个核负责的 segment 区间边界
4. 生成 `segment_split[block_dim + 1]`

这一步和 SpMM 的 `BuildRowNnzBalancedPartitions` 思路一致：不按段数平均分，而按每段包含的 item 数近似均衡。

### 3.2 Tiling 参数构造

CPU 将三个标量打包为 `SegmentReduceTilingData`：

```text
numItems      总 item 数
numSegments   总段数
featDim       每个 item 的特征维度
```

### 3.3 传参与 Launch

CPU 将 `segment_split` 和 `tiling` 拷到 Device，然后 launch 对应 kernel。launch 后同步并释放临时 buffer。

---

## 4. CPU 传给 NPU 的参数

```text
offsets          段偏移数组（长度 numSegments+1）
feat             输入特征矩阵（连续存储）
output           输出矩阵
segment_split    CPU 构造的核心分工边界（长度 blockDim+1）
tiling           { numItems, numSegments, featDim }
```

参数非常简洁，因为数据本身连续，不需要列映射或窗口描述。

---

## 5. NPU 做了什么

四个 kernel（sum/mean/max/min）结构完全一致，仅规约算子不同。全部为 AIV 向量路径。

```text
┌──────────────────────────────────────────────────────────┐
│  AIV Core                                                │
│  1. 读 segment_split，确定本核负责的 segment 区间        │
│  2. 逐 segment：                                         │
│     - 读 offsets[seg], offsets[seg+1] 确定 item 范围     │
│     - 初始化累加器                                       │
│     - 分批搬入连续 item 特征到 UB（DataCopyPad）         │
│     - 逐批做向量 Add/Max/Min                             │
│     - 写回 output[seg, :]                                │
└──────────────────────────────────────────────────────────┘
```

### 初始值


| reduce | 空段输出 | 非空段累加器初始值  |
| ------ | ---- | ---------- |
| sum    | 0    | 0          |
| max    | 0    | `-FLT_MAX` |
| min    | 0    | `FLT_MAX`  |


### 与 SpMM AIV 路径的区别

SegmentReduce 的数据是连续的，所以 `CopyInBatch` 可以一次搬 `itemCount` 行（利用 `DataCopyPad` 的 `repeatCount`），而 SpMM 需要逐邻居 gather 不连续行。这让 SegmentReduce 的搬运效率更高。

---

## 6. 技术点总结


| 层面  | 技术点                                                               |
| --- | ----------------------------------------------------------------- |
| CPU | 按段 item 数做负载均衡（前缀和二分）、tiling 参数打包、临时 buffer 管理                    |
| NPU | DataCopyPad 连续批量搬运、32 字节对齐 + rightPadding、向量 Add/Max/Min 规约、双缓冲流水 |


---

## 7. 与 SpMM 的对比

```text
┌───────────────┬──────────────────────────────┬───────────────────────────┐
│ 对比项        │ SpMM                         │ SegmentReduce             │
├───────────────┼──────────────────────────────┼───────────────────────────┤
│ 数据访问模式  │ 不规则 gather（按 indices）  │ 连续访问（按 offsets）    │
│ CPU 预处理    │ 窗口化 + 稠密块 + 映射表     │ 仅负载均衡切分            │
│ NPU 核类型    │ AIC + AIV（sum）/ AIV only   │ AIV only                  │
│ 搬运方式      │ 逐邻居单行搬运               │ 批量连续搬运              │
│ 是否有缓存    │ 有（图结构不变时复用）       │ 无（每次重新构造）        │
│ 数据类型      │ half                         │ float                     │
└───────────────┴──────────────────────────────┴───────────────────────────┘
```

