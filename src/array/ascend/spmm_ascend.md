# Ascend SpMM 算子实现技术说明

> 本文说明 `dgl-ascend` 中 CSR 格式 SpMM 的实现架构，重点覆盖 CPU/NPU 分工、传参与关键技术点。

---

## 1. 算子概述

支持 `op = copy_lhs`，`reduce = sum / mean / max / min`，图格式为 CSR。

```
out[i, :] = Reduce( feat[j, :] ),  j ∈ N(i)
```

执行策略：

- `sum / mean`：**AIC + AIV 混合路径**，CPU 按窗口密度分流
- `max / min`：**AIV 纯向量路径**，CPU 按行 nnz 负载均衡

---

## 2. 总体架构

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         DGL / Python 调用层                         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   CPU 入口：SpMMCsrAscend (spmm.cc)                 │
│                                                                     │
│  1. 参数校验 + 解析 CSR/Feature 规模                                │
│  2. 查询预处理缓存                                                  │
│  3. 缓存未命中 → 根据 reduce 构造预处理元数据                       │
│  4. 预处理完成后 launch NPU Kernel                                  │
└───────────────┬─────────────────────────────────────┬───────────────┘
                │                                     │
                │  sum/mean                           │  max / min
                ▼                                     ▼
┌─────────────────────────────────┐     ┌──────────────────────────────┐
│  CPU 预处理：窗口化 + 分类      │     │  CPU 预处理：按 row nnz 切分 │
│                                 │     │                              │
│  - 16 行窗口划分                │     │  - 拷回 indptr               │
│  - unique column 提取           │     │  - 统计每行 nnz              │
│  - density 计算                 │     │  - 生成 vector_row_split     │
│  - 高密度 → AIC                 │     └──────────────┬───────────────┘
│  - 低密度 → AIV                 │                    │
└───────────────┬─────────────────┘                    │
                └────────────────────┬─────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           NPU Kernel                                │
│                                                                     │
│  sum/sum:  AIV(稀疏窗口 gather+Add)  +  AIC(稠密窗口 block GEMM)    │
│  max:  AIV(gather + Max)                                            │
│  min:  AIV(gather + Min)                                            │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
                          output 写回
```

**分工一句话**：CPU 做图分析和任务重写，NPU 做数据搬运和片上计算。

---

## 3. CPU 做了什么

CPU 的核心价值是把不规则 CSR 翻译成 NPU 友好的执行计划：

- **图结构预处理**：窗口化、稠密块编码、列映射表构造
- **负载均衡**：按实际工作量（nnz / tc_blocks）分配核心，避免长尾空转
- **缓存复用**：图结构不变时跳过预处理，直接复用已有元数据
- **运行时集成**：异步拷贝、kernel launch、同步控制

---

## 4. CPU 传给 NPU 的参数

### `max / min` 路径

```text
featureData         输入特征矩阵
outputData          输出矩阵
indptrData          CSR 行指针
indicesData         CSR 列索引
vectorRowSplitData  CPU 按 row nnz 负载均衡后的行切分边界
numDstRows / numSrcRows / featureDim / nonZeroCount
```

### `sum / mean` 路径

除上述原始图/特征外，还有 CPU 构造的执行计划：


| 参数组    | 内容                                                                                                  |
| ------ | --------------------------------------------------------------------------------------------------- |
| AIC 任务 | `denseBlockData`(压缩稠密块), `cubeWindowIdsData`, `cubeWinSplitData`, `winEdgePtrData`, `colToEdgeData` |
| AIV 任务 | `vectorWindowIdsData`, `vectorWinSplitData`                                                         |
| 规模信息   | `totalTcBlocks`, `vectorWindowCount`, `cubeWindowCount`, `columnToEdgeLength`                       |


> `sum` 传给 NPU 的不仅是数据，还有"任务怎么分、块怎么取、列怎么映射"。

---

## 5. `max / min` 路径详解

**CPU 流程**：拷回 indptr → 统计每行 nnz → 按 nnz 负载均衡 → 生成 `vector_row_split` → launch kernel

**NPU 流程**（全部 AIV）：

```text
┌──────────────────────────────────────────────────────┐
│  AIV Core                                            │
│  1. 读 vectorRowSplitData，确定本核 row 范围         │
│  2. 逐行：读 indptr 定位邻居 → 分批搬入 UB          │
│     → 逐元素 Max/Min → 写回 output[row, :]          │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 6. `sum / mean` 路径详解

### 6.1 混合执行的核心思想

对 16 行窗口，若行间大量共享源点列，可压缩为小块矩阵乘：

```
Y(16×N) = A(16×K') × X(K'×N)
```

- `A`：窗口内 0/1 邻接块（CPU 预构造）
- `X`：窗口涉及的唯一源点特征子矩阵

高密度窗口交 AIC 做 GEMM，低密度窗口交 AIV 逐行累加。

```text
                    SUM 路径混合执行

        CSR + Feature
              │
              ▼
   ┌─────────────────────────┐
   │  CPU: 窗口分析 + 分类    │
   └────────────┬─────────────┘
         ┌──────┴──────┐
         ▼             ▼
  ┌────────────┐ ┌──────────────┐
  │ 稀疏窗口   │ │ 高密度窗口   │
  │ → AIV      │ │ → AIC        │
  │ gather+Add │ │ block GEMM   │
  └─────┬──────┘ └──────┬───────┘
        └───────┬────────┘
                ▼
         output 合并写回
```

### 6.2 CPU 窗口分类逻辑

按 16 行切窗 → 提取 unique columns → 计算 density → 按密度排序取 top-k 为 cube windows，其余为 vector windows。

### 6.3 CPU 为 AIC 构造的数据

- `dense_blocks`：16×K' 的 0/1 稠密块（列按 16 对齐）
- `cube_window_ids`：哪些窗口交给 AIC
- `win_edge_ptr` + `column_to_edge`：窗口唯一列到源点特征行的映射

### 6.4 NPU 执行

**AIV**：按 `vectorWinSplitData` 分工 → 遍历窗口内每行 → gather 邻居特征 → Add → 写回

**AIC**：按 `cubeWinSplitData` 分工 → 读 dense block + gather 特征行 → 数据重排 → Mmad 矩阵乘 → Fixpipe 转 half 写回

AIC 由于片上 L0A/L0B/L0C 容量有限，会沿 K 和 N 两个维度做切片（tiling）。

---

## 7. 技术点总结


| 层面  | 技术点                                                                                |
| --- | ---------------------------------------------------------------------------------- |
| CPU | 图窗口化、密度分析、异构核心映射、按工作量负载均衡、预处理缓存                                                    |
| AIV | DataCopyPad 对齐搬运、UB 批量缓存、向量 Add/Max/Min 规约、双缓冲流水                                   |
| AIC | 窗口稠密化 block GEMM、Nd2Nz/LoadData 布局变换、Mmad Tensor Cube、Fixpipe 类型转换写回、K/N 双维 tiling |


---

## 8. BSpMM（Batched SpMM）

BSpMM 处理 3D 输入特征，即每个节点有多个 batch 的特征向量。与 SpMM 共享同一套 CPU 预处理逻辑和缓存，差异仅体现在 NPU 侧的数据寻址方式。

### 8.1 与 SpMM 的关系


| 对比项     | SpMM                  | BSpMM                             |
| ------- | --------------------- | --------------------------------- |
| 输入形状    | `[nodes, featureDim]` | `[nodes, batchCount, featureDim]` |
| CPU 预处理 | 相同                    | 相同（复用同一份缓存）                       |
| NPU 执行  | 每行处理 1 次              | 每行处理 `batchCount` 次               |
| 额外参数    | 无                     | `batchCount`                      |


CPU 侧完全一致——窗口划分、负载均衡、稠密块构造都不涉及 batch 维度。NPU 侧在原有逻辑外层多套一个 `batchIndex` 循环。

### 8.2 NPU 侧差异

**寻址变化**：特征布局从 `feat[node * N]` 变为 `feat[(node * batchCount + batchIndex) * featureDim]`，输出同理。

**AIV 路径**：对每一行，遍历 `batchCount` 次，每次 gather 对应 batch 的邻居特征并规约。

**AIC 路径**：对每个窗口，遍历 `batchCount` 次，每次从特征矩阵中按 `batchIndex` 偏移 gather 对应列，执行相同的块矩阵乘。`dense_blocks`（邻接结构）跨 batch 共享，无需重复加载。

### 8.3 BSpMM 传参

在 SpMM 参数基础上，仅增加一个标量：

```text
batchCount    batch 维度大小
```

其余参数含义和布局与 SpMM 完全相同。