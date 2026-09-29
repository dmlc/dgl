// BFSNodesFrontiers Ascend host 特化（aten 派发）
//
// tiling 计算为纯 host C++ 整数运算 + acl API（A 类红线合规：无 torch 计算/搬运），
// 算式与 kernel 侧共用 `bfs_ws_layout.h`（防漂移）。
//
// 流程：输入校验 → tiling 计算 → workspace 分配（aclrtMalloc wsBytes+16）→
// syncGm/errflag 同流预清零（aclrtMemsetAsync）→ launch（直启）→
// 同步 + errflag 回读校验 → 计数槽回读 → 精确长度 IdArray + D2D 截断 → Frontiers。
#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
#define ASCEND_CALL(func)                                                \
  {                                                                      \
    aclError e = (func);                                                 \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e;              \
  }

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

// 直启入口 host stub（与 kernel 入口符号 bfs_nodes_frontiers 匹配，8 个 GM_ADDR）
extern "C" uint32_t aclrtlaunch_bfs_nodes_frontiers(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* indices, void* eid, void* source,
    void* ids, void* sections, void* ws, void* tiling);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}
}  // namespace dgl
#endif

#include <dgl/array.h>
#include <dgl/aten/csr.h>
#include <dgl/runtime/device_api.h>

#include "../array_op.h"
#include "bfs_ws_layout.h"

#include <vector>

namespace dgl {
namespace aten {
namespace impl {

#ifdef DGL_USE_ASCEND

namespace {

// host 输入校验
// array.cc 已校验 graph/source 同 device、indices/source 同 dtype、CSR 方阵）
void BfsHostCheck(const CSRMatrix& csr, IdArray source, bool is_edge) {
  CHECK(csr.indptr->ndim == 1 && csr.indices->ndim == 1 && source->ndim == 1)
      << "bfs_nodes_frontiers: all inputs must be 1-D";
  CHECK(csr.indptr->shape[0] >= 1)
      << "bfs_nodes_frontiers: indptr.numel() must be >= 1";
  CHECK(csr.indptr->dtype == csr.indices->dtype)
      << "bfs_nodes_frontiers: indptr and indices must share one dtype";
  const int64_t numRows = csr.num_rows;
  const int64_t nnz = csr.indices->shape[0];
  CHECK(numRows < (int64_t{1} << 31) && nnz < (int64_t{1} << 31))
      << "bfs_nodes_frontiers: numRows/nnz must be < 2^31 "
         "(int32 去重键实现边界)";
  const int64_t S = source->shape[0];
  CHECK(!(numRows == 0 && S > 0))
      << "bfs_nodes_frontiers: source out of range (empty graph num_rows==0 "
         "but source non-empty)";
  if (is_edge && CSRHasData(csr)) {
    CHECK(csr.data->dtype == csr.indices->dtype)
        << "bfs_nodes_frontiers: eid must share one dtype with indices";
    CHECK(csr.data->ndim == 1 && csr.data->shape[0] == nnz)
        << "bfs_nodes_frontiers: eid.numel() must equal indices.numel()";
  }
}

// tiling 计算
// 核数用 aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)，失败回退 8）
void CalcBfsTiling(int64_t numRows, int64_t nnz, int64_t S, int64_t hasEid,
                   int64_t elemSize, int32_t device_id,
                   bfscore::BfsTilingData* td) {
  int64_t ubBytes =
      (numRows + 1) * elemSize + nnz * elemSize * (1 + hasEid) + numRows * 4;
  int32_t chunkSize = (elemSize == 4) ? 1024 : 512;
  int64_t numChunksMax = (nnz + chunkSize - 1) / chunkSize;
  if (numChunksMax < 1) numChunksMax = 1;
  int64_t coreMax = 8;
  aclError ret = aclrtGetDeviceInfo(static_cast<uint32_t>(device_id),
                                    ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreMax);
  if (ret != ACL_SUCCESS || coreMax <= 0) coreMax = 8;

  int64_t key, numBlocks;
  if (ubBytes <= bfscore::UB_KEY0_BUDGET) {
    key = 0;  // 单核 UB 常驻（快路径）
    numBlocks = 1;
  } else if (numRows >= bfscore::KEY1_MIN_ROWS && nnz < 2 * numRows) {
    key = 1;  // 单核 GM 标量（深图启发式）
    numBlocks = 1;
  } else {
    key = 2;  // 多核分层（kernel 内 BFS_ENABLE_KEY2_MULTICORE=false 时退避串行）
    numBlocks = (nnz + 1023) / 1024;
    if (numBlocks < 1) numBlocks = 1;
    if (numBlocks > coreMax) numBlocks = coreMax;
  }
  td->numRows = numRows;
  td->nnz = nnz;
  td->S = S;
  td->hasEid = hasEid;
  td->tilingKey = key;
  td->numBlocks = numBlocks;
  td->numChunksMax = numChunksMax;
  td->dtype_bytes = static_cast<uint32_t>(elemSize);
}

// 持久化 tiling buffer：线程 + 设备双键缓存（分配一次、复用，消除 per-call 的
// malloc/free/sync）；切换设备时先释放旧设备 buffer 再在新设备重分配，
// 析构兜底线程退出时回收设备内存（避免跨设备复用 GM 指针与设备内存泄漏）
struct TilingDevBuf {
  void* ptr = nullptr;
  int32_t device_id = -1;

  // 析构不得抛出异常：非抛出释放 + 日志（review 反馈）
  ~TilingDevBuf() {
    if (ptr != nullptr) {
      aclError e = aclrtFree(ptr);
      ptr = nullptr;
      if (e != ACL_SUCCESS) {
        LOG(WARNING) << "TilingDevBuf destructor: aclrtFree failed, code: " << e;
      }
    }
  }

  void* Get(const bfscore::BfsTilingData& tiling, int32_t device_id) {
    if (ptr != nullptr && device_id != this->device_id) {
      ASCEND_CALL(aclrtFree(ptr));
      ptr = nullptr;
    }
    if (ptr == nullptr) {
      ASCEND_CALL(aclrtMalloc(&ptr, sizeof(bfscore::BfsTilingData),
                              ACL_MEM_MALLOC_HUGE_FIRST));
      this->device_id = device_id;
    }
    ASCEND_CALL(aclrtMemcpy(ptr, sizeof(bfscore::BfsTilingData), &tiling,
                            sizeof(bfscore::BfsTilingData),
                            ACL_MEMCPY_HOST_TO_DEVICE));
    return ptr;
  }

  TilingDevBuf() = default;
  TilingDevBuf(const TilingDevBuf&) = delete;
  TilingDevBuf& operator=(const TilingDevBuf&) = delete;
};

static thread_local TilingDevBuf tiling_dev_buf;

void* GetTilingDev(const bfscore::BfsTilingData& tiling, int32_t device_id) {
  return tiling_dev_buf.Get(tiling, device_id);
}

// 设备内存 RAII 守卫：析构兜底 aclrtFree，覆盖 launch/CHECK 抛异常路径。
// 析构函数不得抛出异常，故析构走非抛出释放路径。
struct DeviceMemGuard {
  void* ptr = nullptr;

  ~DeviceMemGuard() { FreeNoThrow(); }

  void Allocate(size_t size) {
    Free();  // 防重入：先释放已有指针
    ASCEND_CALL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
  }

  void* get() const { return ptr; }

  void Free() {
    if (ptr != nullptr) {
      ASCEND_CALL(aclrtFree(ptr));
      ptr = nullptr;
    }
  }

  void FreeNoThrow() {
    if (ptr != nullptr) {
      aclError e = aclrtFree(ptr);
      ptr = nullptr;
      if (e != ACL_SUCCESS) {
        LOG(WARNING) << "DeviceMemGuard destructor: aclrtFree failed, code: " << e;
      }
    }
  }

  DeviceMemGuard() = default;
  DeviceMemGuard(const DeviceMemGuard&) = delete;
  DeviceMemGuard& operator=(const DeviceMemGuard&) = delete;
};

template <typename IdType>
Frontiers BFSNodesFrontiersAscend(const CSRMatrix& csr, IdArray source) {
  const DGLContext ctx = source->ctx;
  CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context";
  const int64_t numRows = csr.num_rows;
  const int64_t nnz = csr.indices->shape[0];
  const int64_t S = source->shape[0];
  const int64_t elemSize = sizeof(IdType);

  BfsHostCheck(csr, source, /*is_edge=*/false);

  // 早退：空源 / 空图 → 空输出，不 launch
  if (S == 0 || numRows == 0) {
    Frontiers front;
    front.ids = IdArray::Empty({0}, DGLDataType{kDGLInt, sizeof(IdType) * 8, 1}, ctx);
    front.sections = IdArray::Empty({0}, DGLDataType{kDGLInt, 64, 1}, ctx);
    front.tags = NullArray();
    return front;
  }

  ASCEND_CALL(aclrtSetDevice(ctx.device_id));
  auto stream = dgl::runtime::getCurrentAscendStream();

  // 输出上界缓冲
  const int64_t idsLenUb = S + numRows;
  const int64_t secLenUb = numRows + 1;
  IdArray ids_ub =
      IdArray::Empty({idsLenUb}, DGLDataType{kDGLInt, sizeof(IdType) * 8, 1}, ctx);
  IdArray secs_ub = IdArray::Empty({secLenUb}, DGLDataType{kDGLInt, 64, 1}, ctx);

  // tiling（nodes 版 hasEid 恒 0，eid 传 nullptr）
  const int64_t hasEid = 0;
  bfscore::BfsTilingData td;
  CalcBfsTiling(numRows, nnz, S, hasEid, elemSize, ctx.device_id, &td);
  td.idsLenUb = idsLenUb;
  td.secLenUb = secLenUb;
  const int64_t wsBytes = bfscore::HsWorkspaceLayout(
      numRows, nnz, (int32_t)td.numBlocks, (int32_t)td.numChunksMax, elemSize)
                              .total;
  td.countOff = wsBytes;  // ws 尾部追加 16B 计数槽（host 分配 wsBytes+16）

  // workspace：per-call 分配；DeviceMemGuard 兜底异常路径自动释放
  DeviceMemGuard ws;
  ws.Allocate(wsBytes + 16);
  const bfscore::BfsWs wsLayout = bfscore::HsWorkspaceLayout(
      numRows, nnz, (int32_t)td.numBlocks, (int32_t)td.numChunksMax, elemSize);
  // syncGm 区预清零：SoftSyncAll 屏障计数器必须 host 同流预清零（独立工程结论；
  // FIFO 保证 memset → kernel 顺序）
  ASCEND_CALL(aclrtMemsetAsync((char*)ws.get() + wsLayout.syncGm,
                               td.numBlocks * 32, 0, td.numBlocks * 32, stream));
  // errflag 区同流清零
  ASCEND_CALL(aclrtMemsetAsync((char*)ws.get() + wsLayout.errflag,
                               td.numBlocks * 32, 0, td.numBlocks * 32, stream));

  void* tiling_dev = GetTilingDev(td, ctx.device_id);

  // launch（nodes：eid=nullptr，hasEid=0）
  aclError err = aclrtlaunch_bfs_nodes_frontiers(
      static_cast<uint32_t>(td.numBlocks), stream, csr.indptr->data,
      csr.indices->data, nullptr, source->data, ids_ub->data, secs_ub->data,
      ws.get(), tiling_dev);
  CHECK(err == ACL_SUCCESS) << "bfs_nodes_frontiers launch failed: " << err;

  // 同步 + errflag 回读
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  const int64_t errBytes = td.numBlocks * 32;
  std::vector<int32_t> errflags(td.numBlocks * 8, 0);
  ASCEND_CALL(aclrtMemcpy(errflags.data(), errBytes,
                          (char*)ws.get() + wsLayout.errflag, errBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
  for (int64_t c = 0; c < td.numBlocks; ++c) {
    int32_t cls = errflags[c * 8];
    CHECK(cls == 0)
        << "bfs_nodes_frontiers: device-side validation failed, class=" << cls
        << " (1: indptr non-monotonic, 2: indptr[num_rows]!=nnz, "
           "3: source out of range, 4: indices out of range, 5: O1 iter guard)";
  }

  // 计数槽回读（idsLen/secLen 精确长度）
  int64_t counts[2] = {0, 0};
  ASCEND_CALL(aclrtMemcpy(counts, 16, (char*)ws.get() + td.countOff, 16,
                          ACL_MEMCPY_DEVICE_TO_HOST));
  const int64_t idsLen = counts[0], secLen = counts[1];
  CHECK(idsLen >= 0 && idsLen <= idsLenUb && secLen >= 0 && secLen <= secLenUb)
      << "bfs_nodes_frontiers: output length exceeds upper bound (idsLen="
      << idsLen << ", secLen=" << secLen << ")";

  // 精确长度输出 + D2D 截断
  IdArray ids =
      IdArray::Empty({idsLen}, DGLDataType{kDGLInt, sizeof(IdType) * 8, 1}, ctx);
  IdArray secs = IdArray::Empty({secLen}, DGLDataType{kDGLInt, 64, 1}, ctx);
  if (idsLen > 0) {
    ASCEND_CALL(aclrtMemcpy(ids->data, idsLen * sizeof(IdType), ids_ub->data,
                            idsLen * sizeof(IdType), ACL_MEMCPY_DEVICE_TO_DEVICE));
  }
  if (secLen > 0) {
    ASCEND_CALL(aclrtMemcpy(secs->data, secLen * sizeof(int64_t), secs_ub->data,
                            secLen * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_DEVICE));
  }

  // 同步已做，workspace 可安全释放（guard 析构仍兜底异常路径）
  ws.Free();

  Frontiers front;
  front.ids = ids;
  front.sections = secs;
  front.tags = NullArray();
  return front;
}

}  // namespace

template <>
Frontiers BFSNodesFrontiers<kDGLAscend, int32_t>(const CSRMatrix& csr,
                                                 IdArray source) {
  return BFSNodesFrontiersAscend<int32_t>(csr, source);
}

template <>
Frontiers BFSNodesFrontiers<kDGLAscend, int64_t>(const CSRMatrix& csr,
                                                 IdArray source) {
  return BFSNodesFrontiersAscend<int64_t>(csr, source);
}

#else  // DGL_USE_ASCEND

template <>
Frontiers BFSNodesFrontiers<kDGLAscend, int32_t>(const CSRMatrix& csr,
                                                 IdArray source) {
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
}

template <>
Frontiers BFSNodesFrontiers<kDGLAscend, int64_t>(const CSRMatrix& csr,
                                                 IdArray source) {
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
}

#endif  // DGL_USE_ASCEND

}  // namespace impl
}  // namespace aten
}  // namespace dgl
