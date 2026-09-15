#include <dgl/array.h>
#include <dgl/aten/array_ops.h>
#include <dgl/runtime/device_api.h>
#include "../kernel_decl.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <dmlc/logging.h>
#include <cstdint>

#include <cstring>

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
#define ASCEND_CALL(func)                                                \
  {                                                                      \
    aclError e = (func);                                                 \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e; \
  }

constexpr uint32_t kCubeCoreCount = 20;
constexpr uint32_t kGatherMmBlockLen = 16;
constexpr uint32_t kGatherMmC0Float = 8;
constexpr uint32_t kGatherMmMaxKTile = 64;
constexpr uint32_t kGatherMmMaxNTile = 64;

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_gather_mm_k1(
    uint32_t blockDim, aclrtStream stream,
    void* aData, void* bData, void* cWsData,
    void* seglenData, void* groupOffsetsData,
    void* idxAPermData, void* idxBGroupData,
    uint32_t m, uint32_t k, uint32_t n,
    uint32_t numGroups, uint32_t nTile, uint32_t kTile);

extern "C" uint32_t aclrtlaunch_gather_mm_scatter_aiv(
    uint32_t blockDim, aclrtStream stream,
    void* cWsData, void* cData,
    void* scatterIdxData,
    uint32_t m, uint32_t n, uint32_t isScatter);

extern "C" uint32_t aclrtlaunch_gather_mm_k2(
    uint32_t blockDim, aclrtStream stream,
    void* aData, void* bData, void* cData,
    void* seglenData, void* groupOffsetsData,
    void* idxAPermData, void* idxBPermData, void* idxCGroupData,
    uint32_t m, uint32_t k, uint32_t n,
    uint32_t numGroups, uint32_t nTile);

// Reuse SegmentMM tiling logic.
static void ComputeGatherMmTiling(uint32_t m, uint32_t n, uint32_t k,
                                  uint32_t &nTile, uint32_t &blockDim) {
  uint32_t nt = kGatherMmMaxNTile;
  if (nt > n) nt = n;
  nt = (nt / kGatherMmBlockLen) * kGatherMmBlockLen;
  if (nt == 0) nt = kGatherMmBlockLen;
  nTile = nt;
  blockDim = kCubeCoreCount;
}

static uint32_t ComputeGatherMmKTile(uint32_t k) {
  uint32_t kt = k;
  if (kt > kGatherMmMaxKTile) kt = kGatherMmMaxKTile;
  kt = (kt / kGatherMmC0Float) * kGatherMmC0Float;
  if (kt == 0) kt = kGatherMmC0Float;
  return kt;
}

// ---------------------------------------------------------------------------
// Host-side grouping: sort rows by group key, build seglen / group_offsets /
// permuted index arrays.
// ---------------------------------------------------------------------------

// Group entry: (key, original_row). Sorted by key to form contiguous groups.
struct GroupEntry {
  uint32_t key;
  uint32_t row;
};

static bool GroupEntryCmp(const GroupEntry &a, const GroupEntry &b) {
  return a.key < b.key;
}

// Build grouping metadata on host.
// Inputs: keys[m] (group key per row), idx_a[m], idx_b[m] (A/B row indices).
// Outputs: seglen[numGroups], groupOffsets[numGroups], rowPerm[m],
//          idxAPerm[m], keyGroup[numGroups] (the key value per group).
static void BuildGrouping(const std::vector<uint32_t> &keys,
                          const std::vector<uint32_t> &idx_a,
                          const std::vector<uint32_t> &idx_b,
                          std::vector<uint32_t> &seglen,
                          std::vector<uint32_t> &groupOffsets,
                          std::vector<uint32_t> &rowPerm,
                          std::vector<uint32_t> &idxAPerm,
                          std::vector<uint32_t> &idxBPerm,
                          std::vector<uint32_t> &keyGroup,
                          uint32_t &numGroups) {
  uint32_t m = static_cast<uint32_t>(keys.size());
  std::vector<GroupEntry> entries(m);
  for (uint32_t i = 0; i < m; ++i) {
    entries[i].key = keys[i];
    entries[i].row = i;
  }
  // Stable sort by key.
  std::stable_sort(entries.begin(), entries.end(), GroupEntryCmp);

  rowPerm.resize(m);
  idxAPerm.resize(m);
  if (!idx_b.empty()) idxBPerm.resize(m);

  seglen.clear();
  groupOffsets.clear();
  keyGroup.clear();
  uint32_t offset = 0;
  numGroups = 0;
  uint32_t i = 0;
  while (i < m) {
    uint32_t curKey = entries[i].key;
    uint32_t j = i;
    while (j < m && entries[j].key == curKey) ++j;
    uint32_t groupSize = j - i;
    seglen.push_back(groupSize);
    groupOffsets.push_back(offset);
    keyGroup.push_back(curKey);
    for (uint32_t k = i; k < j; ++k) {
      uint32_t origRow = entries[k].row;
      rowPerm[offset] = origRow;
      idxAPerm[offset] = idx_a[origRow];
      if (!idx_b.empty()) idxBPerm[offset] = idx_b[origRow];
      ++offset;
    }
    numGroups++;
    i = j;
  }
}

// Copy host vector to device NDArray (int32).
static NDArray HostVecToDevice(const std::vector<uint32_t> &vec,
                               const DGLContext &ctx) {
  DGLContext cpu_ctx{kDGLCPU, 0};
  int64_t size = static_cast<int64_t>(vec.size());
  NDArray cpu_arr = NDArray::Empty({size}, DGLDataType{kDGLInt, 32, 1}, cpu_ctx);
  if (size > 0) {
    std::memcpy(cpu_arr.Ptr<uint32_t>(), vec.data(),
                static_cast<size_t>(size) * sizeof(uint32_t));
  }
  return cpu_arr.CopyTo(ctx);
}

// ===========================================================================
// GatherMMAscend: C[r,:] = A[idx_a[r],:] @ B[idx_b[r],:,:]  (B 3D)
// ===========================================================================
template <typename IdType, typename DType>
static void GatherMMAscend(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b) {
  DGLContext ctx = A->ctx;
  CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  int64_t m = A->shape[0];
  int64_t k = A->shape[1];
  int64_t n = B->shape[2];

  if (m == 0) {
    // Empty output: zero C.
    size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);
    if (c_bytes > 0) {
      ASCEND_CALL(aclrtMemsetAsync(C.Ptr<void>(), c_bytes, 0, c_bytes, nullptr));
      ASCEND_CALL(aclrtSynchronizeStream(nullptr));
    }
    return;
  }

  aclrtStream stream = nullptr;

  // 1. D2H copy idx_a / idx_b (or generate arange if Null).
  DGLContext cpu_ctx{kDGLCPU, 0};
  std::vector<uint32_t> idx_a_host(m), idx_b_host(m);

  if (dgl::aten::IsNullArray(idx_a)) {
    for (int64_t i = 0; i < m; ++i) idx_a_host[i] = static_cast<uint32_t>(i);
  } else {
    NDArray idx_a_cpu = idx_a.CopyTo(cpu_ctx);
    const IdType *pa = idx_a_cpu.Ptr<IdType>();
    for (int64_t i = 0; i < m; ++i) idx_a_host[i] = static_cast<uint32_t>(pa[i]);
  }

  if (dgl::aten::IsNullArray(idx_b)) {
    for (int64_t i = 0; i < m; ++i) idx_b_host[i] = static_cast<uint32_t>(i);
  } else {
    NDArray idx_b_cpu = idx_b.CopyTo(cpu_ctx);
    const IdType *pb = idx_b_cpu.Ptr<IdType>();
    for (int64_t i = 0; i < m; ++i) idx_b_host[i] = static_cast<uint32_t>(pb[i]);
  }

  // 2. Group by idx_b.
  std::vector<uint32_t> keys(m);
  for (int64_t i = 0; i < m; ++i) keys[i] = idx_b_host[i];

  std::vector<uint32_t> seglen, groupOffsets, rowPerm, idxAPerm, idxBPerm, idxBGroup;
  uint32_t numGroups;
  // idx_b_perm not needed for K1 (B is loaded by idx_b_group, not per-row).
  std::vector<uint32_t> dummyBPerm;
  BuildGrouping(keys, idx_a_host, idx_b_host,
                seglen, groupOffsets, rowPerm, idxAPerm, dummyBPerm, idxBGroup,
                numGroups);

  // 3. H2D copy metadata to device.
  NDArray seglen_dev = HostVecToDevice(seglen, ctx);
  NDArray group_offsets_dev = HostVecToDevice(groupOffsets, ctx);
  NDArray idx_a_perm_dev = HostVecToDevice(idxAPerm, ctx);
  NDArray idx_b_group_dev = HostVecToDevice(idxBGroup, ctx);
  NDArray row_perm_dev = HostVecToDevice(rowPerm, ctx);

  // 4. Allocate C_ws workspace (group-sorted, contiguous [m, n]).
  NDArray c_ws = NDArray::Empty({m, n}, A->dtype, ctx);
  size_t ws_bytes = static_cast<size_t>(m) * n * sizeof(float);
  if (ws_bytes > 0) {
    ASCEND_CALL(aclrtMemsetAsync(c_ws.Ptr<void>(), ws_bytes, 0, ws_bytes, stream));
  }

  // 5. Compute tiling.
  uint32_t nTile, blockDimCube;
  ComputeGatherMmTiling(static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                        static_cast<uint32_t>(k), nTile, blockDimCube);
  uint32_t kTile = ComputeGatherMmKTile(static_cast<uint32_t>(k));
  uint32_t blockDim = std::min(blockDimCube, numGroups);
  if (blockDim == 0) blockDim = 1;

  // 6. Launch K1 kernel (AIC Cube GEMM -> C_ws).
  void* a_ptr = A.Ptr<void>();
  void* b_ptr = B.Ptr<void>();
  void* cws_ptr = c_ws.Ptr<void>();
  void* seglen_ptr = seglen_dev.Ptr<void>();
  void* group_offsets_ptr = group_offsets_dev.Ptr<void>();
  void* idx_a_perm_ptr = idx_a_perm_dev.Ptr<void>();
  void* idx_b_group_ptr = idx_b_group_dev.Ptr<void>();

  aclError err = ACLRT_LAUNCH_KERNEL(gather_mm_k1)(
      blockDim, stream, a_ptr, b_ptr, cws_ptr,
      seglen_ptr, group_offsets_ptr, idx_a_perm_ptr, idx_b_group_ptr,
      static_cast<uint32_t>(m), static_cast<uint32_t>(k), static_cast<uint32_t>(n),
      numGroups, nTile, kTile);
  if (err != ACL_SUCCESS) {
    LOG(FATAL) << "gather_mm_k1 kernel launch failed: " << err;
  }
  ASCEND_CALL(aclrtSynchronizeStream(stream));

  // 7. AIV scatter: C_ws[i] -> C[rowPerm[i]] (non-atomic, unique targets).
  uint32_t aivBlockDim = std::min(kCubeCoreCount, static_cast<uint32_t>(m));
  if (aivBlockDim == 0) aivBlockDim = 1;
  void* c_ptr = C.Ptr<void>();
  void* row_perm_ptr = row_perm_dev.Ptr<void>();

  err = ACLRT_LAUNCH_KERNEL(gather_mm_scatter_aiv)(
      aivBlockDim, stream, cws_ptr, c_ptr, row_perm_ptr,
      static_cast<uint32_t>(m), static_cast<uint32_t>(n), 0u);
  if (err != ACL_SUCCESS) {
    LOG(FATAL) << "gather_mm_scatter_aiv kernel launch failed: " << err;
  }
  ASCEND_CALL(aclrtSynchronizeStream(stream));
}

//   Kernel1 (B 3D): C[idx_c[r],:] += A[idx_a[r],:] @ B[idx_b[r],:,:]
//   Kernel2 (B 2D): C[idx_c[r],i,j] += A[idx_a[r],i] * B[idx_b[r],j]
// ===========================================================================
template <typename IdType, typename DType>
static void GatherMMScatterAscend(
    const NDArray A, const NDArray B, NDArray C,
    const NDArray idx_a, const NDArray idx_b, const NDArray idx_c) {
  DGLContext ctx = A->ctx;
  CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  int64_t m = A->shape[0];
  int64_t k = A->shape[1];
  bool b_is_2d = (B->ndim == 2);
  int64_t n = b_is_2d ? B->shape[1] : B->shape[2];

  // Handle idx_c Null: degrade to GatherMM semantics (idx_c[r] = r).
  bool idx_c_null = dgl::aten::IsNullArray(idx_c);

  if (m == 0) {
    return;  // Nothing to do.
  }

  aclrtStream stream = nullptr;

  // Zero C (required for scatter accumulation).
  size_t c_bytes = C.GetSize();
  if (c_bytes > 0) {
    ASCEND_CALL(aclrtMemsetAsync(C.Ptr<void>(), c_bytes, 0, c_bytes, stream));
  }

  // D2H copy idx arrays.
  DGLContext cpu_ctx{kDGLCPU, 0};
  std::vector<uint32_t> idx_a_host(m), idx_b_host(m), idx_c_host(m);

  if (dgl::aten::IsNullArray(idx_a)) {
    for (int64_t i = 0; i < m; ++i) idx_a_host[i] = static_cast<uint32_t>(i);
  } else {
    NDArray idx_a_cpu = idx_a.CopyTo(cpu_ctx);
    const IdType *pa = idx_a_cpu.Ptr<IdType>();
    for (int64_t i = 0; i < m; ++i) idx_a_host[i] = static_cast<uint32_t>(pa[i]);
  }

  if (dgl::aten::IsNullArray(idx_b)) {
    for (int64_t i = 0; i < m; ++i) idx_b_host[i] = static_cast<uint32_t>(i);
  } else {
    NDArray idx_b_cpu = idx_b.CopyTo(cpu_ctx);
    const IdType *pb = idx_b_cpu.Ptr<IdType>();
    for (int64_t i = 0; i < m; ++i) idx_b_host[i] = static_cast<uint32_t>(pb[i]);
  }

  if (idx_c_null) {
    for (int64_t i = 0; i < m; ++i) idx_c_host[i] = static_cast<uint32_t>(i);
  } else {
    NDArray idx_c_cpu = idx_c.CopyTo(cpu_ctx);
    const IdType *pc = idx_c_cpu.Ptr<IdType>();
    for (int64_t i = 0; i < m; ++i) idx_c_host[i] = static_cast<uint32_t>(pc[i]);
  }

  if (!b_is_2d) {
    // ---- Kernel1 (B 3D): group by idx_b ----
    std::vector<uint32_t> keys(m);
    for (int64_t i = 0; i < m; ++i) keys[i] = idx_b_host[i];

    std::vector<uint32_t> seglen, groupOffsets, rowPerm, idxAPerm, idxBPerm, idxBGroup;
    uint32_t numGroups;
    std::vector<uint32_t> dummyBPerm;
    BuildGrouping(keys, idx_a_host, idx_b_host,
                  seglen, groupOffsets, rowPerm, idxAPerm, dummyBPerm, idxBGroup,
                  numGroups);

    NDArray seglen_dev = HostVecToDevice(seglen, ctx);
    NDArray group_offsets_dev = HostVecToDevice(groupOffsets, ctx);
    NDArray idx_a_perm_dev = HostVecToDevice(idxAPerm, ctx);
    NDArray idx_b_group_dev = HostVecToDevice(idxBGroup, ctx);

    // Build scatterIdx: scatterIdx[i] = idx_c[rowPerm[i]].
    std::vector<uint32_t> scatterIdx(m);
    for (int64_t i = 0; i < m; ++i) {
      scatterIdx[i] = idx_c_host[rowPerm[i]];
    }
    NDArray scatter_idx_dev = HostVecToDevice(scatterIdx, ctx);

    // C_ws workspace.
    NDArray c_ws = NDArray::Empty({m, n}, A->dtype, ctx);
    size_t ws_bytes = static_cast<size_t>(m) * n * sizeof(float);
    if (ws_bytes > 0) {
      ASCEND_CALL(aclrtMemsetAsync(c_ws.Ptr<void>(), ws_bytes, 0, ws_bytes, stream));
    }

    uint32_t nTile, blockDimCube;
    ComputeGatherMmTiling(static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                          static_cast<uint32_t>(k), nTile, blockDimCube);
    uint32_t kTile = ComputeGatherMmKTile(static_cast<uint32_t>(k));
    uint32_t blockDim = std::min(blockDimCube, numGroups);
    if (blockDim == 0) blockDim = 1;

    void* a_ptr = A.Ptr<void>();
    void* b_ptr = B.Ptr<void>();
    void* cws_ptr = c_ws.Ptr<void>();
    void* c_ptr = C.Ptr<void>();

    aclError err = ACLRT_LAUNCH_KERNEL(gather_mm_k1)(
        blockDim, stream, a_ptr, b_ptr, cws_ptr,
        seglen_dev.Ptr<void>(), group_offsets_dev.Ptr<void>(),
        idx_a_perm_dev.Ptr<void>(), idx_b_group_dev.Ptr<void>(),
        static_cast<uint32_t>(m), static_cast<uint32_t>(k), static_cast<uint32_t>(n),
        numGroups, nTile, kTile);
    if (err != ACL_SUCCESS) {
      LOG(FATAL) << "gather_mm_k1 kernel launch failed: " << err;
    }
    ASCEND_CALL(aclrtSynchronizeStream(stream));

    // AIV scatter: C_ws[i] -> C[scatterIdx[i]] with atomic add.
    bool isScatter = !idx_c_null;
    uint32_t aivBlockDim = std::min(kCubeCoreCount, static_cast<uint32_t>(m));
    if (aivBlockDim == 0) aivBlockDim = 1;

    err = ACLRT_LAUNCH_KERNEL(gather_mm_scatter_aiv)(
        aivBlockDim, stream, cws_ptr, c_ptr, scatter_idx_dev.Ptr<void>(),
        static_cast<uint32_t>(m), static_cast<uint32_t>(n), isScatter ? 1u : 0u);
    if (err != ACL_SUCCESS) {
      LOG(FATAL) << "gather_mm_scatter_aiv kernel launch failed: " << err;
    }
    ASCEND_CALL(aclrtSynchronizeStream(stream));

  } else {
    // ---- Kernel2 (B 2D): group by idx_c ----
    std::vector<uint32_t> keys(m);
    for (int64_t i = 0; i < m; ++i) keys[i] = idx_c_host[i];

    std::vector<uint32_t> seglen, groupOffsets, rowPerm, idxAPerm, idxBPerm, idxCGroup;
    uint32_t numGroups;
    BuildGrouping(keys, idx_a_host, idx_b_host,
                  seglen, groupOffsets, rowPerm, idxAPerm, idxBPerm, idxCGroup,
                  numGroups);

    NDArray seglen_dev = HostVecToDevice(seglen, ctx);
    NDArray group_offsets_dev = HostVecToDevice(groupOffsets, ctx);
    NDArray idx_a_perm_dev = HostVecToDevice(idxAPerm, ctx);
    NDArray idx_b_perm_dev = HostVecToDevice(idxBPerm, ctx);
    NDArray idx_c_group_dev = HostVecToDevice(idxCGroup, ctx);

    uint32_t nTile, blockDimCube;
    ComputeGatherMmTiling(static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                          static_cast<uint32_t>(k), nTile, blockDimCube);
    uint32_t blockDim = std::min(blockDimCube, numGroups);
    if (blockDim == 0) blockDim = 1;

    void* a_ptr = A.Ptr<void>();
    void* b_ptr = B.Ptr<void>();
    void* c_ptr = C.Ptr<void>();

    aclError err = ACLRT_LAUNCH_KERNEL(gather_mm_k2)(
        blockDim, stream, a_ptr, b_ptr, c_ptr,
        seglen_dev.Ptr<void>(), group_offsets_dev.Ptr<void>(),
        idx_a_perm_dev.Ptr<void>(), idx_b_perm_dev.Ptr<void>(),
        idx_c_group_dev.Ptr<void>(),
        static_cast<uint32_t>(m), static_cast<uint32_t>(k), static_cast<uint32_t>(n),
        numGroups, nTile);
    if (err != ACL_SUCCESS) {
      LOG(FATAL) << "gather_mm_k2 kernel launch failed: " << err;
    }
    ASCEND_CALL(aclrtSynchronizeStream(stream));
  }
}

#endif  // DGL_USE_ASCEND

namespace dgl {
namespace aten {

template <int XPU, typename IdType, typename DType>
void GatherMM(
    const NDArray A, const NDArray B, NDArray out, const NDArray idx_a,
    const NDArray idx_b);

template <int XPU, typename IdType, typename DType>
void GatherMMScatter(
    const NDArray A, const NDArray B, NDArray out, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c);

#ifdef DGL_USE_ASCEND

// ---- GatherMM Ascend specializations (int32/int64 x float32) ----
template <>
void GatherMM<kDGLAscend, int32_t, float>(
    const NDArray A, const NDArray B, NDArray out, const NDArray idx_a,
    const NDArray idx_b) {
  GatherMMAscend<int32_t, float>(A, B, out, idx_a, idx_b);
}

template <>
void GatherMM<kDGLAscend, int64_t, float>(
    const NDArray A, const NDArray B, NDArray out, const NDArray idx_a,
    const NDArray idx_b) {
  GatherMMAscend<int64_t, float>(A, B, out, idx_a, idx_b);
}

// half / double: not supported on Ascend in this version.
template <>
void GatherMM<kDGLAscend, int32_t, uint16_t>(
    const NDArray A, const NDArray B, NDArray out, const NDArray idx_a,
    const NDArray idx_b) {
  LOG(FATAL) << "GatherMM on Ascend: half precision not supported yet.";
}

template <>
void GatherMM<kDGLAscend, int64_t, uint16_t>(
    const NDArray A, const NDArray B, NDArray out, const NDArray idx_a,
    const NDArray idx_b) {
  LOG(FATAL) << "GatherMM on Ascend: half precision not supported yet.";
}

template <>
void GatherMM<kDGLAscend, int32_t, double>(
    const NDArray A, const NDArray B, NDArray out, const NDArray idx_a,
    const NDArray idx_b) {
  LOG(FATAL) << "GatherMM on Ascend: double precision not supported yet.";
}

template <>
void GatherMM<kDGLAscend, int64_t, double>(
    const NDArray A, const NDArray B, NDArray out, const NDArray idx_a,
    const NDArray idx_b) {
  LOG(FATAL) << "GatherMM on Ascend: double precision not supported yet.";
}

// ---- GatherMMScatter Ascend specializations ----
template <>
void GatherMMScatter<kDGLAscend, int32_t, float>(
    const NDArray A, const NDArray B, NDArray out, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c) {
  GatherMMScatterAscend<int32_t, float>(A, B, out, idx_a, idx_b, idx_c);
}

template <>
void GatherMMScatter<kDGLAscend, int64_t, float>(
    const NDArray A, const NDArray B, NDArray out, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c) {
  GatherMMScatterAscend<int64_t, float>(A, B, out, idx_a, idx_b, idx_c);
}

template <>
void GatherMMScatter<kDGLAscend, int32_t, uint16_t>(
    const NDArray A, const NDArray B, NDArray out, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c) {
  LOG(FATAL) << "GatherMMScatter on Ascend: half precision not supported yet.";
}

template <>
void GatherMMScatter<kDGLAscend, int64_t, uint16_t>(
    const NDArray A, const NDArray B, NDArray out, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c) {
  LOG(FATAL) << "GatherMMScatter on Ascend: half precision not supported yet.";
}

template <>
void GatherMMScatter<kDGLAscend, int32_t, double>(
    const NDArray A, const NDArray B, NDArray out, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c) {
  LOG(FATAL) << "GatherMMScatter on Ascend: double precision not supported yet.";
}

template <>
void GatherMMScatter<kDGLAscend, int64_t, double>(
    const NDArray A, const NDArray B, NDArray out, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c) {
  LOG(FATAL) << "GatherMMScatter on Ascend: double precision not supported yet.";
}

#endif  // DGL_USE_ASCEND

}  // namespace aten
}  // namespace dgl
