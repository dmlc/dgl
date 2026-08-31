#include <dgl/array.h>
#include <dgl/runtime/device_api.h>
#include "../kernel_decl.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <dmlc/logging.h>
#include <cstdint>

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
#define ASCEND_CALL(func)                                                \
  {                                                                      \
    aclError e = (func);                                                 \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e; \
  }

constexpr uint32_t kCubeCoreCount = 20;   // 910B3 cube cores
constexpr uint32_t kSegMmBlockLen = 16;
constexpr uint32_t kSegMmC0Float = 8;       // 32B / sizeof(float)
constexpr uint32_t kSegMmMaxKTile = 64;
constexpr uint32_t kSegMmMaxNTile = 64;

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_segment_mm(
    uint32_t blockDim, aclrtStream stream,
    void* aData, void* bData, void* cData,
    uint32_t m, uint32_t k, uint32_t n,
    uint32_t mGrid, uint32_t nGrid,
    uint32_t nTile, uint32_t kTile, uint32_t bTrans);

extern "C" uint32_t aclrtlaunch_segment_mm_backward_b(
    uint32_t blockDim, aclrtStream stream,
    void* aData, void* dCData, void* dBData,
    uint32_t k, uint32_t n, uint32_t m,
    uint32_t mGrid, uint32_t nGrid,
    uint32_t nTile, uint32_t kTile);

// Tiling for a single etype's GEMM [m, k, n]. Computes a 2D (M, N) core grid
// with mGrid*nGrid <= cubeCoreCount, and per-tile sizes (16/8 aligned) capped
// to safe L0A/L0B/L0C capacities.
static void ComputeSegMmTiling(uint32_t m, uint32_t n, uint32_t k,
                               uint32_t &mGrid, uint32_t &nGrid,
                               uint32_t &nTile, uint32_t &blockDim) {
  uint32_t mBlocks = (m + kSegMmBlockLen - 1) / kSegMmBlockLen;
  // Choose mGrid: cover M in 16-blocks, up to cubeCoreCount.
  uint32_t mg = mBlocks;
  if (mg == 0) mg = 1;
  if (mg > kCubeCoreCount) mg = kCubeCoreCount;
  // nGrid: fill remaining cores (at least 1).
  uint32_t ng = kCubeCoreCount / mg;
  if (ng == 0) ng = 1;
  // Cap nGrid so each core handles at least one 16-col block when possible.
  uint32_t nBlocks = (n + kSegMmBlockLen - 1) / kSegMmBlockLen;
  if (ng > nBlocks && nBlocks > 0) ng = nBlocks;
  mGrid = mg;
  nGrid = ng;
  blockDim = mGrid * nGrid;
  if (blockDim == 0) blockDim = 1;

  // nTile: per-Mmad N chunk, 16-aligned, capped for L0B/L0C capacity.
  uint32_t nt = kSegMmMaxNTile;
  uint32_t nPerCore = (n + nGrid - 1) / nGrid;
  if (nt > nPerCore) nt = nPerCore;
  nt = (nt / kSegMmBlockLen) * kSegMmBlockLen;
  if (nt == 0) nt = kSegMmBlockLen;
  nTile = nt;
}

static uint32_t ComputeSegMmKTile(uint32_t k) {
  uint32_t kt = k;
  if (kt > kSegMmMaxKTile) kt = kSegMmMaxKTile;
  kt = (kt / kSegMmC0Float) * kSegMmC0Float;   // 8-aligned (float K alignment)
  if (kt == 0) kt = kSegMmC0Float;
  return kt;
}

template <typename IdType, typename DType>
static void SegmentMMAscend(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans) {
  CHECK(!a_trans) << "SegmentMMAscend does not support a_trans";
  DGLContext ctx = A->ctx;
  CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  const IdType* seglen = seglen_A.Ptr<IdType>();
  const DType* A_data = A.Ptr<DType>();
  const DType* B_data = B.Ptr<DType>();
  DType* C_data = C.Ptr<DType>();

  int64_t num_rel = seglen_A.NumElements();
  int64_t k = B->shape[1];
  int64_t n = B->shape[2];
  if (b_trans) {
    // B is [R, N, K]: cols of A (==K) == B.shape[2].
    k = B->shape[2];
    n = B->shape[1];
  }

  aclrtStream stream = nullptr;

  int64_t a_offset = 0;   // element offset into A (rows accumulate)
  int64_t b_offset = 0;   // element offset into B (per etype B[r])
  int64_t c_offset = 0;   // element offset into C

  for (IdType etype = 0; etype < num_rel; ++etype) {
    uint32_t m = static_cast<uint32_t>(seglen[etype]);
    uint32_t ku = static_cast<uint32_t>(k);
    uint32_t nu = static_cast<uint32_t>(n);
    if (m == 0 || ku == 0 || nu == 0) {
      // Skipped etype: product is the empty/zero GEMM. The kernel is not
      // launched, so the corresponding C slice [m, n] at c_offset must be
      // zeroed explicitly (callers pass torch.empty outputs).
      size_t c_bytes = static_cast<size_t>(m) * nu * sizeof(DType);
      if (c_bytes > 0) {
        ASCEND_CALL(aclrtMemset(C_data + c_offset, c_bytes, 0, c_bytes));
      }
      a_offset += static_cast<int64_t>(seglen[etype]) * k;
      b_offset += static_cast<int64_t>(k) * n;
      c_offset += static_cast<int64_t>(seglen[etype]) * n;
      continue;
    }

    uint32_t mGrid, nGrid, nTile, blockDim;
    ComputeSegMmTiling(m, nu, ku, mGrid, nGrid, nTile, blockDim);
    uint32_t kTile = ComputeSegMmKTile(ku);

    void* a_ptr = const_cast<void*>(static_cast<const void*>(A_data + a_offset));
    void* b_ptr = const_cast<void*>(static_cast<const void*>(B_data + b_offset));
    void* c_ptr = static_cast<void*>(C_data + c_offset);

    aclError err = ACLRT_LAUNCH_KERNEL(segment_mm)(
        blockDim, stream, a_ptr, b_ptr, c_ptr,
        m, ku, nu, mGrid, nGrid, nTile, kTile,
        b_trans ? 1u : 0u);
    if (err != ACL_SUCCESS) {
      LOG(FATAL) << "segment_mm kernel launch failed (etype=" << etype
                 << ") with error code: " << err;
    }
    ASCEND_CALL(aclrtSynchronizeStream(stream));

    a_offset += static_cast<int64_t>(seglen[etype]) * k;
    b_offset += static_cast<int64_t>(k) * n;
    c_offset += static_cast<int64_t>(seglen[etype]) * n;
  }
}

template <typename IdType, typename DType>
static void SegmentMMBackwardBAscend(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen) {
  DGLContext ctx = A->ctx;
  CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  const IdType* seglen_data = seglen.Ptr<IdType>();
  const DType* A_data = A.Ptr<DType>();
  const DType* dC_data = dC.Ptr<DType>();
  DType* dB_data = dB.Ptr<DType>();

  int64_t num_rel = seglen.NumElements();
  int64_t n = A->shape[1];   // dB rows
  int64_t m = dC->shape[1];  // dB cols

  aclrtStream stream = nullptr;

  int64_t a_offset = 0;
  int64_t dC_offset = 0;
  int64_t dB_offset = 0;

  for (IdType etype = 0; etype < num_rel; ++etype) {
    uint32_t k = static_cast<uint32_t>(seglen_data[etype]);
    uint32_t nu = static_cast<uint32_t>(n);
    uint32_t mu = static_cast<uint32_t>(m);
    if (k == 0 || nu == 0 || mu == 0) {
      // Skipped etype: dB[r] = A_seg^T @ dC_seg with empty K is the zero
      // [n, m] matrix. The kernel is not launched, so the dB[r] slice must be
      // zeroed explicitly (callers pass torch.empty outputs).
      size_t db_bytes = static_cast<size_t>(nu) * mu * sizeof(DType);
      if (db_bytes > 0) {
        ASCEND_CALL(aclrtMemset(dB_data + dB_offset, db_bytes, 0, db_bytes));
      }
      a_offset += static_cast<int64_t>(n) * seglen_data[etype];
      dC_offset += static_cast<int64_t>(m) * seglen_data[etype];
      dB_offset += static_cast<int64_t>(n) * m;
      continue;
    }

    // GEMM: outM=n, outN=m, innerK=k. Tile on (outM, outN) = (n, m).
    uint32_t mGrid, nGrid, nTile, blockDim;
    ComputeSegMmTiling(nu, mu, k, mGrid, nGrid, nTile, blockDim);
    uint32_t kTile = ComputeSegMmKTile(k);

    void* a_ptr = const_cast<void*>(static_cast<const void*>(A_data + a_offset));
    void* dC_ptr = const_cast<void*>(static_cast<const void*>(dC_data + dC_offset));
    void* dB_ptr = static_cast<void*>(dB_data + dB_offset);

    aclError err = ACLRT_LAUNCH_KERNEL(segment_mm_backward_b)(
        blockDim, stream, a_ptr, dC_ptr, dB_ptr,
        k, nu, mu, mGrid, nGrid, nTile, kTile);
    if (err != ACL_SUCCESS) {
      LOG(FATAL) << "segment_mm_backward_b kernel launch failed (etype=" << etype
                 << ") with error code: " << err;
    }
    ASCEND_CALL(aclrtSynchronizeStream(stream));

    a_offset += static_cast<int64_t>(n) * seglen_data[etype];
    dC_offset += static_cast<int64_t>(m) * seglen_data[etype];
    dB_offset += static_cast<int64_t>(n) * m;
  }
}

#endif  // DGL_USE_ASCEND

namespace dgl {
namespace aten {

template <int XPU, typename IdType, typename DType>
void SegmentMM(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans);

template <int XPU, typename IdType, typename DType>
void SegmentMMBackwardB(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);

#ifdef DGL_USE_ASCEND

// ---- SegmentMM Ascend specializations (int32/int64 x float32) ----
template <>
void SegmentMM<kDGLAscend, int32_t, float>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans) {
  SegmentMMAscend<int32_t, float>(A, B, C, seglen_A, a_trans, b_trans);
}

template <>
void SegmentMM<kDGLAscend, int64_t, float>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans) {
  SegmentMMAscend<int64_t, float>(A, B, C, seglen_A, a_trans, b_trans);
}

// half / double: not supported on Ascend in this version (design §1).
template <>
void SegmentMM<kDGLAscend, int32_t, uint16_t>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans) {
  LOG(FATAL) << "SegmentMM on Ascend: half precision not supported yet.";
}

template <>
void SegmentMM<kDGLAscend, int64_t, uint16_t>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans) {
  LOG(FATAL) << "SegmentMM on Ascend: half precision not supported yet.";
}

template <>
void SegmentMM<kDGLAscend, int32_t, double>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans) {
  LOG(FATAL) << "SegmentMM on Ascend: double precision not supported yet.";
}

template <>
void SegmentMM<kDGLAscend, int64_t, double>(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans) {
  LOG(FATAL) << "SegmentMM on Ascend: double precision not supported yet.";
}

// ---- SegmentMMBackwardB Ascend specializations ----
template <>
void SegmentMMBackwardB<kDGLAscend, int32_t, float>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen) {
  SegmentMMBackwardBAscend<int32_t, float>(A, dC, dB, seglen);
}

template <>
void SegmentMMBackwardB<kDGLAscend, int64_t, float>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen) {
  SegmentMMBackwardBAscend<int64_t, float>(A, dC, dB, seglen);
}

template <>
void SegmentMMBackwardB<kDGLAscend, int32_t, uint16_t>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen) {
  LOG(FATAL) << "SegmentMMBackwardB on Ascend: half precision not supported yet.";
}

template <>
void SegmentMMBackwardB<kDGLAscend, int64_t, uint16_t>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen) {
  LOG(FATAL) << "SegmentMMBackwardB on Ascend: half precision not supported yet.";
}

template <>
void SegmentMMBackwardB<kDGLAscend, int32_t, double>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen) {
  LOG(FATAL) << "SegmentMMBackwardB on Ascend: double precision not supported yet.";
}

template <>
void SegmentMMBackwardB<kDGLAscend, int64_t, double>(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen) {
  LOG(FATAL) << "SegmentMMBackwardB on Ascend: double precision not supported yet.";
}

#endif  // DGL_USE_ASCEND

}  // namespace aten
}  // namespace dgl
