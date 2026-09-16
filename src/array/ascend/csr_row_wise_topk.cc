/**
 * Copyright (c) 2024 by Contributors
 * @file csr_row_wise_topk.cc
 * @brief Ascend host launcher for CSR row-wise topk.
 *
 * Multi-core (v2 sampling skeleton): the host computes per-row pick counts
 * from row degrees in one fused pass fed by a bulk indptr D2H (pick counts
 * depend only on k and degree, never on weights), builds nnz-balanced
 * row-range partitions plus per-block output offsets as prefix sums,
 * allocates the output exactly, and launches the AIV kernel. Blocks write
 * disjoint output ranges.
 *
 * Weight normalization: the hardware sort network is float-only, so
 * int32/int64/float64 weights are normalized to float32 on the host with a
 * min-offset shift (double arithmetic) that keeps every value f32-exact
 * whenever the row-global range is below 2^24. float32 weights pass
 * through unchanged.
 */

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
#define ASCEND_CALL(func)                                   \
  {                                                         \
    aclError e = (func);                                    \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e; \
  }

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_csr_row_wise_topk_int32(
    uint32_t blockDim, aclrtStream stream, void* indptr, void* indices,
    void* data, void* rows, void* weight, void* out_rows, void* out_cols,
    void* out_idxs, void* row_split, void* out_starts, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_row_wise_topk_int64(
    uint32_t blockDim, aclrtStream stream, void* indptr, void* indices,
    void* data, void* rows, void* weight, void* out_rows, void* out_cols,
    void* out_idxs, void* row_split, void* out_starts, void* tiling);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}  // namespace runtime
}  // namespace dgl
#endif  // DGL_USE_ASCEND

#include <dgl/array.h>
#include <dgl/aten/array_ops.h>
#include <dgl/aten/csr.h>
#include <dgl/runtime/device_api.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "../array_op.h"
#include "csr_row_wise_topk_tiling.h"

namespace dgl {
namespace aten {
namespace impl {

#ifdef DGL_USE_ASCEND
namespace {

// Returns the device's vector-core count, queried at runtime so the
// launch adapts to any SoC (910B family: 40 AIV; other families or
// trimmed vNPU instances differ). Falls back to the arch default when
// the query is unavailable.
uint32_t QueryVectorCoreCount(int device_id) {
  int64_t core_num = 0;
  aclError err =
      aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &core_num);
  if (err != ACL_SUCCESS || core_num <= 0 || core_num > 4096) {
    return kDefaultVectorCoreCount;
  }
  return static_cast<uint32_t>(core_num);
}

// Returns the per-vector-core unified-buffer budget in bytes (minus the
// runtime-reserved tail). Queried at runtime because UB size differs
// across SoCs (192KB on 910B, 248KB on 950PR).
uint32_t QueryUbAvailableBytes(int device_id) {
  int64_t ub_bytes = 0;
  aclError err = aclrtGetDeviceInfo(
      device_id, ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE, &ub_bytes);
  if (err != ACL_SUCCESS ||
      ub_bytes <= static_cast<int64_t>(kUbReservedBytes) ||
      ub_bytes > (1 << 30)) {
    return kDefaultUbBytes - kUbReservedBytes;
  }
  // Reaching here means ub_bytes > kUbReservedBytes (checked above).
  return static_cast<uint32_t>(ub_bytes - kUbReservedBytes);
}

// Builds nnz-balanced partitions over row weights (spmm precedent):
// returns num_parts+1 boundaries so each part covers a contiguous row
// range with roughly equal total weight.
std::vector<uint32_t> BuildBalancedPartitions(
    const std::vector<uint32_t>& weights, uint32_t num_parts) {
  std::vector<uint32_t> boundaries(num_parts + 1, 0);
  if (num_parts == 0) return boundaries;

  const uint32_t item_count = static_cast<uint32_t>(weights.size());
  boundaries[num_parts] = item_count;
  if (item_count == 0) return boundaries;

  if (item_count <= num_parts) {
    for (uint32_t i = 0; i <= item_count; ++i) boundaries[i] = i;
    for (uint32_t i = item_count + 1; i <= num_parts; ++i)
      boundaries[i] = item_count;
    return boundaries;
  }

  std::vector<double> prefix(item_count + 1, 0.0);
  for (uint32_t i = 0; i < item_count; ++i)
    prefix[i + 1] = prefix[i] + weights[i];
  const double total_weight = prefix[item_count];
  if (total_weight <= 0.0) {
    for (uint32_t part = 1; part < num_parts; ++part)
      boundaries[part] = part * item_count / num_parts;
    return boundaries;
  }

  for (uint32_t part = 1; part < num_parts; ++part) {
    const double target = total_weight * part / num_parts;
    auto it = std::lower_bound(prefix.begin(), prefix.end(), target);
    boundaries[part] = static_cast<uint32_t>(it - prefix.begin());
  }
  // Enforce monotonicity against ties landing on the same boundary.
  for (uint32_t part = 1; part < num_parts; ++part) {
    if (boundaries[part] < boundaries[part - 1])
      boundaries[part] = boundaries[part - 1];
    if (boundaries[part] > item_count) boundaries[part] = item_count;
  }
  return boundaries;
}

// Enqueues an async H2D of a host uint32 table on the launch stream.
// Callers batch several uploads and synchronize ONCE before the stack
// buffers go out of scope (UploadSync): an async copy only captures the
// source pointer, so the sync must happen before the owning frame returns.
void* UploadHostUInt32(const std::vector<uint32_t>& host, aclrtStream stream) {
  void* dev = nullptr;
  ASCEND_CALL(aclrtMalloc(
      &dev, host.size() * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
  aclError e = aclrtMemcpyAsync(
      dev, host.size() * sizeof(uint32_t), host.data(),
      host.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE, stream);
  if (e != ACL_SUCCESS) {
    // Free the allocation before failing loudly — the raw handle is
    // only freed by the caller after the launch path completes, so a
    // failed copy would leak it (review finding, PR #42).
    aclrtFree(dev);
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e;
  }
  return dev;
}

// Normalizes a non-f32 weight array to float32 on the host with a
// min-offset shift: every value is mapped into [0, range], which is
// f32-exact whenever range < 2^24 (integer inputs) and recovers the
// relative spacing of large-base float64 clusters. The shift never
// changes the in-row ordering — topk compares within rows only.
// Returns the input untouched when it is already float32.
NDArray NormalizeWeightToF32(NDArray weight) {
  const auto& dt = weight->dtype;
  if (dt.code == kDGLFloat && dt.bits == 32) return weight;

  const int64_t n = weight->shape[0];
  DGLContext cpu_ctx{kDGLCPU, 0};
  DGLDataType f32{kDGLFloat, 32, 1};
  NDArray weight_cpu = weight.CopyTo(cpu_ctx);
  NDArray weight_f32 = NDArray::Empty({n}, f32, cpu_ctx);
  float* dst = static_cast<float*>(weight_f32->data);

  if (dt.code == kDGLInt && dt.bits == 32) {
    const int32_t* src = static_cast<const int32_t*>(weight_cpu->data);
    int32_t mn = src[0];
    for (int64_t i = 1; i < n; ++i) mn = std::min(mn, src[i]);
    for (int64_t i = 0; i < n; ++i)
      dst[i] = static_cast<float>(static_cast<double>(src[i]) - mn);
  } else if (dt.code == kDGLInt && dt.bits == 64) {
    const int64_t* src = static_cast<const int64_t*>(weight_cpu->data);
    int64_t mn = src[0];
    for (int64_t i = 1; i < n; ++i) mn = std::min(mn, src[i]);
    for (int64_t i = 0; i < n; ++i)
      dst[i] = static_cast<float>(static_cast<double>(src[i]) - mn);
  } else if (dt.code == kDGLFloat && dt.bits == 64) {
    const double* src = static_cast<const double*>(weight_cpu->data);
    double mn = src[0];
    for (int64_t i = 1; i < n; ++i) mn = std::min(mn, src[i]);
    for (int64_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i] - mn);
  } else {
    LOG(FATAL) << "Unsupported weight dtype for Ascend row-wise topk";
  }
  return weight_f32.CopyTo(weight->ctx);
}

}  // namespace
#endif  // DGL_USE_ASCEND

template <DGLDeviceType XPU, typename IdType, typename DType>
COOMatrix CSRRowWiseTopk(
    CSRMatrix mat, IdArray rows, int64_t k, NDArray weight, bool ascending) {
#ifdef DGL_USE_ASCEND
  auto ctx = mat.indptr->ctx;
  CHECK(ctx.device_type == kDGLAscend)
      << "Expected Ascend device context for CSRRowWiseTopk";
  CHECK_VALID_CONTEXT(weight, rows);
  // Defense in depth: weight gathers go through the eid mapping, so the
  // weight array must cover every edge id the CSR can reference.
  CHECK(weight->shape[0] >= mat.indices->shape[0])
      << "weight length " << weight->shape[0]
      << " is shorter than the matrix nnz " << mat.indices->shape[0];
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  const bool select_all = (k == -1);
  const int64_t num_rows = rows->shape[0];
  const uint8_t nbits = mat.indptr->dtype.bits;

  // num_rows == 0 implies no rows to process; k == 0 never reaches the
  // operator (the graph layer returns an empty subgraph up front), but the
  // early exit keeps direct aten calls safe too.
  if (num_rows == 0 || k == 0 || mat.indptr->shape[0] <= 1) {
    IdArray empty_row = aten::NewIdArray(0, ctx, nbits);
    return COOMatrix(
        mat.num_rows, mat.num_cols, empty_row, empty_row, empty_row);
  }

  // Per-row pick counts depend only on k and degree. Degrees are two
  // adjacent indptr reads, so one bulk D2H of indptr feeding a single
  // fused host pass (rows copy -> degree -> picks -> prefix) replaces the
  // old single-core CSRGetRowNNZ launch + separate D2H + multiple O(n)
  // loops — on a 1M-row graph that kernel alone cost 45 ms (D6).
  const int64_t indptr_len = mat.indptr->shape[0];
  std::vector<IdType> indptr_host(indptr_len);
  ASCEND_CALL(aclrtMemcpy(
      indptr_host.data(), indptr_len * sizeof(IdType), mat.indptr->data,
      indptr_len * sizeof(IdType), ACL_MEMCPY_DEVICE_TO_HOST));
  std::vector<IdType> rows_host(num_rows);
  ASCEND_CALL(aclrtMemcpy(
      rows_host.data(), num_rows * sizeof(IdType), rows->data,
      num_rows * sizeof(IdType), ACL_MEMCPY_DEVICE_TO_HOST));

  // Single fused pass: degree, picks, and prefix together.
  std::vector<uint32_t> picks(num_rows);
  std::vector<uint32_t> prefix(num_rows + 1, 0);
  const uint32_t k32 = select_all ? 0u : static_cast<uint32_t>(k);
  for (int64_t i = 0; i < num_rows; ++i) {
    const IdType r = rows_host[i];
    uint32_t d = 0;
    if (r >= 0 && static_cast<int64_t>(r) + 1 < indptr_len) {
      d = static_cast<uint32_t>(
          indptr_host[static_cast<int64_t>(r) + 1] - indptr_host[r]);
    }
    picks[i] = select_all ? d : std::min(k32, d);
    prefix[i + 1] = prefix[i] + picks[i];
  }

  // nnz-balanced row partitions across all vector cores (spmm pattern).
  const uint32_t block_dim = QueryVectorCoreCount(ctx.device_id);
  const std::vector<uint32_t> row_split =
      BuildBalancedPartitions(picks, block_dim);

  // Per-block output offsets as prefix sums of picks over row ranges.
  std::vector<uint32_t> out_starts(block_dim + 1, 0);
  for (uint32_t b = 0; b <= block_dim; ++b)
    out_starts[b] = prefix[row_split[b]];
  const int64_t max_output = out_starts[block_dim];
  CHECK(max_output <= static_cast<int64_t>(std::numeric_limits<IdType>::max()))
      << "Output size " << max_output << " exceeds IdType range";

  auto stream = dgl::runtime::getCurrentAscendStream();
  const bool has_data = aten::CSRHasData(mat);
  void* data_ptr = has_data ? mat.data->data : nullptr;

  // Sort-network window bound: the launcher must size the kernel window
  // inside a single Sort call's capacity (repeatTimes <= 255). Rows beyond
  // the window take the GM fallback path in the kernel.
  const uint32_t ub_available = QueryUbAvailableBytes(ctx.device_id);
  constexpr uint32_t kUbBytesPerWindowElem = 68;  // kernel's UB layout
  uint32_t window =
      std::min(ub_available / kUbBytesPerWindowElem, kSortMaxElemsPerCall);
  // Device-verified envelope on 910B3 (CANN 9.1.0, 2026-09): the UB
  // sort path matches CPU exactly up to 992-element windows; larger
  // windows run the same code but the current runtime returns stale
  // staging for them, so they are rejected until re-validated.
  constexpr uint32_t kVerifiedWindowElems = 992;
  window = std::min(window, kVerifiedWindowElems);
  CHECK(window >= 32) << "UB budget too small for the topk window";

  // Rows wider than the UB window are rejected here: the scalar-heap
  // GM fallback is unreliable on the current runtime (its TBuf
  // scratch reads back stale, so the eviction never fires and wide
  // rows keep their first `window` edges) and the runtime faults the
  // vector core on several of its shapes. Rows up to the window take
  // the device-verified UB sort path; hub-scale rows need the
  // two-level merge design and are tracked as follow-up work.
  for (int64_t i = 0; i < num_rows; ++i) {
    const IdType r = rows_host[i];
    const uint32_t d =
        (r >= 0 && static_cast<int64_t>(r) + 1 < indptr_len)
            ? static_cast<uint32_t>(
                  indptr_host[static_cast<int64_t>(r) + 1] - indptr_host[r])
            : 0;
    CHECK(d <= window)
        << "select_topk on Ascend does not yet support rows wider than "
        << "the UB window (" << d << " > " << window
        << "); the GM fallback for hub-scale rows is under repair";
  }

  uint32_t tiling_data[kTilingHeaderWords] = {
      static_cast<uint32_t>(num_rows),
      select_all ? 0u : static_cast<uint32_t>(k),
      static_cast<uint32_t>(select_all ? 1 : 0),
      static_cast<uint32_t>(ascending ? 1 : 0),
      static_cast<uint32_t>(has_data ? 1 : 0),
      static_cast<uint32_t>(mat.num_rows),
      ub_available,
      0,  // reserved
  };

  IdArray picked_row = aten::NewIdArray(max_output, ctx, nbits);
  IdArray picked_col = aten::NewIdArray(max_output, ctx, nbits);
  IdArray picked_idx = aten::NewIdArray(max_output, ctx, nbits);

  if (max_output == 0) {
    return COOMatrix(
        mat.num_rows, mat.num_cols,
        picked_row.CreateView({0}, picked_row->dtype),
        picked_col.CreateView({0}, picked_col->dtype),
        picked_idx.CreateView({0}, picked_idx->dtype));
  }

  // Zero the output buffers on the launch stream (spmm pattern): DGL's
  // array allocator reuses device memory without zeroing, so slots the
  // kernel does not write (idle blocks) must read as 0, not stale data
  // from earlier launches.
  const int64_t out_bytes = max_output * (nbits / 8);
  ASCEND_CALL(
      aclrtMemsetAsync(picked_row->data, out_bytes, 0, out_bytes, stream));
  ASCEND_CALL(
      aclrtMemsetAsync(picked_col->data, out_bytes, 0, out_bytes, stream));
  ASCEND_CALL(
      aclrtMemsetAsync(picked_idx->data, out_bytes, 0, out_bytes, stream));

  NDArray weight_f32 = NormalizeWeightToF32(weight);

  void* tiling_dev = nullptr;
  ASCEND_CALL(
      aclrtMalloc(&tiling_dev, sizeof(tiling_data), ACL_MEM_MALLOC_HUGE_FIRST));
  aclError e = aclrtMemcpyAsync(
      tiling_dev, sizeof(tiling_data), tiling_data, sizeof(tiling_data),
      ACL_MEMCPY_HOST_TO_DEVICE, stream);
  if (e != ACL_SUCCESS) {
    aclrtFree(tiling_dev);  // do not leak on the failure path (PR #42)
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e;
  }
  void* row_split_dev = UploadHostUInt32(row_split, stream);
  void* out_starts_dev = UploadHostUInt32(out_starts, stream);
  // All three async uploads capture stack vectors: one sync covers them
  // before this frame returns (D6 — previously three separate syncs).

  if (std::is_same<IdType, int32_t>::value) {
    aclError err = aclrtlaunch_csr_row_wise_topk_int32(
        block_dim, stream, mat.indptr->data, mat.indices->data, data_ptr,
        rows->data, weight_f32->data, picked_row->data, picked_col->data,
        picked_idx->data, row_split_dev, out_starts_dev, tiling_dev);
    CHECK(err == ACL_SUCCESS)
        << "csr_row_wise_topk_int32 launch failed: " << err;
  } else {
    aclError err = aclrtlaunch_csr_row_wise_topk_int64(
        block_dim, stream, mat.indptr->data, mat.indices->data, data_ptr,
        rows->data, weight_f32->data, picked_row->data, picked_col->data,
        picked_idx->data, row_split_dev, out_starts_dev, tiling_dev);
    CHECK(err == ACL_SUCCESS)
        << "csr_row_wise_topk_int64 launch failed: " << err;
  }

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(tiling_dev));
  ASCEND_CALL(aclrtFree(row_split_dev));
  ASCEND_CALL(aclrtFree(out_starts_dev));

  // Exact allocation means no trim is needed: the kernel filled exactly
  // max_output entries.
  return COOMatrix(
      mat.num_rows, mat.num_cols, picked_row, picked_col, picked_idx);
#else
  LOG(FATAL) << "Ascend support is not compiled. "
                "Please compile with -DUSE_ASCEND=ON";
  return {};
#endif  // DGL_USE_ASCEND
}

template COOMatrix CSRRowWiseTopk<kDGLAscend, int32_t, float>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseTopk<kDGLAscend, int64_t, float>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseTopk<kDGLAscend, int32_t, double>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseTopk<kDGLAscend, int64_t, double>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseTopk<kDGLAscend, int32_t, int32_t>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseTopk<kDGLAscend, int64_t, int32_t>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseTopk<kDGLAscend, int32_t, int64_t>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseTopk<kDGLAscend, int64_t, int64_t>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
