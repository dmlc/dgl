/**
 * Copyright (c) 2024 by Contributors
 * @file csr_row_wise_sampling_uniform_fused.cc
 * @brief Ascend host launcher for uniform CSR row-wise fused sampling.
 *
 * Multi-core design (twin of csr_row_wise_sampling_uniform.cc per
 * ADR-0011): the host computes per-row pick counts from row degrees (one
 * CSRGetRowNNZ launch + one D2H copy), builds nnz-balanced row-range
 * partitions plus per-block output offsets as prefix sums, allocates all
 * outputs exactly, and launches the AIV kernel. Blocks write disjoint
 * output ranges, so the kernel needs no cross-block reduction.
 *
 * Fused extras:
 * - Returns pair<CSRMatrix, IdArray>: the block-local CSR (indptr/col/idx)
 *   plus picked_coo_rows carrying the ROW POSITION of each picked edge.
 * - seed_mapping: the kernel emits a compact interleaved (rid, pos) pair
 *   array (ADR-0014 — AscendC has no GM scatter primitive, so the kernel
 *   only produces contiguous batch writes). This launcher scatters the
 *   pairs into seed_mapping on the host after the kernel completes
 *   (A1 closure: one D2H of the pairs + one H2D of the mapping).
 * - new_seed_nodes: host vector filled with a copy of rows (CPU
 *   reference semantics, CSRRowWisePickFused).
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

extern "C" uint32_t aclrtlaunch_csr_row_wise_sampling_uniform_fused_int32(
    uint32_t blockDim, aclrtStream stream, void* indptr, void* indices,
    void* data, void* rows, void* out_ptr, void* out_rows, void* out_cols,
    void* out_idxs, void* seed_pairs, void* row_split, void* out_starts,
    void* tiling);

extern "C" uint32_t aclrtlaunch_csr_row_wise_sampling_uniform_fused_prep_int32(
    uint32_t blockDim, aclrtStream stream, void* deg, void* picks,
    void* row_split, void* out_starts, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_row_wise_sampling_uniform_fused_prep_int64(
    uint32_t blockDim, aclrtStream stream, void* deg, void* picks,
    void* row_split, void* out_starts, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_row_wise_sampling_uniform_fused_int64(
    uint32_t blockDim, aclrtStream stream, void* indptr, void* indices,
    void* data, void* rows, void* out_ptr, void* out_rows, void* out_cols,
    void* out_idxs, void* seed_pairs, void* row_split, void* out_starts,
    void* tiling);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}  // namespace runtime
}  // namespace dgl
#endif  // DGL_USE_ASCEND

#include <dgl/array.h>
#include <dgl/aten/array_ops.h>
#include <dgl/aten/csr.h>
#include <dgl/random.h>
#include <dgl/runtime/device_api.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "../array_op.h"
#include "csr_row_wise_sampling_uniform_fused_tiling.h"

namespace dgl {
namespace aten {
namespace impl {

namespace {

// Returns the device's vector-core count, queried at runtime so the
// launch adapts to any SoC (910B family: 40 AIV; other families or
// trimmed vNPU instances differ). Falls back to the arch default when
// the query is unavailable. Clamped to kMaxSamplingBlocksFused: the prep
// kernel's UB boundary tables hold that many entries plus one, and the
// tiling hand-off has no way to grow them — beyond the cap the op
// simply uses fewer blocks than there are cores.
uint32_t QueryVectorCoreCountFused(int device_id) {
  int64_t core_num = 0;
  aclError err =
      aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &core_num);
  if (err != ACL_SUCCESS || core_num <= 0 || core_num > 4096) {
    return kDefaultVectorCoreCountFused;
  }
  uint32_t dim = static_cast<uint32_t>(core_num);
  return dim <= kMaxSamplingBlocksFused ? dim : kMaxSamplingBlocksFused;
}

// Returns the per-vector-core unified-buffer budget in bytes (minus the
// runtime-reserved tail). Queried at runtime because UB size differs
// across SoCs (192KB on 910B, 248KB on 950PR).
uint32_t QueryUbAvailableBytesFused(int device_id) {
  int64_t ub_bytes = 0;
  aclError err = aclrtGetDeviceInfo(
      device_id, ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE, &ub_bytes);
  if (err != ACL_SUCCESS ||
      ub_bytes <= static_cast<int64_t>(kUbReservedBytesFused) ||
      ub_bytes > (1 << 30)) {
    return kDefaultUbBytesFused - kUbReservedBytesFused;
  }
  // Reaching here means ub_bytes > kUbReservedBytesFused (checked above).
  return static_cast<uint32_t>(ub_bytes - kUbReservedBytesFused);
}

// Builds nnz-balanced partitions over row weights (spmm precedent):
// returns num_parts+1 boundaries so each part covers a contiguous row
// range with roughly equal total weight.
std::vector<uint32_t> BuildBalancedPartitionsFused(
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

// Per-block output offsets: prefix sums of the per-row pick counts at
// the partition boundaries.
std::vector<uint32_t> BlockOutputStarts(
    const std::vector<uint32_t>& picks, const std::vector<uint32_t>& row_split,
    uint32_t block_dim) {
  std::vector<uint32_t> out_starts(block_dim + 1, 0);
  std::vector<uint32_t> prefix(picks.size() + 1, 0);
  for (size_t i = 0; i < picks.size(); ++i)
    prefix[i + 1] = prefix[i] + picks[i];
  for (uint32_t b = 0; b <= block_dim; ++b)
    out_starts[b] = prefix[row_split[b]];
  return out_starts;
}

// Per-row pick counts: select-all picks the degree, replace picks the
// fanout for non-empty rows, no-replace picks min(fanout, degree).
template <typename IdType>
std::vector<uint32_t> ComputeRowPicks(
    const CSRMatrix& mat, const IdArray& rows, int64_t num_rows,
    uint32_t fanout, bool select_all, bool replace) {
  NDArray deg = CSRGetRowNNZ<kDGLAscend, IdType>(mat, rows);
  std::vector<IdType> deg_host(num_rows);
  ASCEND_CALL(aclrtMemcpy(
      deg_host.data(), num_rows * sizeof(IdType), deg->data,
      num_rows * sizeof(IdType), ACL_MEMCPY_DEVICE_TO_HOST));
  std::vector<uint32_t> picks(num_rows);
  for (int64_t i = 0; i < num_rows; ++i) {
    const uint32_t d = static_cast<uint32_t>(deg_host[i]);
    picks[i] = select_all ? d
               : replace  ? (d == 0 ? 0u : fanout)
                          : std::min(fanout, d);
  }
  return picks;
}

// Empty/degenerate early-exit result: a zeroed block CSR indptr (the CPU
// reference allocates it unconditionally) plus empty edge arrays.
std::pair<CSRMatrix, IdArray> EmptyResult(
    const DGLContext& ctx, int64_t num_rows, uint8_t nbits) {
  IdArray block_csr_indptr = aten::NewIdArray(num_rows + 1, ctx, nbits);
  IdArray empty = aten::NewIdArray(0, ctx, nbits);
  // Zero indptr: DGL's allocator reuses memory without zeroing.
  const int64_t indptr_bytes = (num_rows + 1) * (nbits / 8);
  auto stream0 = dgl::runtime::getCurrentAscendStream();
  if (num_rows + 1 > 0) {
    ASCEND_CALL(aclrtMemsetAsync(
        block_csr_indptr->data, indptr_bytes, 0, indptr_bytes, stream0));
    ASCEND_CALL(aclrtSynchronizeStream(stream0));
  }
  return std::make_pair(
      CSRMatrix(num_rows, 0, block_csr_indptr, empty, empty), empty);
}

// Packs the three launch tables (tiling header + row_split + out_starts)
// into ONE device buffer and uploads it in one shot. The tables are
// consumed through typed __gm__ pointers at fixed offsets, so their
// device addresses are (base, base+tiling_words, base+tiling+split).
// One malloc + one copy + one sync replaces three of each — the sync
// chain dominated the launcher profile (host-bound, not kernel-bound).
struct PackedLaunchTables {
  void* dev = nullptr;
  size_t bytes = 0;
  void* tiling_ptr = nullptr;
  void* row_split_ptr = nullptr;
  void* out_starts_ptr = nullptr;
};

PackedLaunchTables UploadLaunchTablesFused(
    const uint32_t* tiling_data, size_t tiling_words,
    const std::vector<uint32_t>& row_split,
    const std::vector<uint32_t>& out_starts, aclrtStream stream) {
  PackedLaunchTables t;
  const size_t split_words = row_split.size();
  const size_t starts_words = out_starts.size();
  t.bytes = (tiling_words + split_words + starts_words) * sizeof(uint32_t);
  std::vector<uint32_t> packed;
  packed.reserve(tiling_words + split_words + starts_words);
  packed.insert(packed.end(), tiling_data, tiling_data + tiling_words);
  packed.insert(packed.end(), row_split.begin(), row_split.end());
  packed.insert(packed.end(), out_starts.begin(), out_starts.end());
  ASCEND_CALL(aclrtMalloc(&t.dev, t.bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  // packed lives on this frame; sync before returning so the async copy
  // never reads dead stack memory (queue-discipline rule).
  ASCEND_CALL(aclrtMemcpyAsync(
      t.dev, t.bytes, packed.data(), t.bytes, ACL_MEMCPY_HOST_TO_DEVICE,
      stream));
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  t.tiling_ptr = t.dev;
  t.row_split_ptr = static_cast<char*>(t.dev) + tiling_words * sizeof(uint32_t);
  t.out_starts_ptr = static_cast<char*>(t.dev) +
                     (tiling_words + split_words) * sizeof(uint32_t);
  return t;
}

}  // namespace

#ifdef DGL_USE_ASCEND
namespace {

// Scatters the compact (rid, pos) pair array into seed_mapping on the
// host (ADR-0014 A1): D2H the pairs, write mapping[rid] = pos, H2D the
// touched mapping back. The mapping is a full device array (length =
// graph nodes), so the round-trip is mapping-sized; acceptable because
// the fused host path already moves mapping-sized arrays per layer.
template <typename IdType>
void ScatterSeedMappingFromPairs(
    const IdArray& seed_pairs, int64_t pair_count, const IdArray& seed_mapping,
    int64_t num_rows, aclrtStream stream) {
  if (pair_count <= 0 || !seed_mapping.defined()) return;
  const int64_t mapping_len = seed_mapping->shape[0];
  // The mapping spans the graph's node count; the CSR row count is the
  // seed count. A homo graph makes them equal, but heterographs and
  // partial seed sets make mapping_len >= needed rows; both are valid.
  CHECK(mapping_len >= num_rows)
      << "seed_mapping length " << mapping_len
      << " is smaller than the row count " << num_rows;
  // PINNED staging + async D2H under one sync. The earlier pipelined
  // attempt failed because std::vector is pageable host memory — async
  // D2H into pageable buffers is not reliably visible on this stack.
  // Pinned memory (aclrtMallocHost) is the DMA-safe counterpart and lets
  // both downloads share a single stream sync (two synchronous
  // round-trips serialized twice the latency).
  void* pairs_pinned = nullptr;
  void* mapping_pinned = nullptr;
  ASCEND_CALL(aclrtMallocHost(&pairs_pinned, 2 * pair_count * sizeof(IdType)));
  ASCEND_CALL(aclrtMallocHost(&mapping_pinned, mapping_len * sizeof(IdType)));
  ASCEND_CALL(aclrtMemcpyAsync(
      pairs_pinned, 2 * pair_count * sizeof(IdType), seed_pairs->data,
      2 * pair_count * sizeof(IdType), ACL_MEMCPY_DEVICE_TO_HOST, stream));
  ASCEND_CALL(aclrtMemcpyAsync(
      mapping_pinned, mapping_len * sizeof(IdType), seed_mapping->data,
      mapping_len * sizeof(IdType), ACL_MEMCPY_DEVICE_TO_HOST, stream));
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  IdType* pairs = static_cast<IdType*>(pairs_pinned);
  IdType* mapping = static_cast<IdType*>(mapping_pinned);
  for (int64_t p = 0; p < pair_count; ++p) {
    const IdType rid = pairs[2 * p];
    const IdType pos = pairs[2 * p + 1];
    // Invalid rows emit sentinel pairs (rid untouched, out of range);
    // skip them and clamp valid rids (defense in depth: never trust the
    // device buffer).
    if (rid < 0 || rid >= static_cast<IdType>(mapping_len)) continue;
    mapping[rid] = pos;
  }
  ASCEND_CALL(aclrtMemcpyAsync(
      seed_mapping->data, mapping_len * sizeof(IdType), mapping_pinned,
      mapping_len * sizeof(IdType), ACL_MEMCPY_HOST_TO_DEVICE, stream));
  // Pinned buffers are freed below only after this sync — the transfer
  // source must outlive the copy (queue-discipline rule).
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFreeHost(pairs_pinned));
  ASCEND_CALL(aclrtFreeHost(mapping_pinned));
}

// Zero-output path: the kernel did not run, but valid rows still need
// their mapping positions (row i of rows maps to seed_mapping[rid]=i).
// rows is a device array; pull it once and scatter directly.
template <typename IdType>
void ScatterSeedMappingFromRows(
    const IdArray& rows, const IdArray& seed_mapping, int64_t num_rows,
    aclrtStream stream) {
  if (num_rows <= 0 || !seed_mapping.defined()) return;
  std::vector<IdType> rows_host(num_rows);
  ASCEND_CALL(aclrtMemcpy(
      rows_host.data(), num_rows * sizeof(IdType), rows->data,
      num_rows * sizeof(IdType), ACL_MEMCPY_DEVICE_TO_HOST));
  const int64_t mapping_len = seed_mapping->shape[0];
  std::vector<IdType> mapping(mapping_len);
  ASCEND_CALL(aclrtMemcpy(
      mapping.data(), mapping_len * sizeof(IdType), seed_mapping->data,
      mapping_len * sizeof(IdType), ACL_MEMCPY_DEVICE_TO_HOST));
  for (int64_t i = 0; i < num_rows; ++i) {
    const IdType rid = rows_host[i];
    if (rid >= 0 && rid < static_cast<IdType>(mapping_len)) {
      mapping[rid] = static_cast<IdType>(i);
    }
  }
  ASCEND_CALL(aclrtMemcpyAsync(
      seed_mapping->data, mapping_len * sizeof(IdType), mapping.data(),
      mapping_len * sizeof(IdType), ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ASCEND_CALL(aclrtSynchronizeStream(stream));
}

// Zero-output degenerate path: the kernel does not run, but valid rows
// still get their mapping positions and the indptr stays well-formed.
template <typename IdType>
std::pair<CSRMatrix, IdArray> ZeroOutputResult(
    const CSRMatrix& mat, const DGLContext& ctx, const IdArray& rows,
    const IdArray& seed_mapping, std::vector<IdType>* new_seed_nodes,
    int64_t num_rows, uint8_t nbits, bool map_seed_nodes, aclrtStream stream,
    IdArray picked_coo_rows, IdArray picked_col, IdArray picked_idx,
    IdArray block_csr_indptr) {
  const int64_t indptr_bytes = (num_rows + 1) * (nbits / 8);
  ASCEND_CALL(aclrtMemsetAsync(
      block_csr_indptr->data, indptr_bytes, 0, indptr_bytes, stream));
  if (map_seed_nodes) {
    ScatterSeedMappingFromRows<IdType>(rows, seed_mapping, num_rows, stream);
    if (new_seed_nodes != nullptr) {
      new_seed_nodes->resize(num_rows);
      ASCEND_CALL(aclrtMemcpy(
          new_seed_nodes->data(), num_rows * sizeof(IdType), rows->data,
          num_rows * sizeof(IdType), ACL_MEMCPY_DEVICE_TO_HOST));
    }
  }
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  return std::make_pair(
      CSRMatrix(
          num_rows, 0, block_csr_indptr,
          picked_col.CreateView({0}, picked_col->dtype),
          picked_idx.CreateView({0}, picked_idx->dtype)),
      picked_coo_rows.CreateView({0}, picked_coo_rows->dtype));
}

}  // namespace
#endif  // DGL_USE_ASCEND

template <DGLDeviceType XPU, typename IdType, bool map_seed_nodes>
std::pair<CSRMatrix, IdArray> CSRRowWiseSamplingUniformFused(
    CSRMatrix mat, IdArray rows, IdArray seed_mapping,
    std::vector<IdType>* new_seed_nodes, int64_t num_samples, bool replace) {
#ifdef DGL_USE_ASCEND
  auto ctx = mat.indptr->ctx;
  CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context for "
                                          "CSRRowWiseSamplingUniformFused";
  // Defense in depth (ADR-0012): the upstream Python layer historically
  // created the mapping as int64 regardless of graph idtype, which the
  // CPU path silently misreads (4-byte offset UB). Reject it here.
  if (map_seed_nodes) {
    CHECK(seed_mapping.defined() && seed_mapping->dtype == mat.indptr->dtype)
        << "seed_mapping dtype must match the CSR idtype "
           "(got bits="
        << (seed_mapping.defined() ? seed_mapping->dtype.bits : 0) << ", want "
        << mat.indptr->dtype.bits << ")";
  }
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  const bool select_all = (num_samples == -1);
  replace = (replace && !select_all);

  const int64_t num_rows = rows->shape[0];
  const uint8_t nbits = mat.indptr->dtype.bits;

  if (map_seed_nodes && new_seed_nodes != nullptr) {
    new_seed_nodes->clear();
  }

  // Early exit: nothing to sample. The block CSR still needs its indptr
  // (num_rows + 1 zeros) so downstream consumers see a well-formed empty
  // matrix (CPU reference allocates block_csr_indptr unconditionally).
  if (num_rows == 0 || mat.indptr->shape[0] <= 1 || num_samples == 0) {
    return EmptyResult(ctx, num_rows, nbits);
  }

  // Per-row pick counts (degrees come back to the host once).
  const uint32_t fanout = select_all ? 0u : static_cast<uint32_t>(num_samples);
  // Replacement sampling draws num_picks == fanout picks per row into a
  // UB-sized scratch regardless of degree; a fanout beyond the window
  // would overflow the pick scratch and output staging (the kernel's
  // window check bounds degree, not fanout). Reject it here — sampling
  // with replacement beyond a few thousand picks per row is far outside
  // the training workloads this operator serves.
  if (replace) {
    const uint32_t ub_available = QueryUbAvailableBytesFused(ctx.device_id);
    const uint32_t window_elems = (ub_available - kMetaUbReserveFused) /
                                  kWindowInstancesFused / sizeof(IdType);
    CHECK(fanout <= window_elems)
        << "fused sampling fanout " << fanout
        << " exceeds the per-core window of " << window_elems
        << " (with-replace draws that many picks into UB scratch)";
  }
  // PERF-DEBT-4 (done): the pick/partition chain runs on the DEVICE.
  // CSRGetRowNNZ leaves the degrees in GM; a single-core prep kernel
  // turns them into picks + row_split + out_starts in one scan, so the
  // degree D2H, the host-side partition arithmetic, and the table
  // upload all disappear. The host still needs max_output to size the
  // output arrays — one scalar D2H after the prep kernel syncs.
  const uint32_t block_dim = QueryVectorCoreCountFused(ctx.device_id);

  auto stream = dgl::runtime::getCurrentAscendStream();
  NDArray deg = CSRGetRowNNZ<kDGLAscend, IdType>(mat, rows);
  IdArray picks_arr = aten::NewIdArray(num_rows, ctx, 8 * sizeof(uint32_t));
  IdArray row_split_arr =
      aten::NewIdArray(block_dim + 1, ctx, 8 * sizeof(uint32_t));
  IdArray out_starts_arr =
      aten::NewIdArray(block_dim + 1, ctx, 8 * sizeof(uint32_t));

  uint32_t prep_tiling[5] = {
      static_cast<uint32_t>(num_rows),
      fanout,
      static_cast<uint32_t>(replace ? 1 : 0),
      static_cast<uint32_t>(select_all ? 1 : 0),
      block_dim,
  };
  void* prep_tiling_dev = nullptr;
  ASCEND_CALL(aclrtMalloc(
      &prep_tiling_dev, sizeof(prep_tiling), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpyAsync(
      prep_tiling_dev, sizeof(prep_tiling), prep_tiling, sizeof(prep_tiling),
      ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ASCEND_CALL(aclrtSynchronizeStream(stream));  // stack-backed upload
  {
    // Degrees carry the graph idtype — dispatch the matching variant.
    const aclError err =
        std::is_same<IdType, int32_t>::value
            ? aclrtlaunch_csr_row_wise_sampling_uniform_fused_prep_int32(
                  1, stream, deg->data, picks_arr->data, row_split_arr->data,
                  out_starts_arr->data, prep_tiling_dev)
            : aclrtlaunch_csr_row_wise_sampling_uniform_fused_prep_int64(
                  1, stream, deg->data, picks_arr->data, row_split_arr->data,
                  out_starts_arr->data, prep_tiling_dev);
    CHECK(err == ACL_SUCCESS) << "fused prep kernel launch failed: " << err;
  }
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(prep_tiling_dev));

  uint32_t max_output_u32 = 0;
  ASCEND_CALL(aclrtMemcpy(
      &max_output_u32, sizeof(uint32_t),
      static_cast<char*>(out_starts_arr->data) + block_dim * sizeof(uint32_t),
      sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST));
  const int64_t max_output = max_output_u32;
  CHECK(max_output <= static_cast<int64_t>(std::numeric_limits<IdType>::max()))
      << "Output size " << max_output << " exceeds IdType range";
  const bool has_data = aten::CSRHasData(mat);
  void* data_ptr = has_data ? mat.data->data : nullptr;

  // The tiling block carries 32-bit fields; a seed set beyond uint32
  // would silently truncate and corrupt every row boundary below.
  CHECK(num_rows <= std::numeric_limits<uint32_t>::max())
      << "fused sampling supports at most "
      << std::numeric_limits<uint32_t>::max() << " seed rows, got " << num_rows;
  CHECK(
      mat.num_rows <=
      static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "fused sampling supports at most "
      << std::numeric_limits<uint32_t>::max() << " CSR rows, got "
      << mat.num_rows;
  uint32_t tiling_data[kTilingHeaderWordsFused] = {
      static_cast<uint32_t>(num_rows),
      fanout,
      static_cast<uint32_t>(replace ? 1 : 0),
      static_cast<uint32_t>(has_data ? 1 : 0),
      static_cast<uint32_t>(RandomEngine::ThreadLocal()->RandInt(1000000000)),
      static_cast<uint32_t>(select_all ? 1 : 0),
      static_cast<uint32_t>(mat.num_rows),
      QueryUbAvailableBytesFused(ctx.device_id),
      static_cast<uint32_t>(map_seed_nodes ? 1 : 0),
  };

  IdArray picked_coo_rows = aten::NewIdArray(max_output, ctx, nbits);
  IdArray picked_col = aten::NewIdArray(max_output, ctx, nbits);
  IdArray picked_idx = aten::NewIdArray(max_output, ctx, nbits);
  IdArray block_csr_indptr = aten::NewIdArray(num_rows + 1, ctx, nbits);
  // Compact (rid, pos) pair array: one pair per valid row when mapping.
  // Interleaved (rid, pos) pairs: TWO elements per row. When
  // map_seed_nodes is false the kernel never touches it; a 1-element
  // dummy keeps the GlobalTensor binding safe (ADR-0014).
  const int64_t pair_count = map_seed_nodes ? num_rows : 1;
  IdArray seed_pairs = aten::NewIdArray(2 * pair_count, ctx, nbits);

  if (max_output == 0) {
    return ZeroOutputResult(
        mat, ctx, rows, seed_mapping, new_seed_nodes, num_rows, nbits,
        map_seed_nodes, stream, picked_coo_rows, picked_col, picked_idx,
        block_csr_indptr);
  }

  // No output zeroing on the main path: every slot is provably written.
  // out_ptr gets one entry per row (invalid rids included); the edge
  // arrays are covered exactly because out_starts is the prefix sum of
  // the same pick counts the kernel recomputes (host/kernel formulas
  // verified identical, segmented paths return exact counts); and
  // seed_pairs emits one pair per row (sentinels for invalid rids).
  // The degenerate paths still zero their indptr explicitly. The memset
  // chain cost ~9ms of launcher wall time and serialized ahead of the
  // launch on the same stream; allocator-reuse ghost data only ever
  // mattered for UNwritten slots.

  // The prep kernel already produced row_split/out_starts in device
  // memory — only the small tiling header uploads now.
  void* tiling_dev = nullptr;
  ASCEND_CALL(
      aclrtMalloc(&tiling_dev, sizeof(tiling_data), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpyAsync(
      tiling_dev, sizeof(tiling_data), tiling_data, sizeof(tiling_data),
      ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ASCEND_CALL(aclrtSynchronizeStream(stream));  // stack-backed upload

  if (std::is_same<IdType, int32_t>::value) {
    aclError err = aclrtlaunch_csr_row_wise_sampling_uniform_fused_int32(
        block_dim, stream, mat.indptr->data, mat.indices->data, data_ptr,
        rows->data, block_csr_indptr->data, picked_coo_rows->data,
        picked_col->data, picked_idx->data, seed_pairs->data,
        row_split_arr->data, out_starts_arr->data, tiling_dev);
    CHECK(err == ACL_SUCCESS)
        << "csr_row_wise_sampling_uniform_fused_int32 launch failed: " << err;
  } else {
    aclError err = aclrtlaunch_csr_row_wise_sampling_uniform_fused_int64(
        block_dim, stream, mat.indptr->data, mat.indices->data, data_ptr,
        rows->data, block_csr_indptr->data, picked_coo_rows->data,
        picked_col->data, picked_idx->data, seed_pairs->data,
        row_split_arr->data, out_starts_arr->data, tiling_dev);
    CHECK(err == ACL_SUCCESS)
        << "csr_row_wise_sampling_uniform_fused_int64 launch failed: " << err;
  }

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(tiling_dev));

  // ADR-0014 A1 closure: scatter the compact (rid, pos) pairs into
  // seed_mapping on the host, then write the mapping back. The kernel is
  // already synchronized, so a plain D2H copy is safe.
  if (map_seed_nodes) {
    ScatterSeedMappingFromPairs<IdType>(
        seed_pairs, num_rows, seed_mapping, num_rows, stream);
    if (new_seed_nodes != nullptr) {
      new_seed_nodes->resize(num_rows);
      ASCEND_CALL(aclrtMemcpy(
          new_seed_nodes->data(), num_rows * sizeof(IdType), rows->data,
          num_rows * sizeof(IdType), ACL_MEMCPY_DEVICE_TO_HOST));
    }
  }

  return std::make_pair(
      CSRMatrix(num_rows, max_output, block_csr_indptr, picked_col, picked_idx),
      picked_coo_rows);
#else
  LOG(FATAL) << "Ascend support is not compiled. "
                "Please compile with -DUSE_ASCEND=ON";
  return {};
#endif  // DGL_USE_ASCEND
}

template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingUniformFused<kDGLAscend, int32_t, true>(
    CSRMatrix, IdArray, IdArray, std::vector<int32_t>*, int64_t, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingUniformFused<kDGLAscend, int64_t, true>(
    CSRMatrix, IdArray, IdArray, std::vector<int64_t>*, int64_t, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingUniformFused<kDGLAscend, int32_t, false>(
    CSRMatrix, IdArray, IdArray, std::vector<int32_t>*, int64_t, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingUniformFused<kDGLAscend, int64_t, false>(
    CSRMatrix, IdArray, IdArray, std::vector<int64_t>*, int64_t, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
