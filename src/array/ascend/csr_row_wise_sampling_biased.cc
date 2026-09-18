/**
 * Copyright (c) 2024 by Contributors
 * @file csr_row_wise_sampling_biased.cc
 * @brief Ascend host launcher for tag-bucketed biased CSR row-wise sampling.
 *
 * Pipeline (mirrors the uniform v2 launcher):
 *  1. Normalize bias on the host: down-cast float64 to float32 (double is
 *     forbidden in __aicore__), clamp negative / NaN weights to zero —
     the CPU reference only counts buckets with bias > 0, so the clamp is
     semantics-preserving — and upload to the graph device.
 *  2. Per-row pick counts come from a small helper kernel that sums the
 *     positive-bias bucket sizes of each seed row (the plain CSRGetRowNNZ
 *     kernel cannot see bias), then D2H so the host can build the
 *     nnz-balanced row_split / out_starts tables (spmm pattern).
 *  3. The main kernel samples bucket-by-bucket per row; blocks write
 *     disjoint output slices, so no cross-block reduction is needed.
 * The per-row pick counts are baked into out_starts by the host (prefix
 * sums over rows), which the kernel reads back per row — no second table.
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

extern "C" uint32_t aclrtlaunch_csr_row_wise_sampling_biased_int32(
    uint32_t blockDim, aclrtStream stream, void* indptr, void* indices,
    void* data, void* rows, void* tag_offset, void* bias, void* out_rows,
    void* out_cols, void* out_idxs, void* row_split, void* out_starts,
    void* tiling);

extern "C" uint32_t aclrtlaunch_csr_row_wise_sampling_biased_int64(
    uint32_t blockDim, aclrtStream stream, void* indptr, void* indices,
    void* data, void* rows, void* tag_offset, void* bias, void* out_rows,
    void* out_cols, void* out_idxs, void* row_split, void* out_starts,
    void* tiling);

extern "C" uint32_t aclrtlaunch_csr_row_wise_sampling_biased_num_picks_int32(
    uint32_t blockDim, aclrtStream stream, void* rows, void* tag_offset,
    void* bias, void* out, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_row_wise_sampling_biased_num_picks_int64(
    uint32_t blockDim, aclrtStream stream, void* rows, void* tag_offset,
    void* bias, void* out, void* tiling);

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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "../array_op.h"
#include "csr_row_wise_sampling_biased_tiling.h"

namespace dgl {
namespace aten {
namespace impl {

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
      device_id, ACL_DEV_ATTR_LOCAL_MEM_PER_VECTOR_CORE, &ub_bytes);
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

// Async upload: the caller keeps `host` alive until it synchronizes the
// stream (or the launch completes). Batching uploads under one sync cuts
// per-call stall — each sync costs ~1ms on this stack when the stream is
// otherwise idle.
void* UploadHostUInt32Async(
    const std::vector<uint32_t>& host, aclrtStream stream) {
  void* dev = nullptr;
  ASCEND_CALL(aclrtMalloc(
      &dev, host.size() * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpyAsync(
      dev, host.size() * sizeof(uint32_t), host.data(),
      host.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE, stream));
  return dev;
}

// Uploads a host uint32 table to device memory on the launch stream and
// synchronizes before returning.
void* UploadHostUInt32(const std::vector<uint32_t>& host, aclrtStream stream) {
  void* dev = UploadHostUInt32Async(host, stream);
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  return dev;
}

// Copies a device uint32 array to host.
std::vector<uint32_t> CopyDeviceArrayToHostUInt32(
    const void* dev_ptr, size_t count) {
  std::vector<uint32_t> host(count);
  if (count == 0) return host;
  ASCEND_CALL(aclrtMemcpy(
      host.data(), count * sizeof(uint32_t), dev_ptr, count * sizeof(uint32_t),
      ACL_MEMCPY_DEVICE_TO_HOST));
  return host;
}

// Normalizes the bias array on the host: float32 stays, float64 is
// down-cast, negative / NaN entries are clamped to zero (the CPU path
// only counts buckets with bias > 0, so this is semantics-preserving).
// Returns a device float32 buffer of num_tags elements on the graph
// device. Caller frees with aclrtFree.
std::vector<float> BiasToHostClamped(const FloatArray& bias) {
  const int64_t num_tags = bias->shape[0];
  const int64_t bias_bits = bias->dtype.bits;
  const size_t bytes = static_cast<size_t>(num_tags) * (bias_bits / 8);
  std::vector<uint8_t> raw(bytes);
  if (bias->ctx.device_type == kDGLCPU) {
    std::memcpy(raw.data(), bias->data, bytes);
  } else {  // already on the device: D2H once (T is small)
    ASCEND_CALL(aclrtMemcpy(
        raw.data(), bytes, bias->data, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
  }

  std::vector<float> bias_host(static_cast<size_t>(num_tags));
  if (bias_bits == 32) {
    std::vector<float> tmp(static_cast<size_t>(num_tags));
    std::memcpy(tmp.data(), raw.data(), bytes);
    bias_host.swap(tmp);
  } else {  // 64
    std::vector<double> tmp(static_cast<size_t>(num_tags));
    std::memcpy(tmp.data(), raw.data(), bytes);
    for (int64_t t = 0; t < num_tags; ++t)
      bias_host[t] = static_cast<float>(tmp[t]);
  }

  // NaN or negative -> zero-weight bucket (CPU counts only bias > 0).
  std::replace_if(
      bias_host.begin(), bias_host.end(), [](float w) { return !(w >= 0.0f); },
      0.0f);
  return bias_host;
}

// Launches the pick-count helper kernel and returns the per-row picks on
// the host (biased nnz per seed row). Frees nothing; caller frees
// picks_dev / picks_tiling_dev.
std::vector<uint32_t> ComputeRowPicks(
    const IdArray& rows, const void* tag_offset_ptr, void* bias_dev,
    bool is_int32, int64_t num_rows, uint32_t fanout, bool replace,
    bool select_all, int64_t num_total_rows, int64_t num_tags, int device_id,
    aclrtStream stream, void** picks_dev, void** picks_tiling_dev) {
  uint32_t tiling_data[kTilingHeaderWords] = {
      static_cast<uint32_t>(num_rows),
      fanout,
      static_cast<uint32_t>(replace ? 1 : 0),
      0,
      0,
      static_cast<uint32_t>(select_all ? 1 : 0),
      static_cast<uint32_t>(num_total_rows),
      0,
      static_cast<uint32_t>(num_tags),
  };
  ASCEND_CALL(aclrtMalloc(
      picks_tiling_dev, sizeof(tiling_data), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpyAsync(
      *picks_tiling_dev, sizeof(tiling_data), tiling_data, sizeof(tiling_data),
      ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ASCEND_CALL(aclrtSynchronizeStream(stream));

  ASCEND_CALL(aclrtMalloc(
      picks_dev, num_rows * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
  // Zero the pick counts before the launch: idle blocks (when block_dim
  // exceeds the row count) never write their slots, and the allocator does
  // not zero fresh device memory.
  ASCEND_CALL(aclrtMemsetAsync(
      *picks_dev, num_rows * sizeof(uint32_t), 0, num_rows * sizeof(uint32_t),
      stream));
  // The pick-count kernel partitions rows evenly across ALL vector cores
  // (csr_get_row_nnz pattern). Launching it with blockDim=1 left ~97% of
  // the device idle and dominated the device time at larger tag counts
  // (measured: 26ns per scalar tag read x O(n*T) on a single core).
  const uint32_t picks_block_dim = QueryVectorCoreCount(device_id);
  aclError err;
  if (is_int32) {
    err = aclrtlaunch_csr_row_wise_sampling_biased_num_picks_int32(
        picks_block_dim, stream, rows->data, const_cast<void*>(tag_offset_ptr),
        bias_dev, *picks_dev, *picks_tiling_dev);
  } else {
    err = aclrtlaunch_csr_row_wise_sampling_biased_num_picks_int64(
        picks_block_dim, stream, rows->data, const_cast<void*>(tag_offset_ptr),
        bias_dev, *picks_dev, *picks_tiling_dev);
  }
  CHECK(err == ACL_SUCCESS)
      << "csr_row_wise_sampling_biased_num_picks launch failed: " << err;
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  return CopyDeviceArrayToHostUInt32(*picks_dev, num_rows);
}

// Uploads the normalized float32 bias to the graph device. Caller frees
// with aclrtFree.
void* NormalizeBiasToDevice(const FloatArray& bias, int device_id) {
  const std::vector<float> bias_host = BiasToHostClamped(bias);
  const int64_t num_tags = bias->shape[0];
  void* dev = nullptr;
  ASCEND_CALL(aclrtSetDevice(device_id));
  ASCEND_CALL(
      aclrtMalloc(&dev, num_tags * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(
      dev, num_tags * sizeof(float), bias_host.data(), num_tags * sizeof(float),
      ACL_MEMCPY_HOST_TO_DEVICE));
  return dev;
}

// Fills the main-kernel tiling block and launches the sampler. Returns
// aclError from the launch.
aclError LaunchMainSampler(
    bool is_int32, uint32_t block_dim, aclrtStream stream, void* indptr,
    void* indices, void* data, void* rows, void* tag_offset, void* bias_dev,
    void* out_rows, void* out_cols, void* out_idxs, void* row_split_dev,
    void* out_starts_dev, int64_t num_rows, uint32_t fanout, bool replace,
    bool has_data, uint32_t seed, bool select_all, int64_t num_total_rows,
    int64_t num_tags, int device_id) {
  uint32_t tiling_data[kTilingHeaderWords] = {
      static_cast<uint32_t>(num_rows),
      fanout,
      static_cast<uint32_t>(replace ? 1 : 0),
      static_cast<uint32_t>(has_data ? 1 : 0),
      seed,
      static_cast<uint32_t>(select_all ? 1 : 0),
      static_cast<uint32_t>(num_total_rows),
      QueryUbAvailableBytes(device_id),
      static_cast<uint32_t>(num_tags),
  };
  void* tiling_dev = nullptr;
  ASCEND_CALL(
      aclrtMalloc(&tiling_dev, sizeof(tiling_data), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpyAsync(
      tiling_dev, sizeof(tiling_data), tiling_data, sizeof(tiling_data),
      ACL_MEMCPY_HOST_TO_DEVICE, stream));
  // tiling_data is a stack array: wait for the copy to land before the
  // frame that owns it returns.
  ASCEND_CALL(aclrtSynchronizeStream(stream));

  aclError err;
  if (is_int32) {
    err = aclrtlaunch_csr_row_wise_sampling_biased_int32(
        block_dim, stream, indptr, indices, data, rows, tag_offset, bias_dev,
        out_rows, out_cols, out_idxs, row_split_dev, out_starts_dev,
        tiling_dev);
  } else {
    err = aclrtlaunch_csr_row_wise_sampling_biased_int64(
        block_dim, stream, indptr, indices, data, rows, tag_offset, bias_dev,
        out_rows, out_cols, out_idxs, row_split_dev, out_starts_dev,
        tiling_dev);
  }
  ASCEND_CALL(aclrtFree(tiling_dev));
  return err;
}

// Validates the tag_offset matrix shape against the bias length.
void CheckTagInputs(const NDArray& tag_offset, int64_t num_tags) {
  CHECK_EQ(tag_offset->ndim, 2)
      << "tag_offset should have shape [num_nodes, num_tags + 1]";
  CHECK_EQ(tag_offset->shape[1], num_tags + 1)
      << "tag_offset's second dimension should equal num_tags + 1";
  CHECK_LE(num_tags, static_cast<int64_t>(kMaxTagCount))
      << "Biased sampling supports at most " << kMaxTagCount
      << " tag buckets, got " << num_tags;
}

// Builds an empty COOMatrix from zero-length views of the outputs.
COOMatrix EmptyResult(
    const CSRMatrix& mat, IdArray picked_row, IdArray picked_col,
    IdArray picked_idx) {
  return COOMatrix(
      mat.num_rows, mat.num_cols, picked_row.CreateView({0}, picked_row->dtype),
      picked_col.CreateView({0}, picked_col->dtype),
      picked_idx.CreateView({0}, picked_idx->dtype));
}

// Allocates and zeroes the three output arrays (allocator reuse guard).
std::tuple<IdArray, IdArray, IdArray> AllocZeroedOutputs(
    int64_t max_output, const DGLContext& ctx, uint8_t nbits,
    aclrtStream stream) {
  IdArray picked_row = aten::NewIdArray(max_output, ctx, nbits);
  IdArray picked_col = aten::NewIdArray(max_output, ctx, nbits);
  IdArray picked_idx = aten::NewIdArray(max_output, ctx, nbits);
  const int64_t out_bytes = max_output * (nbits / 8);
  // CANN 9.0 rejects aclrtMemsetAsync with size 0; the all-zero-bias case
  // reaches here with max_output == 0 and the early-exit path below returns
  // empty views anyway, so the memset is only needed for non-empty outputs.
  if (max_output > 0) {
    ASCEND_CALL(
        aclrtMemsetAsync(picked_row->data, out_bytes, 0, out_bytes, stream));
    ASCEND_CALL(
        aclrtMemsetAsync(picked_col->data, out_bytes, 0, out_bytes, stream));
    ASCEND_CALL(
        aclrtMemsetAsync(picked_idx->data, out_bytes, 0, out_bytes, stream));
  }
  return {picked_row, picked_col, picked_idx};
}

}  // namespace

template <DGLDeviceType XPU, typename IdType, typename FloatType>
COOMatrix CSRRowWiseSamplingBiased(
    CSRMatrix mat, IdArray rows, int64_t num_samples, NDArray tag_offset,
    FloatArray bias, bool replace) {
#ifdef DGL_USE_ASCEND
  static_assert(
      std::is_same<FloatType, float>::value,
      "Ascend biased sampling only supports float32 bias");
  auto ctx = mat.indptr->ctx;
  CHECK(ctx.device_type == kDGLAscend)
      << "Expected Ascend device context for CSRRowWiseSamplingBiased";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  const bool select_all = (num_samples == -1);
  replace = (replace && !select_all);

  const int64_t num_rows = rows->shape[0];
  const uint8_t nbits = mat.indptr->dtype.bits;
  const int64_t num_tags = bias->shape[0];
  CheckTagInputs(tag_offset, num_tags);

  // num_samples == 0 implies !select_all (select_all means -1).
  if (num_rows == 0 || mat.indptr->shape[0] <= 1 || num_samples == 0) {
    IdArray empty_row = aten::NewIdArray(0, ctx, nbits);
    return COOMatrix(
        mat.num_rows, mat.num_cols, empty_row, empty_row, empty_row);
  }

  auto stream = dgl::runtime::getCurrentAscendStream();

  // Bias: clamp + float32 on the graph device.
  void* bias_dev = NormalizeBiasToDevice(bias, ctx.device_id);

  // Per-row pick counts via the helper kernel (biased nnz per row).
  const uint32_t fanout = select_all ? 0u : static_cast<uint32_t>(num_samples);
  void* picks_dev = nullptr;
  void* picks_tiling_dev = nullptr;
  const std::vector<uint32_t> picks = ComputeRowPicks(
      rows, tag_offset->data, bias_dev, std::is_same<IdType, int32_t>::value,
      num_rows, fanout, replace, select_all, mat.num_rows, num_tags,
      ctx.device_id, stream, &picks_dev, &picks_tiling_dev);

  // nnz-balanced row partitions across all vector cores (spmm pattern).
  const uint32_t block_dim = QueryVectorCoreCount(ctx.device_id);
  const std::vector<uint32_t> row_split =
      BuildBalancedPartitions(picks, block_dim);

  // Per-row output starts as prefix sums over picks; the last entry is
  // the total. The kernel reads row i's pick count back as
  // out_starts[i+1] - out_starts[i], so this table carries both the
  // per-block offsets and the per-row counts.
  std::vector<uint32_t> out_starts(num_rows + 1, 0);
  for (int64_t i = 0; i < num_rows; ++i)
    out_starts[i + 1] = out_starts[i] + picks[i];
  const int64_t max_output = out_starts[num_rows];
  CHECK(max_output <= static_cast<int64_t>(std::numeric_limits<IdType>::max()))
      << "Output size " << max_output << " exceeds IdType range";

  auto outputs = AllocZeroedOutputs(max_output, ctx, nbits, stream);
  IdArray picked_row = std::get<0>(outputs);
  IdArray picked_col = std::get<1>(outputs);
  IdArray picked_idx = std::get<2>(outputs);

  if (max_output == 0) {
    ASCEND_CALL(aclrtFree(picks_dev));
    ASCEND_CALL(aclrtFree(picks_tiling_dev));
    ASCEND_CALL(aclrtFree(bias_dev));
    return EmptyResult(mat, picked_row, picked_col, picked_idx);
  }

  // Main kernel launch.
  const bool has_data = aten::CSRHasData(mat);
  void* data_ptr = has_data ? mat.data->data : nullptr;
  // Async uploads: row_split / out_starts live until after the main
  // launch, so their copies ride under LaunchMainSampler's tiling sync
  // instead of paying two extra stalls.
  void* row_split_dev = UploadHostUInt32Async(row_split, stream);
  void* out_starts_dev = UploadHostUInt32Async(out_starts, stream);

  aclError err = LaunchMainSampler(
      std::is_same<IdType, int32_t>::value, block_dim, stream, mat.indptr->data,
      mat.indices->data, data_ptr, rows->data, tag_offset->data, bias_dev,
      picked_row->data, picked_col->data, picked_idx->data, row_split_dev,
      out_starts_dev, num_rows, fanout, replace, has_data,
      static_cast<uint32_t>(RandomEngine::ThreadLocal()->RandInt(1000000000)),
      select_all, mat.num_rows, num_tags, ctx.device_id);
  CHECK(err == ACL_SUCCESS)
      << "csr_row_wise_sampling_biased launch failed: " << err;

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(row_split_dev));
  ASCEND_CALL(aclrtFree(out_starts_dev));
  ASCEND_CALL(aclrtFree(picks_dev));
  ASCEND_CALL(aclrtFree(picks_tiling_dev));
  ASCEND_CALL(aclrtFree(bias_dev));

  return COOMatrix(
      mat.num_rows, mat.num_cols,
      picked_row.CreateView({max_output}, picked_row->dtype),
      picked_col.CreateView({max_output}, picked_col->dtype),
      picked_idx.CreateView({max_output}, picked_idx->dtype));
#else
  LOG(FATAL) << "Ascend support is not compiled. "
                "Please compile with -DUSE_ASCEND=ON";
  return {};
#endif  // DGL_USE_ASCEND
}

template COOMatrix CSRRowWiseSamplingBiased<kDGLAscend, int32_t, float>(
    CSRMatrix, IdArray, int64_t, NDArray, FloatArray, bool);

template COOMatrix CSRRowWiseSamplingBiased<kDGLAscend, int64_t, float>(
    CSRMatrix, IdArray, int64_t, NDArray, FloatArray, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
