/**
 * Copyright (c) 2024 by Contributors
 * @file csr_row_wise_sampling_uniform.cc
 * @brief Ascend host launcher for uniform CSR row-wise sampling.
 *
 * Multi-core (v2): the host computes per-row pick counts from row degrees
 * (one CSRGetRowNNZ launch + one D2H copy), builds nnz-balanced row-range
 * partitions (spmm BuildBalancedPartitions pattern) plus per-block output
 * offsets as prefix sums, allocates the output exactly, and launches the
 * 40-core AIV kernel. Blocks write disjoint output ranges, so the kernel
 * needs no cross-block reduction.
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

extern "C" uint32_t aclrtlaunch_csr_row_wise_sampling_uniform_int32(
    uint32_t blockDim, aclrtStream stream, void* indptr, void* indices,
    void* data, void* rows, void* out_ptr, void* out_rows, void* out_cols,
    void* out_idxs, void* row_split, void* out_starts, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_row_wise_sampling_uniform_int64(
    uint32_t blockDim, aclrtStream stream, void* indptr, void* indices,
    void* data, void* rows, void* out_ptr, void* out_rows, void* out_cols,
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
#include <dgl/random.h>
#include <dgl/runtime/device_api.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "../array_op.h"
#include "csr_row_wise_sampling_uniform_tiling.h"

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
      device_id, ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE, &ub_bytes);
  if (err != ACL_SUCCESS ||
      ub_bytes <= static_cast<int64_t>(kUbReservedBytes) ||
      ub_bytes > (1 << 30)) {
    return kDefaultUbBytes - kUbReservedBytes;
  }
  // Reaching here means ub_bytes > kUbReservedBytes (checked above).
  return static_cast<uint32_t>(ub_bytes - kUbReservedBytes);
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

// Uploads a host uint32 table to device memory on the launch stream.
// The stream is synchronized BEFORE the caller's stack buffers go out of
// scope: an async copy only captures the source pointer, so returning
// without a sync would upload stack garbage (spmm precedes its cached
// uploads with the same sync).
void* UploadHostUInt32(const std::vector<uint32_t>& host, aclrtStream stream) {
  void* dev = nullptr;
  ASCEND_CALL(aclrtMalloc(
      &dev, host.size() * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpyAsync(
      dev, host.size() * sizeof(uint32_t), host.data(),
      host.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  return dev;
}

}  // namespace

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform(
    CSRMatrix mat, IdArray rows, int64_t num_samples, bool replace) {
#ifdef DGL_USE_ASCEND
  auto ctx = mat.indptr->ctx;
  CHECK(ctx.device_type == kDGLAscend)
      << "Expected Ascend device context for CSRRowWiseSamplingUniform";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  const bool select_all = (num_samples == -1);
  replace = (replace && !select_all);

  const int64_t num_rows = rows->shape[0];
  const uint8_t nbits = mat.indptr->dtype.bits;

  // num_samples == 0 implies !select_all (select_all means -1).
  if (num_rows == 0 || mat.indptr->shape[0] <= 1 || num_samples == 0) {
    IdArray empty_row = aten::NewIdArray(0, ctx, nbits);
    return COOMatrix(
        mat.num_rows, mat.num_cols, empty_row, empty_row, empty_row);
  }

  // Per-row pick counts (degrees come back to the host once).
  const uint32_t fanout = select_all ? 0u : static_cast<uint32_t>(num_samples);
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

  // nnz-balanced row partitions across all vector cores (spmm pattern).
  // blockDim is always the full vector-core count: the block count seen
  // by the kernel must match the tables' sizes, and blocks with no rows
  // exit immediately.
  const uint32_t block_dim = QueryVectorCoreCount(ctx.device_id);
  const std::vector<uint32_t> row_split =
      BuildBalancedPartitions(picks, block_dim);

  // Per-block output offsets as prefix sums of picks over row ranges.
  // Blocks write disjoint output slices; the last entry is the total.
  std::vector<uint32_t> out_starts(block_dim + 1, 0);
  {
    std::vector<uint32_t> prefix(num_rows + 1, 0);
    for (int64_t i = 0; i < num_rows; ++i) prefix[i + 1] = prefix[i] + picks[i];
    for (uint32_t b = 0; b <= block_dim; ++b)
      out_starts[b] = prefix[row_split[b]];
  }
  const int64_t max_output = out_starts[block_dim];
  CHECK(max_output <= static_cast<int64_t>(std::numeric_limits<IdType>::max()))
      << "Output size " << max_output << " exceeds IdType range";

  auto stream = dgl::runtime::getCurrentAscendStream();
  const bool has_data = aten::CSRHasData(mat);
  void* data_ptr = has_data ? mat.data->data : nullptr;

  uint32_t tiling_data[kTilingHeaderWords] = {
      static_cast<uint32_t>(num_rows),
      fanout,
      static_cast<uint32_t>(replace ? 1 : 0),
      static_cast<uint32_t>(has_data ? 1 : 0),
      static_cast<uint32_t>(RandomEngine::ThreadLocal()->RandInt(1000000000)),
      static_cast<uint32_t>(select_all ? 1 : 0),
      static_cast<uint32_t>(mat.num_rows),
      QueryUbAvailableBytes(ctx.device_id),
  };

  IdArray picked_row = aten::NewIdArray(max_output, ctx, nbits);
  IdArray picked_col = aten::NewIdArray(max_output, ctx, nbits);
  IdArray picked_idx = aten::NewIdArray(max_output, ctx, nbits);
  IdArray out_ptr = aten::NewIdArray(num_rows + 1, ctx, nbits);

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

  void* tiling_dev = nullptr;
  ASCEND_CALL(
      aclrtMalloc(&tiling_dev, sizeof(tiling_data), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpyAsync(
      tiling_dev, sizeof(tiling_data), tiling_data, sizeof(tiling_data),
      ACL_MEMCPY_HOST_TO_DEVICE, stream));
  // tiling_data is a stack array: wait for the copy to land before the
  // frame that owns it returns.
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  void* row_split_dev = UploadHostUInt32(row_split, stream);
  void* out_starts_dev = UploadHostUInt32(out_starts, stream);

  if (std::is_same<IdType, int32_t>::value) {
    aclError err = aclrtlaunch_csr_row_wise_sampling_uniform_int32(
        block_dim, stream, mat.indptr->data, mat.indices->data, data_ptr,
        rows->data, out_ptr->data, picked_row->data, picked_col->data,
        picked_idx->data, row_split_dev, out_starts_dev, tiling_dev);
    CHECK(err == ACL_SUCCESS)
        << "csr_row_wise_sampling_uniform_int32 launch failed: " << err;
  } else {
    aclError err = aclrtlaunch_csr_row_wise_sampling_uniform_int64(
        block_dim, stream, mat.indptr->data, mat.indices->data, data_ptr,
        rows->data, out_ptr->data, picked_row->data, picked_col->data,
        picked_idx->data, row_split_dev, out_starts_dev, tiling_dev);
    CHECK(err == ACL_SUCCESS)
        << "csr_row_wise_sampling_uniform_int64 launch failed: " << err;
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

template COOMatrix CSRRowWiseSamplingUniform<kDGLAscend, int32_t>(
    CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform<kDGLAscend, int64_t>(
    CSRMatrix, IdArray, int64_t, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
