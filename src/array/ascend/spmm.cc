#include <dgl/array.h>
#include <dgl/aten/csr.h>
#include <dgl/aten/array_ops.h>
#include <dgl/runtime/device_api.h>
#include "../kernel_decl.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <utility>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <string>
#include <dmlc/logging.h>
#include <cstdint>
#include <cstdlib>

#ifdef DGL_USE_ASCEND
#include <Python.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <acl/acl_op.h>
#define ASCEND_CALL(func)                                                \
  {                                                                      \
    aclError e = (func);                                                 \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e; \
  }

constexpr uint32_t windowSize = 16;
constexpr uint32_t tcBlockWidth = 16;
constexpr uint32_t cubeCoreCount = 20;
constexpr uint32_t vectorCoreCount = 40;
constexpr uint16_t kHalfOne = 0x3c00;

#ifdef DGL_USE_PYTORCH_NPU_STREAM
#include <torch_npu/csrc/core/npu/NPUStream.h>
static bool use_pytorch_stream() {
    static bool val = []() {
        const char* env = std::getenv("DGL_SPMM_USE_PYTORCH_STREAM");
        if (!env) return false;
        return env[0] == '1' && env[1] == '\0';
    }();
    return val;
}
#endif

static bool use_aiv_only_sum() {
    static bool val = []() {
        const char* env = std::getenv("DGL_SPMM_SUM_AIV_ONLY");
        if (!env) return false;
        return env[0] == '1' && env[1] == '\0';
    }();
    return val;
}

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

// Unified SPMM kernels (template-based, FP32+FP16)

// Tiling struct for unified SPMM kernel (mirrors spmm_unified_tiling.h)
struct SpmmUnifiedAivTilingData {
    uint32_t numDstRows;
    uint32_t numSrcRows;
    uint32_t featureDim;
    uint32_t nonZeroCount;
    uint32_t batchCount;
    uint32_t dtype;  // 0=FP32, 1=FP16
    uint32_t isCopyRhs;  // 0=copy_lhs (gather), 1=copy_rhs (sequential read)
};
struct SpmmUnifiedMaxMinTilingData {
    uint32_t numDstRows;
    uint32_t numSrcRows;
    uint32_t featureDim;
    uint32_t nonZeroCount;
    uint32_t batchCount;
    uint32_t dtype;
};

extern "C" uint32_t aclrtlaunch_spmm_unified_aiv(
    uint32_t blockDim, aclrtStream stream,
    void* feat, void* out, void* indptr, void* indices,
    void* row_split, void* tiling);
extern "C" uint32_t aclrtlaunch_spmm_unified_max(
    uint32_t blockDim, aclrtStream stream,
    void* feat, void* out, void* indptr, void* indices,
    void* row_split, void* tiling);
extern "C" uint32_t aclrtlaunch_spmm_unified_min(
    uint32_t blockDim, aclrtStream stream,
    void* feat, void* out, void* indptr, void* indices,
    void* row_split, void* tiling);

extern "C" uint32_t aclrtlaunch_spmm_sum(
    uint32_t blockDim, aclrtStream stream, void* denseBlockData,
    void* featureData, void* outputData, void* indptrData, void* indicesData,
    void* vectorWindowIdsData, void* vectorWinSplitData, void* cubeWindowIdsData,
    void* cubeWinSplitData, void* winEdgePtrData, void* colToEdgeData,
    uint32_t numDstRows, uint32_t numSrcRows, uint32_t featureDim,
    uint32_t nonZeroCount, uint32_t totalTcBlocks, uint32_t vectorWindowCount,
    uint32_t cubeWindowCount, uint32_t columnToEdgeLength);

extern "C" uint32_t aclrtlaunch_spmm_sum_aiv(
    uint32_t blockDim, aclrtStream stream, void* featureData, void* outputData,
    void* indptrData, void* indicesData, void* vectorRowSplitData,
    uint32_t numDstRows, uint32_t numSrcRows, uint32_t featureDim,
    uint32_t nonZeroCount);

extern "C" uint32_t aclrtlaunch_bspmm_sum(
    uint32_t blockDim, aclrtStream stream, void* denseBlockData,
    void* featureData, void* outputData, void* indptrData, void* indicesData,
    void* vectorWindowIdsData, void* vectorWinSplitData, void* cubeWindowIdsData,
    void* cubeWinSplitData, void* winEdgePtrData, void* colToEdgeData,
    uint32_t numDstRows, uint32_t numSrcRows, uint32_t featureDim,
    uint32_t nonZeroCount, uint32_t totalTcBlocks, uint32_t vectorWindowCount,
    uint32_t cubeWindowCount, uint32_t columnToEdgeLength, uint32_t batchCount);

extern "C" uint32_t aclrtlaunch_spmm_max(
    uint32_t blockDim, aclrtStream stream, void* featureData, void* outputData,
    void* indptrData, void* indicesData, void* vectorRowSplitData,
    uint32_t numDstRows, uint32_t numSrcRows, uint32_t featureDim,
    uint32_t nonZeroCount);

extern "C" uint32_t aclrtlaunch_bspmm_max(
    uint32_t blockDim, aclrtStream stream, void* featureData, void* outputData,
    void* indptrData, void* indicesData, void* vectorRowSplitData,
    uint32_t numDstRows, uint32_t numSrcRows, uint32_t featureDim,
    uint32_t nonZeroCount, uint32_t batchCount);

extern "C" uint32_t aclrtlaunch_spmm_min(
    uint32_t blockDim, aclrtStream stream, void* featureData, void* outputData,
    void* indptrData, void* indicesData, void* vectorRowSplitData,
    uint32_t numDstRows, uint32_t numSrcRows, uint32_t featureDim,
    uint32_t nonZeroCount);

extern "C" uint32_t aclrtlaunch_bspmm_min(
    uint32_t blockDim, aclrtStream stream, void* featureData, void* outputData,
    void* indptrData, void* indicesData, void* vectorRowSplitData,
    uint32_t numDstRows, uint32_t numSrcRows, uint32_t featureDim,
    uint32_t nonZeroCount, uint32_t batchCount);

template <typename T>
static std::vector<uint32_t> CopyDeviceArrayToHostUInt32(
    const T* device_ptr, size_t count, aclrtStream stream) {
  std::vector<T> host_raw(count);
  if (count > 0) {
    ASCEND_CALL(aclrtMemcpyAsync(
        host_raw.data(), count * sizeof(T), device_ptr, count * sizeof(T),
        ACL_MEMCPY_DEVICE_TO_HOST, stream));
    ASCEND_CALL(aclrtSynchronizeStream(stream));
  }
  std::vector<uint32_t> host(count);
  for (size_t i = 0; i < count; ++i) {
    host[i] = static_cast<uint32_t>(host_raw[i]);
  }
  return host;
}

static std::vector<uint32_t> BuildBalancedPartitions(
    const std::vector<uint32_t>& weights, uint32_t num_parts) {
  std::vector<uint32_t> boundaries(num_parts + 1, 0);
  if (num_parts == 0) return boundaries;

  uint32_t item_count = static_cast<uint32_t>(weights.size());
  boundaries[num_parts] = item_count;
  if (item_count == 0) return boundaries;

  if (item_count <= num_parts) {
    for (uint32_t i = 0; i <= item_count; ++i) boundaries[i] = i;
    for (uint32_t i = item_count + 1; i <= num_parts; ++i) boundaries[i] = item_count;
    return boundaries;
  }

  std::vector<double> prefix(item_count + 1, 0.0);
  for (uint32_t i = 0; i < item_count; ++i) {
    prefix[i + 1] = prefix[i] + static_cast<double>(weights[i]);
  }
  double total_weight = prefix[item_count];
  if (total_weight <= 0.0) {
    for (uint32_t part = 1; part < num_parts; ++part) {
      boundaries[part] = part * item_count / num_parts;
    }
    return boundaries;
  }

  uint32_t previous_boundary = 0;
  for (uint32_t part = 1; part < num_parts; ++part) {
    double target_weight = total_weight * part / num_parts;
    auto it = std::lower_bound(prefix.begin(), prefix.end(), target_weight);
    uint32_t boundary = static_cast<uint32_t>(it - prefix.begin());
    boundary = std::min(std::max(boundary, previous_boundary), item_count);
    boundaries[part] = boundary;
    previous_boundary = boundary;
  }
  return boundaries;
}

static float ChooseCubeWindowRatio(uint32_t feature_dim) {
  float ratio = 0.643f - 0.062f * std::log(static_cast<float>(feature_dim));
  return std::min(0.30f, std::max(0.095f, ratio));
}

static std::vector<uint32_t> BuildRowNnzBalancedPartitions(
    const std::vector<uint32_t>& row_pointers, uint32_t num_parts) {
  uint32_t num_rows = row_pointers.empty()
      ? 0
      : static_cast<uint32_t>(row_pointers.size() - 1);
  std::vector<uint32_t> row_nnz(num_rows, 0);
  for (uint32_t row = 0; row < num_rows; ++row) {
    row_nnz[row] = row_pointers[row + 1] - row_pointers[row];
  }
  return BuildBalancedPartitions(row_nnz, num_parts);
}

struct SpMMPreprocessCacheKey {
  int device_id;
  std::string reduce;
  const void* indptr_ptr;
  const void* indices_ptr;
  uint32_t num_rows;
  uint32_t num_cols;
  uint32_t num_edges;
  uint32_t out_dim;

  bool operator==(const SpMMPreprocessCacheKey& other) const {
    return device_id == other.device_id &&
        reduce == other.reduce &&
        indptr_ptr == other.indptr_ptr &&
        indices_ptr == other.indices_ptr &&
        num_rows == other.num_rows &&
        num_cols == other.num_cols &&
        num_edges == other.num_edges &&
        out_dim == other.out_dim;
  }
};

struct SpMMPreprocessCacheKeyHash {
  size_t operator()(const SpMMPreprocessCacheKey& key) const {
    size_t h = std::hash<int>{}(key.device_id);
    auto combine = [&h](size_t value) {
      h ^= value + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    combine(std::hash<std::string>{}(key.reduce));
    combine(std::hash<const void*>{}(key.indptr_ptr));
    combine(std::hash<const void*>{}(key.indices_ptr));
    combine(std::hash<uint32_t>{}(key.num_rows));
    combine(std::hash<uint32_t>{}(key.num_cols));
    combine(std::hash<uint32_t>{}(key.num_edges));
    combine(std::hash<uint32_t>{}(key.out_dim));
    return h;
  }
};

struct DeviceMemoryDeleter {
  void operator()(void* ptr) const {
    if (ptr != nullptr) {
      aclrtFree(ptr);
    }
  }
};

using DeviceMemoryPtr = std::unique_ptr<void, DeviceMemoryDeleter>;

static DeviceMemoryPtr MakeDeviceBuffer(
    const void* src, size_t bytes, aclrtStream stream) {
  void* dst = nullptr;
  size_t alloc_bytes = std::max<size_t>(bytes, 1);
  ASCEND_CALL(aclrtMalloc(&dst, alloc_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  if (bytes > 0) {
    ASCEND_CALL(aclrtMemcpyAsync(
        dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream));
  }
  return DeviceMemoryPtr(dst);
}

struct MaxMinPreprocessCacheValue {
  DeviceMemoryPtr vector_row_split_dev;
};

struct SumPreprocessCacheValue {
  DeviceMemoryPtr dense_blocks_dev;
  DeviceMemoryPtr vector_window_ids_dev;
  DeviceMemoryPtr vector_core_boundaries_dev;
  DeviceMemoryPtr cube_window_ids_dev;
  DeviceMemoryPtr cube_core_boundaries_dev;
  DeviceMemoryPtr win_edge_ptr_dev;
  DeviceMemoryPtr column_to_edge_dev;
  uint32_t total_tc_blocks = 0;
  uint32_t vector_window_count = 0;
  uint32_t cube_window_count = 0;
  uint32_t column_to_edge_length = 0;
};

static std::unordered_map<
    SpMMPreprocessCacheKey,
    std::shared_ptr<MaxMinPreprocessCacheValue>,
    SpMMPreprocessCacheKeyHash>
    g_maxmin_preprocess_cache;
static std::unordered_map<
    SpMMPreprocessCacheKey,
    std::shared_ptr<SumPreprocessCacheValue>,
    SpMMPreprocessCacheKeyHash>
    g_sum_preprocess_cache;
static std::mutex g_spmm_preprocess_cache_mutex;

#endif

namespace dgl {
namespace aten {
/**
 * @brief Ascend NPU implementation of SpMM on CSR format using AscendC kernel.
 * 
 * This implementation uses AscendC kernel with CopyIn-Compute-CopyOut framework.
 * The computation is performed directly on NPU using optimized AscendC kernels.
 * 
 * @note Only supports copy_lhs + sum operation currently
 */
template <typename IdType, typename DType>
void SpMMCsrAscend(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {

  // Validate operation — copy_rhs is handled by the caller (SpMMCsrUnified
  // via GatherByIndex), so SpMMCsrAscend only sees copy_lhs after pre-gather.
  if (op != "copy_lhs" || (reduce != "sum" && reduce != "max" && reduce != "min")) {
    LOG(FATAL) << "SpMMCsrAscend only supports copy_lhs+sum/max/min operation. "
               << "Got: op=" << op << ", reduce=" << reduce;
  }

#ifdef DGL_USE_ASCEND
  
  DGLContext ctx = ufeat->ctx;
  CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));
  
  int64_t num_rows = csr.num_rows;
  int64_t num_cols = csr.num_cols;
  int64_t num_edges = csr.indices->shape[0];
  bool use_bspmm = (ufeat->ndim == 3);
  CHECK(ufeat->ndim == 2 || use_bspmm)
      << "SpMMCsrAscend only supports 2D SpMM or 3D BSpMM input features. Got ndim="
      << ufeat->ndim;
  int64_t batch_count = use_bspmm ? ufeat->shape[1] : 1;
  int64_t out_dim = use_bspmm ? ufeat->shape[2] : ((out->ndim > 1) ? out->shape[1] : 1);

  
  const IdType* indptr_ptr = static_cast<const IdType*>(csr.indptr->data);
  const IdType* indices_ptr = static_cast<const IdType*>(csr.indices->data);
  
  aclrtStream stream = nullptr;  // Use default stream

  uint32_t num_rows_u32 = static_cast<uint32_t>(num_rows);
  uint32_t num_cols_u32 = static_cast<uint32_t>(num_cols);
  uint32_t num_edges_u32 = static_cast<uint32_t>(num_edges);
  uint32_t out_dim_u32 = static_cast<uint32_t>(out_dim);
  SpMMPreprocessCacheKey cache_key{
      ctx.device_id,
      reduce,
      static_cast<const void*>(indptr_ptr),
      static_cast<const void*>(indices_ptr),
      num_rows_u32,
      num_cols_u32,
      num_edges_u32,
      out_dim_u32};

  if (reduce == "max" || reduce == "min") {
    std::shared_ptr<MaxMinPreprocessCacheValue> cache_value;
    {
      std::lock_guard<std::mutex> lock(g_spmm_preprocess_cache_mutex);
      auto it = g_maxmin_preprocess_cache.find(cache_key);
      if (it != g_maxmin_preprocess_cache.end()) {
        cache_value = it->second;
      }
    }

    if (!cache_value) {
      std::vector<uint32_t> row_pointers =
          CopyDeviceArrayToHostUInt32(indptr_ptr, static_cast<size_t>(num_rows + 1), stream);
      std::vector<uint32_t> vector_core_row_split =
          BuildRowNnzBalancedPartitions(row_pointers, vectorCoreCount);

      auto new_cache_value = std::make_shared<MaxMinPreprocessCacheValue>();
      new_cache_value->vector_row_split_dev = MakeDeviceBuffer(
          vector_core_row_split.data(),
          vector_core_row_split.size() * sizeof(uint32_t), stream);
      ASCEND_CALL(aclrtSynchronizeStream(stream));

      std::lock_guard<std::mutex> lock(g_spmm_preprocess_cache_mutex);
      auto [it, inserted] =
          g_maxmin_preprocess_cache.emplace(cache_key, new_cache_value);
      cache_value = inserted ? new_cache_value : it->second;
      if (!inserted) {
        LOG(INFO) << "[Ascend][SpMM][Cache] reused concurrent max/min cache reduce="
                  << reduce << " device=" << ctx.device_id;
      }
    }

    void* vector_row_split_dev = cache_value->vector_row_split_dev.get();

    ASCEND_CALL(aclrtMemsetAsync(out->data, out.GetSize(), 0, out.GetSize(), stream));
    uint32_t blockDim = vectorCoreCount;
    aclError launch_err = ACL_SUCCESS;
    if (reduce == "max") {
      launch_err = use_bspmm ? ACLRT_LAUNCH_KERNEL(bspmm_max)(
          blockDim, stream, ufeat->data, out->data,
          const_cast<void*>(static_cast<const void*>(indptr_ptr)),
          const_cast<void*>(static_cast<const void*>(indices_ptr)),
          vector_row_split_dev, num_rows_u32,
          num_cols_u32, out_dim_u32,
          num_edges_u32, static_cast<uint32_t>(batch_count))
          : ACLRT_LAUNCH_KERNEL(spmm_max)(
          blockDim, stream, ufeat->data, out->data,
          const_cast<void*>(static_cast<const void*>(indptr_ptr)),
          const_cast<void*>(static_cast<const void*>(indices_ptr)),
          vector_row_split_dev, num_rows_u32,
          num_cols_u32, out_dim_u32,
          num_edges_u32);
    } else {
      launch_err = use_bspmm ? ACLRT_LAUNCH_KERNEL(bspmm_min)(
          blockDim, stream, ufeat->data, out->data,
          const_cast<void*>(static_cast<const void*>(indptr_ptr)),
          const_cast<void*>(static_cast<const void*>(indices_ptr)),
          vector_row_split_dev, num_rows_u32,
          num_cols_u32, out_dim_u32,
          num_edges_u32, static_cast<uint32_t>(batch_count))
          : ACLRT_LAUNCH_KERNEL(spmm_min)(
          blockDim, stream, ufeat->data, out->data,
          const_cast<void*>(static_cast<const void*>(indptr_ptr)),
          const_cast<void*>(static_cast<const void*>(indices_ptr)),
          vector_row_split_dev, num_rows_u32,
          num_cols_u32, out_dim_u32,
          num_edges_u32);
    }
    if (launch_err != ACL_SUCCESS) {
      LOG(FATAL) << "spmm_" << reduce
                 << " kernel launch failed with error code: " << launch_err;
    }
    ASCEND_CALL(aclrtSynchronizeStream(stream));
    return;
  }

  // AIV-only sum: bypass sum preprocess entirely
  if (!use_bspmm && use_aiv_only_sum()) {
    std::vector<uint32_t> row_pointers =
        CopyDeviceArrayToHostUInt32(indptr_ptr, static_cast<size_t>(num_rows + 1), stream);
    std::vector<uint32_t> row_split =
        BuildRowNnzBalancedPartitions(row_pointers, vectorCoreCount);
    auto split_dev = MakeDeviceBuffer(
        row_split.data(), row_split.size() * sizeof(uint32_t), stream);
    ASCEND_CALL(aclrtSynchronizeStream(stream));
    ASCEND_CALL(aclrtMemsetAsync(out->data, out.GetSize(), 0, out.GetSize(), stream));
    uint32_t blockDim = vectorCoreCount;
    aclError launch_err = ACLRT_LAUNCH_KERNEL(spmm_sum_aiv)(
        blockDim, stream, ufeat->data, out->data,
        const_cast<void*>(static_cast<const void*>(indptr_ptr)),
        const_cast<void*>(static_cast<const void*>(indices_ptr)),
        split_dev.get(), num_rows_u32, num_cols_u32,
        out_dim_u32, num_edges_u32);
    if (launch_err != ACL_SUCCESS) {
      LOG(FATAL) << "spmm_sum_aiv kernel launch failed with error code: " << launch_err;
    }
    ASCEND_CALL(aclrtSynchronizeStream(stream));
    return;
  }

  // Original sum path with preprocess cache
  std::shared_ptr<SumPreprocessCacheValue> cache_value;
  {
    std::lock_guard<std::mutex> lock(g_spmm_preprocess_cache_mutex);
    auto it = g_sum_preprocess_cache.find(cache_key);
    if (it != g_sum_preprocess_cache.end()) {
      cache_value = it->second;
    }
  }

  if (!cache_value) {
    std::vector<uint32_t> row_pointers =
        CopyDeviceArrayToHostUInt32(indptr_ptr, static_cast<size_t>(num_rows + 1), stream);
    std::vector<uint32_t> column_indices =
        CopyDeviceArrayToHostUInt32(indices_ptr, static_cast<size_t>(num_edges), stream);

    uint32_t num_windows =
        (num_rows_u32 + windowSize - 1) / windowSize;
    uint32_t top_k_windows = static_cast<uint32_t>(std::ceil(
        num_windows * ChooseCubeWindowRatio(out_dim_u32)));
    top_k_windows = std::min(top_k_windows, num_windows);

    struct WindowInfo {
      float density = 0.0f;
      std::vector<uint32_t> unique_columns;
    };

    std::vector<WindowInfo> windows(num_windows);
    std::vector<uint32_t> non_empty_window_ids;
    for (uint32_t window_id = 0; window_id < num_windows; ++window_id) {
      uint32_t start_node = window_id * windowSize;
      uint32_t end_node = std::min(start_node + windowSize, num_rows_u32);
      uint32_t edge_start = row_pointers[start_node];
      uint32_t edge_end = row_pointers[end_node];
      if (edge_start == edge_end) continue;

      auto& window = windows[window_id];
      window.unique_columns.assign(
          column_indices.begin() + edge_start, column_indices.begin() + edge_end);
      std::sort(window.unique_columns.begin(), window.unique_columns.end());
      window.unique_columns.erase(
          std::unique(window.unique_columns.begin(), window.unique_columns.end()),
          window.unique_columns.end());

      uint32_t column_count = static_cast<uint32_t>(window.unique_columns.size());
      uint32_t aligned_columns =
          ((column_count + tcBlockWidth - 1) / tcBlockWidth) *
          tcBlockWidth;
      window.density = static_cast<float>(edge_end - edge_start) /
          static_cast<float>((end_node - start_node) * aligned_columns);
      non_empty_window_ids.push_back(window_id);
    }

    std::sort(non_empty_window_ids.begin(), non_empty_window_ids.end(),
              [&windows](uint32_t lhs, uint32_t rhs) {
                return windows[lhs].density > windows[rhs].density;
              });
    std::vector<uint8_t> is_cube_window(num_windows, 0);
    uint32_t selected_cube_windows =
        std::min(top_k_windows, static_cast<uint32_t>(non_empty_window_ids.size()));
    for (uint32_t i = 0; i < selected_cube_windows; ++i) {
      is_cube_window[non_empty_window_ids[i]] = 1;
    }

    std::vector<uint32_t> column_to_edge, cube_window_ids, vector_window_ids, win_edge_ptr{0};
    std::vector<uint16_t> dense_blocks;
    uint32_t total_tc_blocks = 0;
    for (uint32_t window_id = 0; window_id < num_windows; ++window_id) {
      const auto& unique_columns = windows[window_id].unique_columns;
      if (unique_columns.empty() || !is_cube_window[window_id]) {
        vector_window_ids.push_back(window_id);
        continue;
      }

      cube_window_ids.push_back(window_id);
      column_to_edge.insert(column_to_edge.end(), unique_columns.begin(), unique_columns.end());
      win_edge_ptr.push_back(static_cast<uint32_t>(column_to_edge.size()));
      uint32_t tc_blocks =
          (static_cast<uint32_t>(unique_columns.size()) + tcBlockWidth - 1) /
          tcBlockWidth;
      uint32_t padded_columns = tc_blocks * tcBlockWidth;
      total_tc_blocks += tc_blocks;
      size_t block_offset = dense_blocks.size();
      dense_blocks.resize(block_offset + windowSize * padded_columns, 0);

      uint32_t start_node = window_id * windowSize;
      uint32_t end_node = std::min(start_node + windowSize, num_rows_u32);
      for (uint32_t row = start_node; row < end_node; ++row) {
        for (uint32_t edge = row_pointers[row]; edge < row_pointers[row + 1]; ++edge) {
          auto it = std::lower_bound(unique_columns.begin(), unique_columns.end(), column_indices[edge]);
          uint32_t local_column = static_cast<uint32_t>(it - unique_columns.begin());
          uint32_t local_row = row - start_node;
          dense_blocks[block_offset + local_row * padded_columns + local_column] = kHalfOne;
        }
      }
    }

    std::vector<uint32_t> cube_work;
    cube_work.reserve(cube_window_ids.size());
    for (size_t i = 0; i < cube_window_ids.size(); ++i) {
      uint32_t column_count = win_edge_ptr[i + 1] - win_edge_ptr[i];
      cube_work.push_back((column_count + tcBlockWidth - 1) / tcBlockWidth);
    }
    std::vector<uint32_t> vector_work;
    vector_work.reserve(vector_window_ids.size());
    for (uint32_t window_id : vector_window_ids) {
      uint32_t start_node = window_id * windowSize;
      uint32_t end_node = std::min(start_node + windowSize, num_rows_u32);
      vector_work.push_back(row_pointers[end_node] - row_pointers[start_node]);
    }
    std::vector<uint32_t> cube_core_boundaries =
        BuildBalancedPartitions(cube_work, cubeCoreCount);
    std::vector<uint32_t> vector_core_boundaries =
        BuildBalancedPartitions(vector_work, vectorCoreCount);

    auto new_cache_value = std::make_shared<SumPreprocessCacheValue>();
    new_cache_value->dense_blocks_dev = MakeDeviceBuffer(
        dense_blocks.data(), dense_blocks.size() * sizeof(uint16_t), stream);
    new_cache_value->vector_window_ids_dev = MakeDeviceBuffer(
        vector_window_ids.data(), vector_window_ids.size() * sizeof(uint32_t), stream);
    new_cache_value->vector_core_boundaries_dev = MakeDeviceBuffer(
        vector_core_boundaries.data(), vector_core_boundaries.size() * sizeof(uint32_t), stream);
    new_cache_value->cube_window_ids_dev = MakeDeviceBuffer(
        cube_window_ids.data(), cube_window_ids.size() * sizeof(uint32_t), stream);
    new_cache_value->cube_core_boundaries_dev = MakeDeviceBuffer(
        cube_core_boundaries.data(), cube_core_boundaries.size() * sizeof(uint32_t), stream);
    new_cache_value->win_edge_ptr_dev = MakeDeviceBuffer(
        win_edge_ptr.data(), win_edge_ptr.size() * sizeof(uint32_t), stream);
    new_cache_value->column_to_edge_dev = MakeDeviceBuffer(
        column_to_edge.data(), column_to_edge.size() * sizeof(uint32_t), stream);
    new_cache_value->total_tc_blocks = total_tc_blocks;
    new_cache_value->vector_window_count = static_cast<uint32_t>(vector_window_ids.size());
    new_cache_value->cube_window_count = static_cast<uint32_t>(cube_window_ids.size());
    new_cache_value->column_to_edge_length = static_cast<uint32_t>(column_to_edge.size());
    ASCEND_CALL(aclrtSynchronizeStream(stream));

    std::lock_guard<std::mutex> lock(g_spmm_preprocess_cache_mutex);
    auto [it, inserted] =
        g_sum_preprocess_cache.emplace(cache_key, new_cache_value);
    cache_value = inserted ? new_cache_value : it->second;
    if (!inserted) {
      LOG(INFO) << "[Ascend][SpMM][Cache] reused concurrent sum cache reduce="
                << reduce << " device=" << ctx.device_id;
    }
  }

  void* dense_blocks_dev = cache_value->dense_blocks_dev.get();
  void* vector_window_ids_dev = cache_value->vector_window_ids_dev.get();
  void* vector_core_boundaries_dev = cache_value->vector_core_boundaries_dev.get();
  void* cube_window_ids_dev = cache_value->cube_window_ids_dev.get();
  void* cube_core_boundaries_dev = cache_value->cube_core_boundaries_dev.get();
  void* win_edge_ptr_dev = cache_value->win_edge_ptr_dev.get();
  void* column_to_edge_dev = cache_value->column_to_edge_dev.get();

  ASCEND_CALL(aclrtMemsetAsync(out->data, out.GetSize(), 0, out.GetSize(), stream));

  uint32_t blockDim = cubeCoreCount;
  aclError launch_err = use_bspmm ? ACLRT_LAUNCH_KERNEL(bspmm_sum)(
      blockDim, stream, dense_blocks_dev, ufeat->data, out->data,
      const_cast<void*>(static_cast<const void*>(indptr_ptr)),
      const_cast<void*>(static_cast<const void*>(indices_ptr)),
      vector_window_ids_dev, vector_core_boundaries_dev, cube_window_ids_dev,
      cube_core_boundaries_dev, win_edge_ptr_dev, column_to_edge_dev,
      num_rows_u32, num_cols_u32,
      out_dim_u32, num_edges_u32,
      cache_value->total_tc_blocks, cache_value->vector_window_count,
      cache_value->cube_window_count,
      cache_value->column_to_edge_length,
      static_cast<uint32_t>(batch_count))
      : ACLRT_LAUNCH_KERNEL(spmm_sum)(
      blockDim, stream, dense_blocks_dev, ufeat->data, out->data,
      const_cast<void*>(static_cast<const void*>(indptr_ptr)),
      const_cast<void*>(static_cast<const void*>(indices_ptr)),
      vector_window_ids_dev, vector_core_boundaries_dev, cube_window_ids_dev,
      cube_core_boundaries_dev, win_edge_ptr_dev, column_to_edge_dev,
      num_rows_u32, num_cols_u32,
      out_dim_u32, num_edges_u32,
      cache_value->total_tc_blocks, cache_value->vector_window_count,
      cache_value->cube_window_count,
      cache_value->column_to_edge_length);
  if (launch_err != ACL_SUCCESS) {
    LOG(FATAL) << "spmm_sum kernel launch failed with error code: " << launch_err;
  }
  ASCEND_CALL(aclrtSynchronizeStream(stream));

#else
  LOG(FATAL) << "Ascend support is not compiled. Please compile with -DUSE_ASCEND=ON";
#endif
}


// Helper: cast int64 CSR to int32
static CSRMatrix CastCSRToInt32SpMM(const CSRMatrix& csr) {
  DGLContext cpu_ctx{kDGLCPU, 0};
  DGLDataType int32_type{kDGLInt, 32, 1};
  auto indptr_cpu = csr.indptr.CopyTo(cpu_ctx);
  auto indices_cpu = csr.indices.CopyTo(cpu_ctx);
  int64_t nnz = csr.indices->shape[0];
  int64_t nrows = csr.num_rows;
  NDArray indptr32_cpu = NDArray::Empty({nrows + 1}, int32_type, cpu_ctx);
  NDArray indices32_cpu = NDArray::Empty({nnz}, int32_type, cpu_ctx);
  const int64_t* ip = static_cast<const int64_t*>(indptr_cpu->data);
  const int64_t* idx = static_cast<const int64_t*>(indices_cpu->data);
  int32_t* ip32 = static_cast<int32_t*>(indptr32_cpu->data);
  int32_t* idx32 = static_cast<int32_t*>(indices32_cpu->data);
  for (int64_t i = 0; i <= nrows; ++i) ip32[i] = static_cast<int32_t>(ip[i]);
  for (int64_t i = 0; i < nnz; ++i) idx32[i] = static_cast<int32_t>(idx[i]);
  NDArray data32 = csr.data;
  if (!IsNullArray(csr.data)) {
    auto data_cpu = csr.data.CopyTo(cpu_ctx);
    data32 = NDArray::Empty({nnz}, int32_type, cpu_ctx);
    const int64_t* d = static_cast<const int64_t*>(data_cpu->data);
    int32_t* d32 = static_cast<int32_t*>(data32->data);
    for (int64_t i = 0; i < nnz; ++i) d32[i] = static_cast<int32_t>(d[i]);
    data32 = data32.CopyTo(csr.indptr->ctx);
  }
  return CSRMatrix(nrows, csr.num_cols,
                    indptr32_cpu.CopyTo(csr.indptr->ctx),
                    indices32_cpu.CopyTo(csr.indptr->ctx),
                    data32, csr.sorted);
}

// Helper: gather NDArray by index, supporting multi-dimensional tensors
// For multi-dim tensors, copies to CPU, gathers, copies back (small overhead
// compared to the NPU kernel itself for typical graph sizes)
static NDArray GatherByIndex(NDArray src, NDArray index) {
  if (IsNullArray(index)) return src;
  if (src->ndim <= 1) return IndexSelect(src, index);

  // Multi-dim gather on CPU (dtype-agnostic: use byte-level memcpy)
  DGLContext cpu_ctx{kDGLCPU, 0};
  NDArray src_cpu = src.CopyTo(cpu_ctx);
  NDArray idx_cpu = index.CopyTo(cpu_ctx);

  const int32_t* idx_ptr = idx_cpu.Ptr<int32_t>();
  int64_t nnz = index->shape[0];
  int64_t feat_dim = 1;
  for (int i = 1; i < src->ndim; ++i) feat_dim *= src->shape[i];
  int64_t dtype_bytes = src->dtype.bits / 8;

  // Create output on CPU
  std::vector<int64_t> out_shape(src->ndim);
  out_shape[0] = nnz;
  for (int i = 1; i < src->ndim; ++i) out_shape[i] = src->shape[i];
  NDArray out_cpu = NDArray::Empty(out_shape, src->dtype, cpu_ctx);

  const char* src_bytes = static_cast<const char*>(src_cpu->data);
  char* out_bytes = static_cast<char*>(out_cpu->data);
  int64_t row_bytes = feat_dim * dtype_bytes;
  for (int64_t i = 0; i < nnz; ++i) {
    int32_t eid = idx_ptr[i];
    std::memcpy(out_bytes + i * row_bytes, src_bytes + eid * row_bytes,
                row_bytes);
  }

  // Copy back to NPU
  return out_cpu.CopyTo(src->ctx);
}

// ============================================================================
// SpMMCsrUnified — unified NPU kernel for FP32 and FP16
// Uses spmm_unified_aiv (sum) / spmm_unified_max / spmm_unified_min
// All paths use pure Vector Core (40 cores), Cube+Vector reserved for future
// ============================================================================
static std::vector<uint32_t> BuildRowSplitUnified(
    const uint32_t* indptr, int64_t num_rows, aclrtStream stream) {
  std::vector<uint32_t> host_indptr(num_rows + 1);
  aclrtMemcpy(host_indptr.data(), sizeof(uint32_t) * (num_rows + 1),
              indptr, sizeof(uint32_t) * (num_rows + 1), ACL_MEMCPY_DEVICE_TO_HOST);
  // Build balanced partition by nnz across 40 vector cores
  std::vector<uint32_t> split(41, 0);
  split[40] = static_cast<uint32_t>(num_rows);
  uint32_t total_nnz = host_indptr.back();
  for (uint32_t part = 1; part < 40; ++part) {
    uint32_t target = static_cast<uint32_t>(
        (static_cast<uint64_t>(part) * total_nnz) / 40);
    split[part] = static_cast<uint32_t>(
        std::lower_bound(host_indptr.begin(), host_indptr.end(), target) -
        host_indptr.begin());
  }
  return split;
}

template <typename IdType, typename DType>
static void SpMMCsrUnified(
    const std::string& op, const std::string& reduce,
    const CSRMatrix& csr, NDArray ufeat, NDArray out,
    std::vector<NDArray> out_aux) {
  if (!((op == "copy_lhs" || op == "copy_rhs") &&
        (reduce == "sum" || reduce == "max" || reduce == "min"))) {
    LOG(FATAL) << "SpMMCsrUnified only supports copy_lhs/copy_rhs + sum/max/min";
  }

  DGLContext ctx = out->ctx;
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  int64_t num_rows = csr.num_rows;
  int64_t num_cols = csr.num_cols;
  int64_t nnz = csr.indices->shape[0];
  bool use_bspmm = (ufeat->ndim == 3);
  int64_t batch_count = use_bspmm ? ufeat->shape[1] : 1;
  int64_t feat_dim = use_bspmm ? ufeat->shape[2] : ((out->ndim > 1) ? out->shape[1] : 1);

  if (num_rows == 0 || nnz == 0) {
    aclrtMemsetAsync(out->data, out.GetSize(), 0, out.GetSize(), nullptr);
    return;
  }

  aclrtStream stream = nullptr;

  // Build row split for load balancing
  const IdType* indptr_ptr = static_cast<const IdType*>(csr.indptr->data);
  // Copy indptr to host for split calculation
  std::vector<uint32_t> host_indptr(num_rows + 1);
  // Handle int64 indptr
  if (std::is_same<IdType, int64_t>::value) {
    std::vector<int64_t> host_indptr64(num_rows + 1);
    aclrtMemcpy(host_indptr64.data(), sizeof(int64_t) * (num_rows + 1),
                indptr_ptr, sizeof(int64_t) * (num_rows + 1), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int64_t i = 0; i <= num_rows; ++i) host_indptr[i] = static_cast<uint32_t>(host_indptr64[i]);
  } else {
    aclrtMemcpy(host_indptr.data(), sizeof(uint32_t) * (num_rows + 1),
                indptr_ptr, sizeof(uint32_t) * (num_rows + 1), ACL_MEMCPY_DEVICE_TO_HOST);
  }

  std::vector<uint32_t> split(41, 0);
  split[40] = static_cast<uint32_t>(num_rows);
  uint32_t total_nnz = host_indptr.back();
  for (uint32_t part = 1; part < 40; ++part) {
    uint32_t target = static_cast<uint32_t>(
        (static_cast<uint64_t>(part) * total_nnz) / 40);
    split[part] = static_cast<uint32_t>(
        std::lower_bound(host_indptr.begin(), host_indptr.end(), target) -
        host_indptr.begin());
  }

  // Copy split to device
  void* split_dev = nullptr;
  ASCEND_CALL(aclrtMalloc(&split_dev, 41 * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(split_dev, 41 * sizeof(uint32_t), split.data(),
                           41 * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE));

  // Build tiling
  SpmmUnifiedAivTilingData tiling;
  tiling.numDstRows = static_cast<uint32_t>(num_rows);
  tiling.numSrcRows = static_cast<uint32_t>(num_cols);
  tiling.featureDim = static_cast<uint32_t>(feat_dim);
  tiling.nonZeroCount = static_cast<uint32_t>(nnz);
  tiling.batchCount = static_cast<uint32_t>(batch_count);
  tiling.dtype = (sizeof(DType) == 4) ? 0 : 1;  // 0=FP32, 1=FP16
  tiling.isCopyRhs = (op == "copy_rhs") ? 1 : 0;

  void* tiling_dev = nullptr;
  ASCEND_CALL(aclrtMalloc(&tiling_dev, sizeof(tiling), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(tiling_dev, sizeof(tiling), &tiling,
                           sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE));

  // Zero output
  ASCEND_CALL(aclrtMemsetAsync(out->data, out.GetSize(), 0, out.GetSize(), stream));

  // Launch kernel
  uint32_t blockDim = 40;
  aclError err;
  if (reduce == "sum") {
    err = ACLRT_LAUNCH_KERNEL(spmm_unified_aiv)(
        blockDim, stream, ufeat->data, out->data,
        const_cast<void*>(static_cast<const void*>(csr.indptr->data)),
        const_cast<void*>(static_cast<const void*>(csr.indices->data)),
        split_dev, tiling_dev);
  } else if (reduce == "max") {
    err = ACLRT_LAUNCH_KERNEL(spmm_unified_max)(
        blockDim, stream, ufeat->data, out->data,
        const_cast<void*>(static_cast<const void*>(csr.indptr->data)),
        const_cast<void*>(static_cast<const void*>(csr.indices->data)),
        split_dev, tiling_dev);
  } else {
    err = ACLRT_LAUNCH_KERNEL(spmm_unified_min)(
        blockDim, stream, ufeat->data, out->data,
        const_cast<void*>(static_cast<const void*>(csr.indptr->data)),
        const_cast<void*>(static_cast<const void*>(csr.indices->data)),
        split_dev, tiling_dev);
  }
  if (err != ACL_SUCCESS) {
    LOG(FATAL) << "spmm_unified_" << reduce << " launch failed: " << err;
  }
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(split_dev));
  ASCEND_CALL(aclrtFree(tiling_dev));
}

// Template specializations for CSR SpMM
template <>
void SpMMCsr<kDGLAscend, int32_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  if ((op == "copy_lhs" || op == "copy_rhs") &&
      (reduce == "sum" || reduce == "max" || reduce == "min") &&
      (op == "copy_lhs" ? ufeat->ndim : efeat->ndim) <= 3) {
    NDArray feat;
    if (op == "copy_rhs") {
      // InCSR's data array contains edge reorder index from COOToCSR.
      // Use it to gather efeat into CSC row order: feat[j] = efeat[data[j]]
      feat = IsNullArray(csr.data) ? efeat : GatherByIndex(efeat, csr.data);
    } else {
      feat = ufeat;
    }
    SpMMCsrUnified<int32_t, float>(op, reduce, csr, feat, out, out_aux);
  } else {
    // CPU fallback for non-copy_lhs ops or multi-dim features
    DGLContext cpu_ctx{kDGLCPU, 0};
    CSRMatrix csr_cpu = csr.CopyTo(cpu_ctx);
    NDArray ufeat_cpu = IsNullArray(ufeat) ? ufeat : ufeat.CopyTo(cpu_ctx);
    NDArray efeat_cpu = IsNullArray(efeat) ? efeat : efeat.CopyTo(cpu_ctx);
    NDArray out_cpu = out.CopyTo(cpu_ctx);
    std::vector<NDArray> out_aux_cpu;
    for (auto& a : out_aux) out_aux_cpu.push_back(IsNullArray(a) ? a : a.CopyTo(cpu_ctx));
    SpMMCsr<kDGLCPU, int32_t, float>(op, reduce, bcast, csr_cpu,
                                      ufeat_cpu, efeat_cpu, out_cpu, out_aux_cpu);
    out_cpu.CopyTo(out);
    for (size_t i = 0; i < out_aux.size(); ++i) {
      if (!IsNullArray(out_aux[i]) && !IsNullArray(out_aux_cpu[i]))
        out_aux_cpu[i].CopyTo(out_aux[i]);
    }
  }
}

template <>
void SpMMCsr<kDGLAscend, int32_t, uint16_t>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  // FP16 copy_rhs: not supported on NPU (SpMMCsrAscend only does copy_lhs,
  // SpMMCsrUnified only supports float). Falls back to CPU for correctness.
  // CPU uint16_t SpMMCsr is not instantiated, so LOG(FATAL) for now.
  // FP32 copy_rhs is fully supported via SpMMCsrUnified.
  if (op == "copy_rhs") {
    LOG(FATAL) << "FP16 copy_rhs not supported on Ascend NPU. "
               << "Use FP32 for copy_e_sum operations.";
  }
  SpMMCsrAscend<int32_t, uint16_t>(op, reduce, bcast, csr, ufeat, efeat, out, out_aux);
}

template <>
void SpMMCsr<kDGLAscend, int64_t, uint16_t>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  CSRMatrix csr32 = CastCSRToInt32SpMM(csr);
  if (op == "copy_rhs") {
    LOG(FATAL) << "FP16 copy_rhs not supported on Ascend NPU. "
               << "Use FP32 for copy_e_sum operations.";
  }
  SpMMCsrAscend<int32_t, uint16_t>(op, reduce, bcast, csr32, ufeat, efeat, out, out_aux);
}

template <>
void SpMMCsr<kDGLAscend, int64_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  CSRMatrix csr32 = CastCSRToInt32SpMM(csr);
  if ((op == "copy_lhs" || op == "copy_rhs") &&
      (reduce == "sum" || reduce == "max" || reduce == "min") &&
      (op == "copy_lhs" ? ufeat->ndim : efeat->ndim) <= 3) {
    NDArray feat;
    if (op == "copy_rhs") {
      feat = IsNullArray(csr32.data) ? efeat : GatherByIndex(efeat, csr32.data);
    } else {
      feat = ufeat;
    }
    SpMMCsrUnified<int32_t, float>(op, reduce, csr32, feat, out, out_aux);
  } else {
    DGLContext cpu_ctx{kDGLCPU, 0};
    CSRMatrix csr32_cpu = csr32.CopyTo(cpu_ctx);
    NDArray ufeat_cpu = IsNullArray(ufeat) ? ufeat : ufeat.CopyTo(cpu_ctx);
    NDArray efeat_cpu = IsNullArray(efeat) ? efeat : efeat.CopyTo(cpu_ctx);
    NDArray out_cpu = out.CopyTo(cpu_ctx);
    std::vector<NDArray> out_aux_cpu;
    for (auto& a : out_aux) out_aux_cpu.push_back(IsNullArray(a) ? a : a.CopyTo(cpu_ctx));
    SpMMCsr<kDGLCPU, int32_t, float>(op, reduce, bcast, csr32_cpu,
                                      ufeat_cpu, efeat_cpu, out_cpu, out_aux_cpu);
    out_cpu.CopyTo(out);
    for (size_t i = 0; i < out_aux.size(); ++i) {
      if (!IsNullArray(out_aux[i]) && !IsNullArray(out_aux_cpu[i]))
        out_aux_cpu[i].CopyTo(out_aux[i]);
    }
  }
}

template <>
void SpMMCsr<kDGLAscend, int32_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
    LOG(FATAL) << "Double precision not fully supported on Ascend yet.";
}

template <>
void SpMMCsr<kDGLAscend, int64_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
    LOG(FATAL) << "Double precision not fully supported on Ascend yet.";
}


/**
 * @brief Ascend implementation of SpMM on COO format.
 * 
 * @note Not implemented yet. COO format SpMM operations will fall back to error.
 * @todo Implement COO SpMM with CPU fallback or native Ascend kernels
 */
template <typename IdType, typename DType>
void SpMMCooAscend(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  LOG(FATAL) << "SpMMCoo on Ascend is not implemented yet. "
             << "Op: " << op << ", Reduce: " << reduce;
}

// Template specializations for COO SpMM (not implemented)
template <>
void SpMMCoo<kDGLAscend, int32_t, uint16_t>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  SpMMCooAscend<int32_t, uint16_t>(op, reduce, bcast, coo, ufeat, efeat, out, out_aux);
}

template <>
void SpMMCoo<kDGLAscend, int64_t, uint16_t>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  SpMMCooAscend<int64_t, uint16_t>(op, reduce, bcast, coo, ufeat, efeat, out, out_aux);
}

template <>
void SpMMCoo<kDGLAscend, int32_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  SpMMCooAscend<int32_t, float>(op, reduce, bcast, coo, ufeat, efeat, out, out_aux);
}

template <>
void SpMMCoo<kDGLAscend, int64_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  SpMMCooAscend<int64_t, float>(op, reduce, bcast, coo, ufeat, efeat, out, out_aux);
}

template <>
void SpMMCoo<kDGLAscend, int32_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
    LOG(FATAL) << "Double precision not fully supported on Ascend yet.";
}

template <>
void SpMMCoo<kDGLAscend, int64_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
    LOG(FATAL) << "Double precision not fully supported on Ascend yet.";
}

} // namespace aten
} // namespace dgl

