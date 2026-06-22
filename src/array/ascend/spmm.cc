#include <dgl/array.h>
#include <dgl/aten/csr.h>
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

#ifdef DGL_USE_ASCEND
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

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_spmm_sum(
    uint32_t blockDim, aclrtStream stream, void* denseBlockData,
    void* featureData, void* outputData, void* indptrData, void* indicesData,
    void* vectorWindowIdsData, void* vectorWinSplitData, void* cubeWindowIdsData,
    void* cubeWinSplitData, void* winEdgePtrData, void* colToEdgeData,
    uint32_t numDstRows, uint32_t numSrcRows, uint32_t featureDim,
    uint32_t nonZeroCount, uint32_t totalTcBlocks, uint32_t vectorWindowCount,
    uint32_t cubeWindowCount, uint32_t columnToEdgeLength);

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

  // Validate operation
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
  
  static aclrtStream spmm_stream = nullptr;
  if (spmm_stream == nullptr) {
    ASCEND_CALL(aclrtCreateStream(&spmm_stream));
  }
  aclrtStream stream = spmm_stream;

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

// Template specializations for CSR SpMM
template <>
void SpMMCsr<kDGLAscend, int32_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  LOG(FATAL) << "Float precision not fully supported on Ascend yet.";
}

template <>
void SpMMCsr<kDGLAscend, int32_t, uint16_t>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  SpMMCsrAscend<int32_t, uint16_t>(op, reduce, bcast, csr, ufeat, efeat, out, out_aux);
}

template <>
void SpMMCsr<kDGLAscend, int64_t, uint16_t>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  LOG(FATAL) << "Int64 precision not fully supported on Ascend yet.";
}

template <>
void SpMMCsr<kDGLAscend, int64_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  LOG(FATAL) << "Int64 precision not fully supported on Ascend yet.";
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
