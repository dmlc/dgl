/**
 * Copyright (c) 2024 by Contributors
 * @file to_block_ascend.cc
 * @brief Ascend device-side IdsMapper for to_block.
 *
 * Implements the AscendIdsMapper functor pattern (matching CUDAIdsMapper)
 * using Ascend C kernels launched via aclrtlaunch_to_block_kernel.
 *
 * Flow per IdsMapper call (ProcessToBlock internally managed):
 *   Phase A: For each ntype with non-empty src_nodes → dedup_hash_build (block_dim=1)
 *   Phase B: For each ntype with non-empty rhs (!include_rhs_in_lhs) → dedup_hash_build
 *   Phase C: Sync + read unique counts → build lhs_nodes + num_nodes_per_type
 *   Phase D: For each etype with edges → map_edges (multi-core)
 *   Phase E: Return new_lhs/new_rhs (mapped edge arrays on device)
 *
 * Tiling uploads use synchronous host-to-device aclrtMemcpy, so no
 * per-tiling aclrtSynchronizeStream is required; a single sync before
 * reading counts back (Phase C) covers all preceding kernels.
 */

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>

#define ASCEND_CALL(func)                                                 \
  {                                                                      \
    aclError e = (func);                                                 \
    CHECK(e == ACL_SUCCESS)                                              \
        << "Ascend Error (" << __FILE__ << ":" << __LINE__ << "): code=" \
        << e;                                                            \
  }

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_to_block_kernel(
    uint32_t blockDim, aclrtStream stream, void* input1, void* input2,
    void* hash_arg1, void* hash_arg2, void* hash_arg3, void* hash_arg4,
    void* output1, void* output2, void* tiling);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}  // namespace runtime
}  // namespace dgl

#endif  // DGL_USE_ASCEND

#include "to_block_ascend.h"
#include "../to_block.h"
#include "../../../array/ascend/to_block_tiling.h"

#include <dgl/array.h>
#include <dgl/aten/array_ops.h>
#include <dgl/base_heterograph.h>
#include <dgl/runtime/device_api.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

namespace dgl {
namespace transform {

namespace {

using namespace dgl::runtime;
using namespace dgl::aten;

// Returns the device's vector-core count, queried at runtime.
// Falls back to the arch default when the query is unavailable.
uint32_t QueryVectorCoreCount(int device_id) {
  int64_t core_num = 0;
  aclError err =
      aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &core_num);
  if (err != ACL_SUCCESS || core_num <= 0 || core_num > 4096) {
    return kToBlockDefaultVectorCoreCount;
  }
  return static_cast<uint32_t>(core_num);
}

// Returns the per-vector-core unified-buffer budget (minus reserved tail).
// Note: ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE is not available in all CANN
// versions. The fallback default (192KB - 2KB = 190KB) is correct for 910B3.
uint32_t QueryUbAvailableBytes(int device_id) {
  (void)device_id;
#ifdef ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE
  int64_t ub_bytes = 0;
  aclError err = aclrtGetDeviceInfo(
      device_id, ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE, &ub_bytes);
  if (err != ACL_SUCCESS ||
      ub_bytes <= static_cast<int64_t>(kToBlockUbReservedBytes) ||
      ub_bytes > (1 << 30)) {
    return kToBlockDefaultUbBytes - kToBlockUbReservedBytes;
  }
  return static_cast<uint32_t>(ub_bytes - kToBlockUbReservedBytes);
#else
  // CANN version does not expose the UB query attribute; use the 910B default.
  return kToBlockDefaultUbBytes - kToBlockUbReservedBytes;
#endif
}

// Next power of two >= x (for x >= 1)
inline int64_t NextPow2(int64_t x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  return x + 1;
}

// Returns the max single-band hash capacity that fits in UB for the given
// dtype. K3 allocates 4 arrays (src_keys/src_vals/dst_keys/dst_vals), so
// 4 * capacity * dtype_bytes must fit in the UB budget:
//   int32: 4 * 8192 * 4 = 128KB <= 190KB ✓
//   int64: 4 * 8192 * 8 = 256KB > 190KB ✗ → cap at 4096 (128KB)
inline int64_t MaxHashCapacityForDtype(uint32_t dtype_bytes) {
  if (dtype_bytes == 8) {
    return kToBlockMaxHashCapacity / 2;  // 4096
  }
  return kToBlockMaxHashCapacity;  // 8192
}

// Computes hash capacity: power-of-two >= 3*num_ids, capped at the UB limit.
// This matches CPU's ~4-6x capacity for good load factor.
//
// R2: If the requested capacity exceeds the single-band UB budget, the
// host-side fails fast with a clear error instead of launching a kernel
// whose hash table would saturate and hang. Banding (multi-band flush) is
// a future enhancement (DESIGN.md §3.2/3.3); until then oversized graphs
// are rejected explicitly.
inline int64_t ComputeHashCapacity(int64_t num_ids, uint32_t dtype_bytes) {
  if (num_ids == 0) return 16;
  int64_t raw = NextPow2(std::max(static_cast<int64_t>(16), num_ids * 3));
  const int64_t max_capacity = MaxHashCapacityForDtype(dtype_bytes);
  if (raw > max_capacity) {
    LOG(FATAL) << "to_block: candidate node count " << num_ids
               << " exceeds the single-band hash table capacity "
               << max_capacity << " for dtype_bytes=" << dtype_bytes
               << ". Banding is not implemented yet; this graph is too large "
                  "for the on-device hash table (DESIGN.md §3.2/3.3).";
  }
  return raw;
}

// The number of unique entries a hash table of the given capacity can hold
// before it is considered saturated (load factor ~0.67).
inline int64_t MaxUniqueForCapacity(int64_t capacity) {
  return static_cast<int64_t>(
      static_cast<double>(capacity) * kToBlockLoadFactorThreshold);
}

/**
 * @brief Allocates device workspace and fills tiling for dedup_hash_build.
 */
void SetupDedupTiling(
    ToBlockTiling* host_tiling, int64_t num_ids, int64_t num_seeds,
    int64_t hash_capacity, uint32_t dtype_bytes, uint32_t ub_avail) {
  memset(host_tiling, 0, sizeof(ToBlockTiling));
  host_tiling->op_type = 0;  // dedup_hash_build
  host_tiling->dtype_bytes = dtype_bytes;
  host_tiling->ub_available = ub_avail;
  host_tiling->num_ids = num_ids;
  host_tiling->num_seeds = num_seeds;
  host_tiling->hash_capacity = hash_capacity;
  host_tiling->hash_mask = hash_capacity - 1;
}

constexpr uint32_t kMinEdgesPerBlock = 64;

// ---------------------------------------------------------------------------
// AscendIdsMapper: device-side to_block ID mapper via Ascend C kernels
// ---------------------------------------------------------------------------
template <typename IdType>
struct AscendIdsMapper {
  std::tuple<std::vector<IdArray>, std::vector<IdArray>> operator()(
      const HeteroGraphPtr& graph, bool include_rhs_in_lhs, int64_t num_ntypes,
      const DGLContext& ctx, const std::vector<int64_t>& maxNodesPerType,
      const std::vector<EdgeArray>& edge_arrays,
      const std::vector<IdArray>& src_nodes,
      const std::vector<IdArray>& rhs_nodes,
      std::vector<IdArray>* const lhs_nodes_ptr,
      std::vector<int64_t>* const num_nodes_per_type_ptr) {
#ifdef DGL_USE_ASCEND
    std::vector<IdArray>& lhs_nodes = *lhs_nodes_ptr;
    std::vector<int64_t>& num_nodes_per_type = *num_nodes_per_type_ptr;
    const bool generate_lhs_nodes = lhs_nodes.empty();
    auto device = runtime::DeviceAPI::Get(ctx);
    auto stream = dgl::runtime::getCurrentAscendStream();
    const uint32_t dtype_bytes = static_cast<uint32_t>(sizeof(IdType));
    const uint32_t ub_avail = QueryUbAvailableBytes(ctx.device_id);
    const int64_t num_etypes = static_cast<int64_t>(edge_arrays.size());

    // -----------------------------------------------------------------------
    // Phase A: Build lhs hash tables + unique_ids
    // For each ntype with non-empty src_nodes: launch K1 (dedup_hash_build).
    // -----------------------------------------------------------------------
    // Device-side arrays that need to persist until Phase D:
    struct NtypeWorkspace {
      void* hash_keys = nullptr;
      void* hash_vals = nullptr;
      void* unique_ids = nullptr;
      void* count = nullptr;
      int64_t capacity = 0;
      int64_t num_ids = 0;
    };
    std::vector<NtypeWorkspace> lhs_ws(num_ntypes);

    if (generate_lhs_nodes) {
      lhs_nodes.reserve(num_ntypes);
      for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
        lhs_nodes.emplace_back(
            NewIdArray(maxNodesPerType[ntype], ctx, sizeof(IdType) * 8));
      }
    }

    // Also allocate rhs workspace if needed (Phase B)
    std::vector<NtypeWorkspace> rhs_ws(num_ntypes);

    // Track tiling device buffers so they can all be freed in Phase E
    std::vector<void*> tiling_devs;

    for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
      if (aten::IsNullArray(src_nodes[ntype])) continue;

      const int64_t num_ids = src_nodes[ntype]->shape[0];
      const int64_t num_seeds =
          include_rhs_in_lhs && !aten::IsNullArray(rhs_nodes[ntype])
              ? rhs_nodes[ntype]->shape[0]
              : 0;
      const int64_t capacity = ComputeHashCapacity(num_ids, dtype_bytes);

      // Allocate workspace
      NtypeWorkspace& ws = lhs_ws[ntype];
      ws.num_ids = num_ids;
      ws.capacity = capacity;
      ws.hash_keys = device->AllocWorkspace(ctx, capacity * dtype_bytes);
      ws.hash_vals = device->AllocWorkspace(ctx, capacity * dtype_bytes);
      ws.unique_ids = device->AllocWorkspace(ctx, num_ids * dtype_bytes);
      ws.count = device->AllocWorkspace(ctx, sizeof(int64_t));

      // Setup tiling
      ToBlockTiling host_tiling;
      SetupDedupTiling(
          &host_tiling, num_ids, num_seeds, capacity, dtype_bytes, ub_avail);

      void* tiling_dev = device->AllocWorkspace(ctx, sizeof(ToBlockTiling));
      tiling_devs.push_back(tiling_dev);
      // CopyDataFromTo (CPU->Ascend) uses synchronous aclrtMemcpy, so the
      // tiling is visible to the kernel without an extra stream sync (F1).
      device->CopyDataFromTo(
          &host_tiling, 0, tiling_dev, 0, sizeof(ToBlockTiling), 
          DGLContext{kDGLCPU, 0}, ctx, DGLDataType{});

      // Launch K1 (dedup_hash_build) — single core
      aclError err = aclrtlaunch_to_block_kernel(
          1, stream,
          src_nodes[ntype]->data,               // input1
          nullptr,                               // input2 (unused)
          ws.hash_keys,                          // hash_arg1 = keys_out
          ws.hash_vals,                          // hash_arg2 = vals_out
          ws.unique_ids,                         // hash_arg3 = unique_out
          ws.count,                              // hash_arg4 = count_out
          nullptr,                               // output1 (unused)
          nullptr,                               // output2 (unused)
          tiling_dev);
      CHECK(err == ACL_SUCCESS)
          << "to_block dedup_hash_build launch failed: " << err;
    }

    // -----------------------------------------------------------------------
    // Phase B: Build rhs hash tables (if !include_rhs_in_lhs)
    // -----------------------------------------------------------------------
    if (!include_rhs_in_lhs) {
      for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
        if (aten::IsNullArray(rhs_nodes[ntype])) continue;

        const int64_t num_ids = rhs_nodes[ntype]->shape[0];
        if (num_ids == 0) continue;
        // rhs nodes are destination nodes, treated as seeds (all unique)
        const int64_t num_seeds = num_ids;
        const int64_t capacity = ComputeHashCapacity(num_ids, dtype_bytes);

        NtypeWorkspace& ws = rhs_ws[ntype];
        ws.num_ids = num_ids;
        ws.capacity = capacity;
        ws.hash_keys = device->AllocWorkspace(ctx, capacity * dtype_bytes);
        ws.hash_vals = device->AllocWorkspace(ctx, capacity * dtype_bytes);
        ws.unique_ids = device->AllocWorkspace(ctx, num_ids * dtype_bytes);
        ws.count = device->AllocWorkspace(ctx, sizeof(int64_t));

        ToBlockTiling host_tiling;
        SetupDedupTiling(
            &host_tiling, num_ids, num_seeds, capacity, dtype_bytes, ub_avail);

        void* tiling_dev = device->AllocWorkspace(ctx, sizeof(ToBlockTiling));
        tiling_devs.push_back(tiling_dev);
        device->CopyDataFromTo(
            &host_tiling, 0, tiling_dev, 0, sizeof(ToBlockTiling),
            DGLContext{kDGLCPU, 0}, ctx, DGLDataType{});

        aclError err = aclrtlaunch_to_block_kernel(
            1, stream,
            rhs_nodes[ntype]->data,              // input1 = rhs_ids
            nullptr,                              // input2 (unused)
            ws.hash_keys,                         // hash_arg1 = keys_out
            ws.hash_vals,                         // hash_arg2 = vals_out
            ws.unique_ids,                        // hash_arg3 = unique_out
            ws.count,                             // hash_arg4 = count_out
            nullptr, nullptr,                     // output (unused)
            tiling_dev);
        CHECK(err == ACL_SUCCESS)
            << "to_block rhs hash build launch failed: " << err;
      }
    }

    // -----------------------------------------------------------------------
    // Phase C: Sync + read counts → build lhs_nodes + num_nodes_per_type
    // -----------------------------------------------------------------------
    if (generate_lhs_nodes) {
      // Read back counts for all ntypes
      for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
        if (lhs_ws[ntype].count == nullptr) {
          // No src_nodes for this type; lhs side stays 0
          num_nodes_per_type[ntype] = 0;
          lhs_nodes[ntype]->shape[0] = 0;
          continue;
        }
        int64_t unique_count = 0;
        ASCEND_CALL(aclrtSynchronizeStream(stream));
        ASCEND_CALL(aclrtMemcpy(
            &unique_count, sizeof(int64_t), lhs_ws[ntype].count,
            sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST));
        if (unique_count < 0) {
          LOG(FATAL) << "to_block: hash table saturation detected on device "
                        "for ntype "
                     << ntype << ". Graph too large for single-band on-device "
                        "hash table (banding not implemented).";
        }
        num_nodes_per_type[ntype] = unique_count;
        lhs_nodes[ntype]->shape[0] = unique_count;
        // Copy the unique ids from workspace into the lhs_nodes array.
        if (unique_count > 0) {
          device->CopyDataFromTo(
              lhs_ws[ntype].unique_ids, 0, lhs_nodes[ntype]->data, 0,
              sizeof(IdType) * unique_count, ctx, ctx,
              DGLDataTypeTraits<IdType>::dtype);
        }
      }
    } else {
      // lhs_nodes given by caller, use their shapes
      for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
        num_nodes_per_type[ntype] = lhs_nodes[ntype]->shape[0];
      }
    }

    // -----------------------------------------------------------------------
    // Phase D: Map edges via K3 (map_edges) for each etype
    // -----------------------------------------------------------------------
    std::vector<IdArray> new_lhs_ids;
    std::vector<IdArray> new_rhs_ids;
    new_lhs_ids.reserve(num_etypes);
    new_rhs_ids.reserve(num_etypes);

    for (int64_t etype = 0; etype < num_etypes; ++etype) {
      const EdgeArray& edges = edge_arrays[etype];
      if (edges.id.defined() && !aten::IsNullArray(edges.src)) {
        const auto src_dst_types = graph->GetEndpointTypes(etype);
        const int src_type = src_dst_types.first;
        const int dst_type = src_dst_types.second;

        const int64_t edge_cnt = edges.src->shape[0];
        IdArray new_src = NewIdArray(edge_cnt, ctx, sizeof(IdType) * 8);
        IdArray new_dst = NewIdArray(edge_cnt, ctx, sizeof(IdType) * 8);

        // Determine which hash tables to use:
        //   src → lhs hash of src_type (always)
        //   dst → if include_rhs_in_lhs: lhs hash of dst_type; else: rhs hash
        NtypeWorkspace& src_ws = lhs_ws[src_type];
        NtypeWorkspace& dst_ws =
            include_rhs_in_lhs ? lhs_ws[dst_type] : rhs_ws[dst_type];

        CHECK(src_ws.count != nullptr)
            << "src type " << src_type << " hash table not built";
        CHECK(dst_ws.count != nullptr)
            << "dst type " << dst_type << " hash table not built";

        constexpr uint32_t kMinEdgesPerBlock = 64;
        const uint32_t block_dim = std::min(
            QueryVectorCoreCount(ctx.device_id),
            static_cast<uint32_t>(
                (edge_cnt + static_cast<int64_t>(kMinEdgesPerBlock) - 1) /
                static_cast<int64_t>(kMinEdgesPerBlock)));
        const uint32_t launch_dim = std::max(block_dim, 1u);

        ToBlockTiling host_tiling;
        memset(&host_tiling, 0, sizeof(ToBlockTiling));
        host_tiling.op_type = 1;  // map_edges
        host_tiling.dtype_bytes = dtype_bytes;
        host_tiling.ub_available = ub_avail;
        host_tiling.num_edges = edge_cnt;
        host_tiling.hash_capacity_src = src_ws.capacity;
        host_tiling.hash_mask_src = src_ws.capacity - 1;
        host_tiling.hash_capacity_dst = dst_ws.capacity;
        host_tiling.hash_mask_dst = dst_ws.capacity - 1;
        host_tiling.block_dim = launch_dim;

        void* tiling_dev = device->AllocWorkspace(ctx, sizeof(ToBlockTiling));
        tiling_devs.push_back(tiling_dev);
        device->CopyDataFromTo(
            &host_tiling, 0, tiling_dev, 0, sizeof(ToBlockTiling),
            DGLContext{kDGLCPU, 0}, ctx, DGLDataType{});

        aclError err = aclrtlaunch_to_block_kernel(
            launch_dim, stream,
            edges.src->data,                     // input1 = edge_src
            edges.dst->data,                     // input2 = edge_dst
            src_ws.hash_keys,                    // hash_arg1 = src_keys
            src_ws.hash_vals,                    // hash_arg2 = src_vals
            dst_ws.hash_keys,                    // hash_arg3 = dst_keys
            dst_ws.hash_vals,                    // hash_arg4 = dst_vals
            new_src->data,                       // output1 = new_src
            new_dst->data,                       // output2 = new_dst
            tiling_dev);
        CHECK(err == ACL_SUCCESS)
            << "to_block map_edges launch failed: " << err;

        new_lhs_ids.emplace_back(new_src);
        new_rhs_ids.emplace_back(new_dst);
      } else {
        // Empty etype → null arrays
        new_lhs_ids.emplace_back(
            aten::NullArray(DGLDataTypeTraits<IdType>::dtype, ctx));
        new_rhs_ids.emplace_back(
            aten::NullArray(DGLDataTypeTraits<IdType>::dtype, ctx));
      }
    }

    // -----------------------------------------------------------------------
    // Phase E: Sync and free workspaces
    // -----------------------------------------------------------------------
    ASCEND_CALL(aclrtSynchronizeStream(stream));

    for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
      if (lhs_ws[ntype].hash_keys)
        device->FreeWorkspace(ctx, lhs_ws[ntype].hash_keys);
      if (lhs_ws[ntype].hash_vals)
        device->FreeWorkspace(ctx, lhs_ws[ntype].hash_vals);
      if (lhs_ws[ntype].unique_ids)
        device->FreeWorkspace(ctx, lhs_ws[ntype].unique_ids);
      if (lhs_ws[ntype].count)
        device->FreeWorkspace(ctx, lhs_ws[ntype].count);

      if (rhs_ws[ntype].hash_keys)
        device->FreeWorkspace(ctx, rhs_ws[ntype].hash_keys);
      if (rhs_ws[ntype].hash_vals)
        device->FreeWorkspace(ctx, rhs_ws[ntype].hash_vals);
      if (rhs_ws[ntype].unique_ids)
        device->FreeWorkspace(ctx, rhs_ws[ntype].unique_ids);
      if (rhs_ws[ntype].count)
        device->FreeWorkspace(ctx, rhs_ws[ntype].count);
    }

    // Free all tiling device buffers
    for (void* td : tiling_devs) {
      if (td) device->FreeWorkspace(ctx, td);
    }

    return std::make_tuple(std::move(new_lhs_ids), std::move(new_rhs_ids));
#else
    LOG(FATAL) << "Ascend support is not compiled. "
                  "Please compile with -DUSE_ASCEND=ON";
    return {};
#endif  // DGL_USE_ASCEND
  }
};

}  // namespace

template <typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>> ToBlockAscend(
    HeteroGraphPtr graph, const std::vector<IdArray>& rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray>* lhs_nodes) {
#ifdef DGL_USE_ASCEND
  auto ctx = graph->Context();
  CHECK(ctx.device_type == kDGLAscend)
      << "Expected Ascend device context for ToBlockAscend";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  return dgl::transform::ProcessToBlock<IdType>(
      graph, rhs_nodes, include_rhs_in_lhs, lhs_nodes,
      AscendIdsMapper<IdType>());
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}

// Explicit instantiations
template std::tuple<HeteroGraphPtr, std::vector<IdArray>>
ToBlockAscend<int32_t>(
    HeteroGraphPtr, const std::vector<IdArray>&, bool,
    std::vector<IdArray>*);

template std::tuple<HeteroGraphPtr, std::vector<IdArray>>
ToBlockAscend<int64_t>(
    HeteroGraphPtr, const std::vector<IdArray>&, bool,
    std::vector<IdArray>*);

}  // namespace transform
}  // namespace dgl