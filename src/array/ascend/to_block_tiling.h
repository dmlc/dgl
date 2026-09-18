#ifndef TO_BLOCK_TILING_H
#define TO_BLOCK_TILING_H

#include <cstdint>

struct ToBlockTiling {
  // Common fields
  uint32_t op_type;        // 0=dedup_hash_build, 1=map_edges
  uint32_t dtype_bytes;    // 4 (int32) or 8 (int64)
  uint32_t ub_available;   // per-core UB budget (runtime query)

  // dedup_hash_build (op_type == 0)
  int64_t num_ids;          // number of source IDs to process
  int64_t num_seeds;        // first N IDs are seed (rhs) nodes
  int64_t hash_capacity;    // table capacity in entries (power of 2)
  int64_t hash_mask;        // capacity - 1 (for quadratic probing & mask)

  // map_edges (op_type == 1)
  int64_t num_edges;        // number of edges to map  
  int64_t hash_capacity_src; // src hash table capacity (entries)
  int64_t hash_mask_src;    // src hash mask
  int64_t hash_capacity_dst; // dst hash table capacity (entries)
  int64_t hash_mask_dst;    // dst hash mask
  uint32_t block_dim;       // total blocks for this launch
  uint32_t chunk_elems;     // per-chunk element count for streaming
};

// Hardware parameters (runtime-queried, fallbacks only)
constexpr uint32_t kToBlockDefaultVectorCoreCount = 40;
constexpr uint32_t kToBlockDefaultUbBytes = 192 * 1024;
constexpr uint32_t kToBlockUbReservedBytes = 2 * 1024;

// Sentinel for empty hash slot (node IDs are non-negative)
constexpr int64_t kToBlockEmptyKey = -1;

// Max hash capacity per band (UB-limited).
// K3 holds 4 arrays (src_keys/src_vals/dst_keys/dst_vals) in UB.
// int32: 4 * 8192 * 4 = 128KB <= 190KB ✓
// int64: capped by host at 4096 (4 * 4096 * 8 = 128KB <= 190KB) ✓
constexpr int64_t kToBlockMaxHashCapacity = 8192;

// Load factor threshold for hash table saturation:
// when unique entries exceed capacity * this factor, the table is
// considered saturated and may not find empty slots for new keys.
constexpr double kToBlockLoadFactorThreshold = 0.67;

// Max probe iterations before declaring probe-loop failure (in case
// the hash table is fully saturated despite host-side checks).
constexpr uint32_t kToBlockMaxProbes = 16384;

#endif  // TO_BLOCK_TILING_H