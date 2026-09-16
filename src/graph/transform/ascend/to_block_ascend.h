/**
 * Copyright (c) 2024 by Contributors
 * @file to_block_ascend.h
 * @brief Ascend device-side ToBlock implementation declarations.
 *
 * ToBlockAscend<IdType> implements the DGL IdsMapper pattern on Ascend NPU,
 * replacing the CPU workaround (ToBlockCPU) that copies data host-side.
 *
 * The implementation uses two Ascend C kernel operations:
 *   1. Dedup hash build (single-core) — order-preserving unique+hash per ntype
 *   2. Edge mapping (multi-core) — hash table lookup for edge endpoints
 */

#ifndef DGL_GRAPH_TRANSFORM_ASCEND_TO_BLOCK_ASCEND_H_
#define DGL_GRAPH_TRANSFORM_ASCEND_TO_BLOCK_ASCEND_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>

#include <tuple>
#include <vector>

namespace dgl {
namespace transform {

/**
 * @brief Ascend device-side ToBlock implementation.
 *
 * Replaces ToBlockCPU for Ascend devices. Performs all hash table
 * construction and edge mapping entirely on-device via Ascend C
 * kernels (dedup_hash_build, map_edges).
 */
template <typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>> ToBlockAscend(
    HeteroGraphPtr graph, const std::vector<IdArray>& rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray>* lhs_nodes);

}  // namespace transform
}  // namespace dgl

#endif  // DGL_GRAPH_TRANSFORM_ASCEND_TO_BLOCK_ASCEND_H_