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