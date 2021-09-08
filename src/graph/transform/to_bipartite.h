/*!
 *  Copyright (c) 2021 by Contributors
 * \file graph/transform/to_bipartite.h
 * \brief Array operator templates
 */

#ifndef DGL_GRAPH_TRANSFORM_TO_BIPARTITE_H_
#define DGL_GRAPH_TRANSFORM_TO_BIPARTITE_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>

#include <tuple>
#include <vector>

namespace dgl {
namespace transform {

/**
 * @brief Create a graph block from the set of
 * src and dst nodes (lhs and rhs respectively).
 *
 * @tparam XPU The type of device to operate on.
 * @tparam IdType The type to use as an index.
 * @param graph The graph from which to extract the block.
 * @param rhs_nodes The destination nodes of the block.
 * @param include_rhs_in_lhs Whether or not to include the
 * destination nodes of the block in the sources nodes.
 * @param [in/out] lhs_nodes The source nodes of the block.
 *
 * @return The block and the induced edges.
 */
template<DLDeviceType XPU, typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>>
ToBlock(HeteroGraphPtr graph, const std::vector<IdArray> &rhs_nodes,
        bool include_rhs_in_lhs, std::vector<IdArray>* lhs_nodes);

}  // namespace transform
}  // namespace dgl

#endif  // DGL_GRAPH_TRANSFORM_TO_BIPARTITE_H_
