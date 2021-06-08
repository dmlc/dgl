/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/transform/cuda_to_block.h
 * \brief Functions to convert a set of edges into a graph block with local
 * ids.
 */


#ifndef DGL_GRAPH_TRANSFORM_CUDA_CUDA_TO_BLOCK_H_
#define DGL_GRAPH_TRANSFORM_CUDA_CUDA_TO_BLOCK_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <vector>
#include <tuple>

namespace dgl {
namespace transform {
namespace cuda {

/**
 * @brief Generate a subgraph with locally numbered vertices, from the given
 * edge set.
 *
 * @param graph The set of edges to construct the subgraph from.
 * @param rhs_nodes The unique set of destination vertices.
 * @param include_rhs_in_lhs Whether or not to include the `rhs_nodes` in the
 * set of source vertices for purposes of local numbering.
 *
 * @return The subgraph, the unique set of source nodes, and the mapping of
 * subgraph edges to global edges.
 */
std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
CudaToBlock(
    HeteroGraphPtr graph,
    const std::vector<IdArray>& rhs_nodes,
    const bool include_rhs_in_lhs);

}  // namespace cuda
}  // namespace transform
}  // namespace dgl

#endif  // DGL_GRAPH_TRANSFORM_CUDA_CUDA_TO_BLOCK_H_
