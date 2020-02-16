/*!
 *  Copyright (c) 2019 by Contributors
 * \file dgl/transform.h
 * \brief DGL graph transformations
 */

#ifndef DGL_TRANSFORM_H_
#define DGL_TRANSFORM_H_

#include <vector>
#include <tuple>
#include <utility>
#include "base_heterograph.h"
#include "array.h"

namespace dgl {

namespace transform {

/*!
 * \brief Given a list of graphs, remove the common nodes that do not have inbound and
 * outbound edges.
 *
 * The graphs should have identical node ID space (i.e. should have the same set of nodes,
 * including types and IDs) and metagraph.
 *
 * \param graphs The list of graphs.
 * \param always_preserve The list of nodes to preserve regardless of whether the inbound
 *                        or outbound edges exist.
 *
 * \return A pair.  The first element is the list of compacted graphs, and the second
 * element is the mapping from the compacted graphs and the original graph.
 */
std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>>
CompactGraphs(
    const std::vector<HeteroGraphPtr> &graphs,
    const std::vector<IdArray> &always_preserve);

/*!
 * \brief Convert a multigraph to a simple graph.
 *
 * \return A triplet of
 * * \c hg : The said simple graph.
 * * \c count : The array of edge occurrences per edge type.
 * * \c edge_map : The mapping from original edge IDs to new edge IDs per edge type.
 *
 * \note Example: consider the following graph:
 *
 *     g = dgl.graph([(0, 1), (1, 3), (2, 2), (1, 3), (1, 4), (1, 4)])
 *
 * Then ToSimpleGraph(g) would yield the following elements:
 *
 * * The first element would be the simple graph itself:
 *
 *       simple_g = dgl.graph([(0, 1), (1, 3), (1, 4), (2, 2)])
 *
 * * The second element is an array \c count.  \c count[i] stands for the number of edges
 *   connecting simple_g.src[i] and simple_g.dst[i] in the original graph.
 *
 *       count[0] = [1, 2, 2, 1]
 *
 * * One can find the mapping between edges from the original graph to the new simple
 *   graph.
 *
 *       edge_map[0] = [0, 1, 3, 1, 2, 2]
 */
std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
ToSimpleGraph(const HeteroGraphPtr graph);

/*!
 * \brief Induce an edge subgraph from inbound edges with K-largest weights for each node.
 *
 * For each edge type, the edges are grouped by destination nodes.  Then for each edge
 * group, the K edges with largest weights are selected.  The selected edges then form
 * a subgraph which is returned.
 *
 * All nodes in the original graph are preserved in the subgraph.
 *
 * \param graph The graph
 * \param weights The edge weights for each edge type
 * \param inbound If true (default), group edges by destination.  Otherwise, group by source.
 * \param smallest If true, select the K edges with smallest weights instead.  Default: false.
 * \return A triplet of the subgraph, the IDs of induced edges, and the weights of induced
 * edges.
 *
 * \note Example: consider the following graph:
 *
 *     g = dgl.graph([(0, 1), (2, 1), (5, 1), (2, 2), (3, 2), (4, 2), (6, 2)])
 *
 * with weights
 *
 *     weights = [1, 2, 3, 4, 5, 6, 7]
 *
 * Then the result of SelectTopK(g, weights, 2) is a triplet of:
 *
 * 1. The subgraph:
 *
 *     sg = dgl.graph([(2, 1), (5, 1), (4, 2), (6, 2)])
 *
 * 2. The edge IDs induced from the original graph
 *
 *     induced_edges = [1, 2, 5, 6]
 */
std::pair<HeteroGraphPtr, std::vector<IdArray>> SelectTopK(
    const HeteroGraphPtr graph,
    const std::vector<NDArray> &weights,
    int K,
    bool inbound = true,
    bool smallest = false);

};  // namespace transform

};  // namespace dgl

#endif  // DGL_TRANSFORM_H_
