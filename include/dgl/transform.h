/**
 *  Copyright (c) 2019 by Contributors
 * @file dgl/transform.h
 * @brief DGL graph transformations
 */

#ifndef DGL_TRANSFORM_H_
#define DGL_TRANSFORM_H_

#include <tuple>
#include <utility>
#include <vector>

#include "array.h"
#include "base_heterograph.h"

namespace dgl {

namespace transform {

/**
 * @brief Given a list of graphs, remove the common nodes that do not have
 * inbound and outbound edges.
 *
 * The graphs should have identical node ID space (i.e. should have the same set
 * of nodes, including types and IDs).
 *
 * @param graphs The list of graphs.
 * @param always_preserve The list of nodes to preserve regardless of whether
 * the inbound or outbound edges exist.
 *
 * @return A pair.  The first element is the list of compacted graphs, and the
 * second element is the mapping from the compacted graphs and the original
 * graph.
 */
std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>> CompactGraphs(
    const std::vector<HeteroGraphPtr> &graphs,
    const std::vector<IdArray> &always_preserve);

/**
 * @brief Convert a graph into a bipartite-structured graph for message passing.
 *
 * Specifically, we create one node type \c ntype_l on the "left" side and
 * another node type \c ntype_r on the "right" side for each node type \c ntype.
 * The nodes of type \c ntype_r would contain the nodes designated by the
 * caller, and node type \c ntype_l would contain the nodes that has an edge
 * connecting to one of the designated nodes.
 *
 * The nodes of \c ntype_l would also contain the nodes in node type \c ntype_r.
 *
 * This function is often used for constructing a series of dependency graphs
 * for multi-layer message passing, where we first construct a series of
 * frontier graphs on the original node space, and run the following to get the
 * bipartite graph needed for message passing with each GNN layer:
 *
 * <code>
 *     bipartites = [None] * len(num_layers)
 *     for l in reversed(range(len(layers))):
 *         bipartites[l], seeds = to_bipartite(frontier[l], seeds)
 *     x = graph.ndata["h"][seeds]
 *     for g, layer in zip(bipartites, layers):
 *         x_src = x
 *         x_dst = x[:len(g.dsttype)]
 *         x = sageconv(g, (x_src, x_dst))
 *     output = x
 * </code>
 *
 * @param graph The graph.
 * @param rhs_nodes Designated nodes that would appear on the right side.
 * @param include_rhs_in_lhs If false, do not include the nodes of node type \c
 * ntype_r in \c ntype_l.
 *
 * @return A triplet containing
 *         * The bipartite-structured graph,
 *         * The induced node from the left side for each graph,
 *         * The induced edges.
 *
 * @note If include_rhs_in_lhs is true, then for each node type \c ntype, the
 * nodes in rhs_nodes[ntype] would always appear first in the nodes of type \c
 * ntype_l in the new graph.
 */
std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>> ToBlock(
    HeteroGraphPtr graph, const std::vector<IdArray> &rhs_nodes,
    bool include_rhs_in_lhs);

/**
 * @brief Convert a multigraph to a simple graph.
 *
 * @return A triplet of
 * * @c hg : The said simple graph.
 * * @c count : The array of edge occurrences per edge type.
 * * @c edge_map : The mapping from original edge IDs to new edge IDs per edge
 * type.
 *
 * @note Example: consider a graph with the following edges
 *
 *     [(0, 1), (1, 3), (2, 2), (1, 3), (1, 4), (1, 4)]
 *
 * Then ToSimpleGraph(g) would yield the following elements:
 *
 * * The first element would be the simple graph itself with the following edges
 *
 *       [(0, 1), (1, 3), (1, 4), (2, 2)]
 *
 * * The second element is an array \c count.  \c count[i] stands for the number
 * of edges connecting simple_g.src[i] and simple_g.dst[i] in the original
 * graph.
 *
 *       count[0] = [1, 2, 2, 1]
 *
 * * One can find the mapping between edges from the original graph to the new
 * simple graph.
 *
 *       edge_map[0] = [0, 1, 3, 1, 2, 2]
 */
std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
ToSimpleGraph(const HeteroGraphPtr graph);

/**
 * @brief Remove edges from a graph.
 *
 * @param graph The graph.
 * @param eids The edge IDs to remove per edge type.
 *
 * @return A pair of the graph with edges removed, as well as the edge ID
 * mapping from the original graph to the new graph per edge type.
 */
std::pair<HeteroGraphPtr, std::vector<IdArray>> RemoveEdges(
    const HeteroGraphPtr graph, const std::vector<IdArray> &eids);

};  // namespace transform

};  // namespace dgl

#endif  // DGL_TRANSFORM_H_
