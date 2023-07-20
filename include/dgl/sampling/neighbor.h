/**
 *  Copyright (c) 2020 by Contributors
 * @file dgl/sampling/neighbor.h
 * @brief Neighborhood-based sampling.
 */
#ifndef DGL_SAMPLING_NEIGHBOR_H_
#define DGL_SAMPLING_NEIGHBOR_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>

#include <tuple>
#include <vector>

namespace dgl {
namespace sampling {

/**
 * @brief Sample from the neighbors of the given nodes and return the sampled
 * edges as a graph.
 *
 * When sampling with replacement, the sampled subgraph could have parallel
 * edges.
 *
 * For sampling without replace, if fanout > the number of neighbors, all the
 * neighbors will be sampled.
 *
 * @param hg The input graph.
 * @param nodes Node IDs of each type. The vector length must be equal to the
 * number of node types. Empty array is allowed.
 * @param fanouts Number of sampled neighbors for each edge type. The vector
 * length should be equal to the number of edge types, or one if they all have
 * the same fanout.
 * @param dir Edge direction.
 * @param probability A vector of 1D float arrays, indicating the transition
 * probability of each edge by edge type.  An empty float array assumes uniform
 * transition.
 * @param exclude_edges Edges IDs of each type which will be excluded during
 * sampling. The vector length must be equal to the number of edges types. Empty
 * array is allowed.
 * @param replace If true, sample with replacement.
 * @return Sampled neighborhoods as a graph. The return graph has the same
 * schema as the original one.
 */
HeteroSubgraph SampleNeighbors(
    const HeteroGraphPtr hg, const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& fanouts, EdgeDir dir,
    const std::vector<FloatArray>& probability,
    const std::vector<IdArray>& exclude_edges, bool replace = true);

/**
 * @brief Sample from the neighbors of the given nodes and convert a graph into
 * a bipartite-structured graph for message passing.
 *
 * Specifically, we create one node type \c ntype_l on the "left" side and
 * another node type \c ntype_r on the "right" side for each node type \c ntype.
 * The nodes of type \c ntype_r would contain the nodes designated by the
 * caller, and node type \c ntype_l would contain the nodes that has an edge
 * connecting to one of the designated nodes.
 *
 * The nodes of \c ntype_l would also contain the nodes in node type \c ntype_r.
 * When sampling with replacement, the sampled subgraph could have parallel
 * edges.
 *
 * For sampling without replace, if fanout > the number of neighbors, all the
 * neighbors will be sampled.
 *
 * Non-deterministic algorithm, requires nodes parameter to store unique Node
 * IDs.
 *
 * @tparam IdType Graph's index data type, can be int32_t or int64_t
 * @param hg The input graph.
 * @param nodes Node IDs of each type. The vector length must be equal to the
 * number of node types. Empty array is allowed.
 * @param mapping External parameter that should be set to a vector of IdArrays
 *                filled with -1, required for mapping of nodes in returned
 *                graph
 * @param fanouts Number of sampled neighbors for each edge type. The vector
 * length should be equal to the number of edge types, or one if they all have
 * the same fanout.
 * @param dir Edge direction.
 * @param probability A vector of 1D float arrays, indicating the transition
 * probability of each edge by edge type.  An empty float array assumes uniform
 * transition.
 * @param exclude_edges Edges IDs of each type which will be excluded during
 * sampling. The vector length must be equal to the number of edges types. Empty
 * array is allowed.
 * @param replace If true, sample with replacement.
 * @return Sampled neighborhoods as a graph. The return graph has the same
 * schema as the original one.
 */
template <typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
SampleNeighborsFused(
    const HeteroGraphPtr hg, const std::vector<IdArray>& nodes,
    const std::vector<IdArray>& mapping, const std::vector<int64_t>& fanouts,
    EdgeDir dir, const std::vector<NDArray>& prob_or_mask,
    const std::vector<IdArray>& exclude_edges, bool replace = true);

/**
 * Select the neighbors with k-largest weights on the connecting edges for each
 * given node.
 *
 * If k > the number of neighbors, all the neighbors are sampled.
 *
 * @param hg The input graph.
 * @param nodes Node IDs of each type. The vector length must be equal to the
 * number of node types. Empty array is allowed.
 * @param k The k value for each edge type. The vector length should be equal to
 * the number of edge types, or one if they all have the same fanout.
 * @param dir Edge direction.
 * @param weight A vector of 1D float arrays, indicating the weights associated
 * witheach edge.
 * @param ascending If true, elements are sorted by ascending order, equivalent
 * to find the K smallest values. Otherwise, find K largest values.
 * @return Sampled neighborhoods as a graph. The return graph has the same
 * schema as the original one.
 */
HeteroSubgraph SampleNeighborsTopk(
    const HeteroGraphPtr hg, const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& k, EdgeDir dir,
    const std::vector<FloatArray>& weight, bool ascending = false);

HeteroSubgraph SampleNeighborsBiased(
    const HeteroGraphPtr hg, const IdArray& nodes, const int64_t fanouts,
    const NDArray& bias, const NDArray& tag_offset, const EdgeDir dir,
    const bool replace);
}  // namespace sampling
}  // namespace dgl

#endif  // DGL_SAMPLING_NEIGHBOR_H_
