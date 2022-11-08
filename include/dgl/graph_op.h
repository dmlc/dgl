/**
 *  Copyright (c) 2018 by Contributors
 * @file dgl/graph_op.h
 * @brief Operations on graph index.
 */
#ifndef DGL_GRAPH_OP_H_
#define DGL_GRAPH_OP_H_

#include <vector>

#include "graph.h"
#include "immutable_graph.h"

namespace dgl {

class GraphOp {
 public:
  /**
   * @brief Return a new graph with all the edges reversed.
   *
   * The returned graph preserves the vertex and edge index in the original
   * graph.
   *
   * @return the reversed graph
   */
  static GraphPtr Reverse(GraphPtr graph);

  /**
   * @brief Return the line graph.
   *
   * If i~j and j~i are two edges in original graph G, then
   * (i,j)~(j,i) and (j,i)~(i,j) are the "backtracking" edges on
   * the line graph.
   *
   * @param graph The input graph.
   * @param backtracking Whether the backtracking edges are included or not
   * @return the line graph
   */
  static GraphPtr LineGraph(GraphPtr graph, bool backtracking);

  /**
   * @brief Return a disjoint union of the input graphs.
   *
   * The new graph will include all the nodes/edges in the given graphs.
   * Nodes/Edges will be relabled by adding the cumsum of the previous graph
   * sizes in the given sequence order. For example, giving input [g1, g2, g3],
   * where they have 5, 6, 7 nodes respectively. Then node#2 of g2 will become
   * node#7 in the result graph. Edge ids are re-assigned similarly.
   *
   * The input list must be either ALL mutable graphs or ALL immutable graphs.
   * The returned graph type is also determined by the input graph type.
   *
   * @param graphs A list of input graphs to be unioned.
   * @return the disjoint union of the graphs
   */
  static GraphPtr DisjointUnion(std::vector<GraphPtr> graphs);

  /**
   * @brief Partition the graph into several subgraphs.
   *
   * This is a reverse operation of DisjointUnion. The graph will be partitioned
   * into num graphs. This requires the given number of partitions to evenly
   * divides the number of nodes in the graph.
   *
   * If the input graph is mutable, the result graphs are mutable.
   * If the input graph is immutable, the result graphs are immutable.
   *
   * @param graph The graph to be partitioned.
   * @param num The number of partitions.
   * @return a list of partitioned graphs
   */
  static std::vector<GraphPtr> DisjointPartitionByNum(
      GraphPtr graph, int64_t num);

  /**
   * @brief Partition the graph into several subgraphs.
   *
   * This is a reverse operation of DisjointUnion. The graph will be partitioned
   * based on the given sizes. This requires the sum of the given sizes is equal
   * to the number of nodes in the graph.
   *
   * If the input graph is mutable, the result graphs are mutable.
   * If the input graph is immutable, the result graphs are immutable.
   *
   * @param graph The graph to be partitioned.
   * @param sizes The number of partitions.
   * @return a list of partitioned graphs
   */
  static std::vector<GraphPtr> DisjointPartitionBySizes(
      GraphPtr graph, IdArray sizes);

  /**
   * @brief Map vids in the parent graph to the vids in the subgraph.
   *
   * If the Id doesn't exist in the subgraph, -1 will be used.
   *
   * @param parent_vid_map An array that maps the vids in the parent graph to
   * the subgraph. The elements store the vertex Ids in the parent graph, and
   * the indices indicate the vertex Ids in the subgraph.
   * @param query The vertex Ids in the parent graph.
   * @return an Id array that contains the subgraph node Ids.
   */
  static IdArray MapParentIdToSubgraphId(IdArray parent_vid_map, IdArray query);

  /**
   * @brief Expand an Id array based on the offset array.
   *
   * For example,
   * ids:     [0, 1, 2, 3, 4],
   * offset:  [0, 2, 2, 5, 6, 7],
   * result:  [0, 0, 2, 2, 2, 3, 4].
   * The offset array has one more element than the ids array.
   * (offset[i], offset[i+1]) shows the location of ids[i] in the result array.
   *
   * @param ids An array that contains the node or edge Ids.
   * @param offset An array that contains the offset after expansion.
   * @return a expanded Id array.
   */
  static IdArray ExpandIds(IdArray ids, IdArray offset);

  /**
   * @brief Convert the graph to a simple graph.
   * @param graph The input graph.
   * @return a new immutable simple graph with no multi-edge.
   */
  static GraphPtr ToSimpleGraph(GraphPtr graph);

  /**
   * @brief Convert the graph to a mutable bidirected graph.
   *
   * If the original graph has m edges for i -> j and n edges for
   * j -> i, the new graph will have max(m, n) edges for both
   * i -> j and j -> i.
   *
   * @param graph The input graph.
   * @return a new mutable bidirected graph.
   */
  static GraphPtr ToBidirectedMutableGraph(GraphPtr graph);

  /**
   * @brief Same as BidirectedMutableGraph except that the returned graph is
   *        immutable.
   * @param graph The input graph.
   * @return a new immutable bidirected
   * graph.
   */
  static GraphPtr ToBidirectedImmutableGraph(GraphPtr graph);
  /**
   * @brief Same as BidirectedMutableGraph except that the returned graph is
   * immutable and call gk_csr_MakeSymmetric in GKlib. This is more efficient
   * than ToBidirectedImmutableGraph. It return a null pointer if the conversion
   * fails.
   *
   * @param graph The input graph.
   * @return a new immutable bidirected graph.
   */
  static GraphPtr ToBidirectedSimpleImmutableGraph(ImmutableGraphPtr ig);

  /**
   * @brief Get a induced subgraph with HALO nodes.
   * The HALO nodes are the ones that can be reached from `nodes` within
   * `num_hops`.
   * @param graph The input graph.
   * @param nodes The input nodes that form the core of the induced subgraph.
   * @param num_hops The number of hops to reach.
   * @return the induced subgraph with HALO nodes.
   */
  static HaloSubgraph GetSubgraphWithHalo(
      GraphPtr graph, IdArray nodes, int num_hops);

  /**
   * @brief Reorder the nodes in the immutable graph.
   * @param graph The input graph.
   * @param new_order The node Ids in the new graph. The index in `new_order` is
   *        old node Ids.
   * @return the graph with reordered node Ids
   */
  static GraphPtr ReorderImmutableGraph(
      ImmutableGraphPtr ig, IdArray new_order);
};

}  // namespace dgl

#endif  // DGL_GRAPH_OP_H_
