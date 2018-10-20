// Graph operations
#ifndef DGL_GRAPH_OP_H_
#define DGL_GRAPH_OP_H_

#include "graph.h"

namespace dgl {

class GraphOp {
 public:
  /*!
   * \brief Return the line graph.
   *
   * If i~j and j~i are two edges in original graph G, then
   * (i,j)~(j,i) and (j,i)~(i,j) are the "backtracking" edges on
   * the line graph.
   *
   * \param graph The input graph.
   * \param backtracking Whether the backtracking edges are included or not
   * \return the line graph
   */
  static Graph LineGraph(const Graph* graph, bool backtracking);

  /*!
   * \brief Return a disjoint union of the input graphs.
   *
   * The new graph will include all the nodes/edges in the given graphs.
   * Nodes/Edges will be relabled by adding the cumsum of the previous graph sizes
   * in the given sequence order. For example, giving input [g1, g2, g3], where
   * they have 5, 6, 7 nodes respectively. Then node#2 of g2 will become node#7
   * in the result graph. Edge ids are re-assigned similarly.
   *
   * \param graphs A list of input graphs to be unioned.
   * \return the disjoint union of the graphs
   */
  static Graph DisjointUnion(std::vector<const Graph*> graphs);

  /*!
   * \brief Partition the graph into several subgraphs.
   *
   * This is a reverse operation of DisjointUnion. The graph will be partitioned
   * into num graphs. This requires the given number of partitions to evenly
   * divides the number of nodes in the graph.
   * 
   * \param graph The graph to be partitioned.
   * \param num The number of partitions.
   * \return a list of partitioned graphs
   */
  static std::vector<Graph> DisjointPartitionByNum(const Graph* graph, int64_t num);

  /*!
   * \brief Partition the graph into several subgraphs.
   *
   * This is a reverse operation of DisjointUnion. The graph will be partitioned
   * based on the given sizes. This requires the sum of the given sizes is equal
   * to the number of nodes in the graph.
   * 
   * \param graph The graph to be partitioned.
   * \param sizes The number of partitions.
   * \return a list of partitioned graphs
   */
  static std::vector<Graph> DisjointPartitionBySizes(const Graph* graph, IdArray sizes);
};

}  // namespace dgl

#endif  // DGL_GRAPH_OP_H_
