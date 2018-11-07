/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/traversal.h
 * \brief Graph traversal routines.
 *
 * Traversal routines generate frontiers. Frontiers can be node frontiers or edge
 * frontiers depending on the traversal function. Each frontier is a
 * list of nodes/edges (specified by their ids). An optional tag can be specified
 * for each node/edge (represented by an int value).
 */
#ifndef DGL_GRAPH_TRAVERSAL_H_
#define DGL_GRAPH_TRAVERSAL_H_

#include <dgl/graph.h>

namespace dgl {
namespace traverse {

/*!
 * \brief Traverse the graph and produce node frontiers in a breadth-first-search (BFS) order.
 *
 * \param sources Source nodes.
 * \param reversed If true, BFS follows the in-edge direction
 * \param visit The function to call when a node is visited; the node id will be
 *              given as its only argument.
 * \param make_frontier The function to make a new froniter; the function should return a
 *                      node iterator to the just created frontier.
 */
template<typename VisitFn, typename FrontierFn>
void BFSNodes(const Graph& graph,
              IdArray source,
              bool reversed,
              VisitFn visit,
              FrontierFn make_frontier) {
  const int64_t len = source->shape[0];
  const int64_t* src_data = static_cast<int64_t*>(source->data);

  std::vector<bool> visited(graph.NumVertices());
  for (int64_t i = 0; i < len; ++i) {
    visited[src_data[i]] = true;
    visit(src_data[i]);
  }
  auto frontier = make_frontier();

  const auto neighbor_iter = reversed? &Graph::PredVec : &Graph::SuccVec;
  while (frontier.size() != 0) {
    for (const dgl_id_t u : frontier) {
      for (auto v : (graph.*neighbor_iter)(u)) {//ret.ids[k])) {
        if (!visited[v]) {
          visit(v);
          //ret.ids.push_back(v);
          visited[v] = true;
        }
      }
      // new node frointer
      //ret.sections.push_back(j - i);
      //i = j;
      //j = ret.ids.size();
    }
    frontier = make_frontier();
  }
}

/*!
 * \brief Produce node frontiers in a topological sort order.
 *
 * \param source Source nodes.
 * \param reversed If true, follows the in-edge direction
 * \return node frontiers
 */
//Frontiers TopologicalNodes(const Graph& graph, bool reversed);

/*!\brief Tags for ``DFSEdges``. */
enum DFSEdgeTag {
  kForward = 0,
  kReverse,
  kNonTree,
};
/*!
 * \brief Produce edge frontiers in a depth-first-search (DFS) order tagged by type.
 *
 * There are three tags: FORWARD(0), REVERSE(1), NONTREE(2)
 *
 * A FORWARD edge is one in which `u` has been visisted but `v` has not.
 * A REVERSE edge is one in which both `u` and `v` have been visisted and the edge
 * is in the DFS tree.
 * A NONTREE edge is one in which both `u` and `v` have been visisted but the edge
 * is NOT in the DFS tree.
 *
 * Multiple source nodes can be specified to start the DFS traversal. Each starting
 * node will result in its own DFS tree, so the resulting frontiers are simply
 * the merge of the frontiers of each DFS tree.
 *
 * \param source Source nodes.
 * \param reversed If true, DFS follows the in-edge direction
 * \param has_reverse_edge If true, REVERSE edges are included
 * \param has_nontree_edge If true, NONTREE edges are included
 * \return the edge frontiers
 */
//Frontiers DFSEdges(const Graph& graph,
                   //IdArray sources,
                   //bool reversed,
                   //bool has_reverse_edge,
                   //bool has_nontree_edge);

}  // namespace traverse
}  // namespace dgl

#endif  // DGL_GRAPH_TRAVERSAL_H_
