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

#include <stack>
#include <tuple>
#include <dgl/graph.h>

namespace dgl {
namespace traverse {

/*!
 * \brief Traverse the graph in a breadth-first-search (BFS) order.
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
      for (auto v : (graph.*neighbor_iter)(u)) {
        if (!visited[v]) {
          visit(v);
          visited[v] = true;
        }
      }
    }
    frontier = make_frontier();
  }
}

/*!
 * \brief Traverse the graph in topological order.
 *
 * \param reversed If true, follows the in-edge direction
 * \param visit The function to call when a node is visited; the node id will be
 *              given as its only argument.
 * \param make_frontier The function to make a new froniter; the function should return a
 *                      node iterator to the just created frontier.
 */
template<typename VisitFn, typename FrontierFn>
void TopologicalNodes(const Graph& graph,
                      bool reversed,
                      VisitFn visit,
                      FrontierFn make_frontier) {
  const auto get_degree = reversed? &Graph::OutDegree : &Graph::InDegree;
  const auto neighbor_iter = reversed? &Graph::PredVec : &Graph::SuccVec;
  uint64_t num_visited_nodes = 0;
  std::vector<uint64_t> degrees(graph.NumVertices(), 0);
  for (dgl_id_t vid = 0; vid < graph.NumVertices(); ++vid) {
    degrees[vid] = (graph.*get_degree)(vid);
    if (degrees[vid] == 0) {
      visit(vid);
      ++num_visited_nodes;
    }
  }
  auto frontier = make_frontier();

  while (frontier.size() != 0) {
    for (const dgl_id_t u : frontier) {
      for (auto v : (graph.*neighbor_iter)(u)) {
        if (--(degrees[v]) == 0) {
          visit(v);
          ++num_visited_nodes;
        }
      }
    }
    // new node frointer
    frontier = make_frontier();
  }
  if (num_visited_nodes != graph.NumVertices()) {
    LOG(FATAL) << "Error in topological traversal: loop detected in the given graph.";
  }
}

/*!\brief Tags for ``DFSEdges``. */
enum DFSEdgeTag {
  kForward = 0,
  kReverse,
  kNonTree,
};
/*!
 * \brief Traverse the graph in a depth-first-search (DFS) order.
 *
 * The traversal visit edges in its DFS order. Edges have three tags:
 * FORWARD(0), REVERSE(1), NONTREE(2)
 *
 * A FORWARD edge is one in which `u` has been visisted but `v` has not.
 * A REVERSE edge is one in which both `u` and `v` have been visisted and the edge
 * is in the DFS tree.
 * A NONTREE edge is one in which both `u` and `v` have been visisted but the edge
 * is NOT in the DFS tree.
 *
 * \param source Source node.
 * \param reversed If true, DFS follows the in-edge direction
 * \param has_reverse_edge If true, REVERSE edges are included
 * \param has_nontree_edge If true, NONTREE edges are included
 * \param visit The function to call when an edge is visited; the edge id and its
 *              tag will be given as the arguments.
 */
template<typename VisitFn>
void DFSLabeledEdges(const Graph& graph,
                     dgl_id_t source,
                     bool reversed,
                     bool has_reverse_edge,
                     bool has_nontree_edge,
                     VisitFn visit) {
  const auto succ = reversed? &Graph::PredVec : &Graph::SuccVec;
  const auto out_edge = reversed? &Graph::InEdgeVec : &Graph::OutEdgeVec;

  if ((graph.*succ)(source).size() == 0) {
    // no out-going edges from the source node
    return;
  }

  typedef std::tuple<dgl_id_t, size_t, bool> StackEntry;
  std::stack<StackEntry> stack;
  std::vector<bool> visited(graph.NumVertices());
  visited[source] = true;
  stack.push(std::make_tuple(source, 0, false));
  dgl_id_t u = 0;
  size_t i = 0;
  bool on_tree = false;

  while (!stack.empty()) {
    std::tie(u, i, on_tree) = stack.top();
    LOG(INFO) << "u=" << u << " i=" << i << " on_tree=" << on_tree;
    const dgl_id_t v = (graph.*succ)(u)[i];
    const dgl_id_t uv = (graph.*out_edge)(u)[i];
    if (visited[v]) {
      if (!on_tree && has_nontree_edge) {
        visit(uv, kNonTree);
      } else if (on_tree && has_reverse_edge) {
        visit(uv, kReverse);
      }
      stack.pop();
      // find next one.
      if (i < (graph.*succ)(u).size() - 1) {
        stack.push(std::make_tuple(u, i+1, false));
      }
    } else {
      visited[v] = true;
      std::get<2>(stack.top()) = true;
      visit(uv, kForward);
      // expand
      if ((graph.*succ)(v).size() > 0) {
        stack.push(std::make_tuple(v, 0, false));
      }
    }
  }
}

}  // namespace traverse
}  // namespace dgl

#endif  // DGL_GRAPH_TRAVERSAL_H_
