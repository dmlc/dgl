/**
 *  Copyright (c) 2018 by Contributors
 * @file graph/traversal.h
 * @brief Graph traversal routines.
 *
 * Traversal routines generate frontiers. Frontiers can be node frontiers or
 * edge frontiers depending on the traversal function. Each frontier is a list
 * of nodes/edges (specified by their ids). An optional tag can be specified for
 * each node/edge (represented by an int value).
 */
#ifndef DGL_GRAPH_TRAVERSAL_H_
#define DGL_GRAPH_TRAVERSAL_H_

#include <dgl/graph_interface.h>

#include <stack>
#include <tuple>
#include <vector>

namespace dgl {
namespace traverse {

/**
 * @brief Traverse the graph in a breadth-first-search (BFS) order.
 *
 * The queue object must suffice following interface:
 *   Members:
 *   void push(dgl_id_t);  // push one node
 *   dgl_id_t top();       // get the first node
 *   void pop();           // pop one node
 *   bool empty();         // return true if the queue is empty
 *   size_t size();        // return the size of the queue
 * For example, std::queue<dgl_id_t> is a valid queue type.
 *
 * The visit function must be compatible with following interface:
 *   void (*visit)(dgl_id_t );
 *
 * The frontier function must be compatible with following interface:
 *   void (*make_frontier)(void);
 *
 * @param graph The graph.
 * @param sources Source nodes.
 * @param reversed If true, BFS follows the in-edge direction.
 * @param queue The queue used to do bfs.
 * @param visit The function to call when a node is visited.
 * @param make_frontier The function to indicate that a new froniter can be
 *        made.
 */
template <typename Queue, typename VisitFn, typename FrontierFn>
void BFSNodes(
    const GraphInterface& graph, IdArray source, bool reversed, Queue* queue,
    VisitFn visit, FrontierFn make_frontier) {
  const int64_t len = source->shape[0];
  const int64_t* src_data = static_cast<int64_t*>(source->data);

  std::vector<bool> visited(graph.NumVertices());
  for (int64_t i = 0; i < len; ++i) {
    const dgl_id_t u = src_data[i];
    visited[u] = true;
    visit(u);
    queue->push(u);
  }
  make_frontier();

  const auto neighbor_iter =
      reversed ? &GraphInterface::PredVec : &GraphInterface::SuccVec;
  while (!queue->empty()) {
    const size_t size = queue->size();
    for (size_t i = 0; i < size; ++i) {
      const dgl_id_t u = queue->top();
      queue->pop();
      for (auto v : (graph.*neighbor_iter)(u)) {
        if (!visited[v]) {
          visited[v] = true;
          visit(v);
          queue->push(v);
        }
      }
    }
    make_frontier();
  }
}

/**
 * @brief Traverse the graph in a breadth-first-search (BFS) order, returning
 *        the edges of the BFS tree.
 *
 * The queue object must suffice following interface:
 *   Members:
 *   void push(dgl_id_t);  // push one node
 *   dgl_id_t top();       // get the first node
 *   void pop();           // pop one node
 *   bool empty();         // return true if the queue is empty
 *   size_t size();        // return the size of the queue
 * For example, std::queue<dgl_id_t> is a valid queue type.
 *
 * The visit function must be compatible with following interface:
 *   void (*visit)(dgl_id_t );
 *
 * The frontier function must be compatible with following interface:
 *   void (*make_frontier)(void);
 *
 * @param graph The graph.
 * @param sources Source nodes.
 * @param reversed If true, BFS follows the in-edge direction.
 * @param queue The queue used to do bfs.
 * @param visit The function to call when a node is visited.
 *        The argument would be edge ID.
 * @param make_frontier The function to indicate that a new frontier can be
 *        made.
 */
template <typename Queue, typename VisitFn, typename FrontierFn>
void BFSEdges(
    const GraphInterface& graph, IdArray source, bool reversed, Queue* queue,
    VisitFn visit, FrontierFn make_frontier) {
  const int64_t len = source->shape[0];
  const int64_t* src_data = static_cast<int64_t*>(source->data);

  std::vector<bool> visited(graph.NumVertices());
  for (int64_t i = 0; i < len; ++i) {
    const dgl_id_t u = src_data[i];
    visited[u] = true;
    queue->push(u);
  }
  make_frontier();

  const auto neighbor_iter =
      reversed ? &GraphInterface::InEdgeVec : &GraphInterface::OutEdgeVec;
  while (!queue->empty()) {
    const size_t size = queue->size();
    for (size_t i = 0; i < size; ++i) {
      const dgl_id_t u = queue->top();
      queue->pop();
      for (auto e : (graph.*neighbor_iter)(u)) {
        const auto uv = graph.FindEdge(e);
        const dgl_id_t v = (reversed ? uv.first : uv.second);
        if (!visited[v]) {
          visited[v] = true;
          visit(e);
          queue->push(v);
        }
      }
    }
    make_frontier();
  }
}

/**
 * @brief Traverse the graph in topological order.
 *
 * The queue object must suffice following interface:
 *   Members:
 *   void push(dgl_id_t);  // push one node
 *   dgl_id_t top();       // get the first node
 *   void pop();           // pop one node
 *   bool empty();         // return true if the queue is empty
 *   size_t size();        // return the size of the queue
 * For example, std::queue<dgl_id_t> is a valid queue type.
 *
 * The visit function must be compatible with following interface:
 *   void (*visit)(dgl_id_t );
 *
 * The frontier function must be compatible with following interface:
 *   void (*make_frontier)(void);
 *
 * @param graph The graph.
 * @param reversed If true, follows the in-edge direction.
 * @param queue The queue used to do bfs.
 * @param visit The function to call when a node is visited.
 * @param make_frontier The function to indicate that a new froniter can be
 *        made.
 */
template <typename Queue, typename VisitFn, typename FrontierFn>
void TopologicalNodes(
    const GraphInterface& graph, bool reversed, Queue* queue, VisitFn visit,
    FrontierFn make_frontier) {
  const auto get_degree =
      reversed ? &GraphInterface::OutDegree : &GraphInterface::InDegree;
  const auto neighbor_iter =
      reversed ? &GraphInterface::PredVec : &GraphInterface::SuccVec;
  uint64_t num_visited_nodes = 0;
  std::vector<uint64_t> degrees(graph.NumVertices(), 0);
  for (dgl_id_t vid = 0; vid < graph.NumVertices(); ++vid) {
    degrees[vid] = (graph.*get_degree)(vid);
    if (degrees[vid] == 0) {
      visit(vid);
      queue->push(vid);
      ++num_visited_nodes;
    }
  }
  make_frontier();

  while (!queue->empty()) {
    const size_t size = queue->size();
    for (size_t i = 0; i < size; ++i) {
      const dgl_id_t u = queue->top();
      queue->pop();
      for (auto v : (graph.*neighbor_iter)(u)) {
        if (--(degrees[v]) == 0) {
          visit(v);
          queue->push(v);
          ++num_visited_nodes;
        }
      }
    }
    make_frontier();
  }

  if (num_visited_nodes != graph.NumVertices()) {
    LOG(FATAL)
        << "Error in topological traversal: loop detected in the given graph.";
  }
}

/** @brief Tags for ``DFSEdges``. */
enum DFSEdgeTag {
  kForward = 0,
  kReverse,
  kNonTree,
};
/**
 * @brief Traverse the graph in a depth-first-search (DFS) order.
 *
 * The traversal visit edges in its DFS order. Edges have three tags:
 * FORWARD(0), REVERSE(1), NONTREE(2).
 *
 * A FORWARD edge is one in which `u` has been visisted but `v` has not.
 * A REVERSE edge is one in which both `u` and `v` have been visisted and the
 * edge is in the DFS tree. A NONTREE edge is one in which both `u` and `v` have
 * been visisted but the edge is NOT in the DFS tree.
 *
 * @param source Source node.
 * @param reversed If true, DFS follows the in-edge direction.
 * @param has_reverse_edge If true, REVERSE edges are included.
 * @param has_nontree_edge If true, NONTREE edges are included.
 * @param visit The function to call when an edge is visited; the edge id and
 *        its tag will be given as the arguments.
 */
template <typename VisitFn>
void DFSLabeledEdges(
    const GraphInterface& graph, dgl_id_t source, bool reversed,
    bool has_reverse_edge, bool has_nontree_edge, VisitFn visit) {
  const auto succ =
      reversed ? &GraphInterface::PredVec : &GraphInterface::SuccVec;
  const auto out_edge =
      reversed ? &GraphInterface::InEdgeVec : &GraphInterface::OutEdgeVec;

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
        stack.push(std::make_tuple(u, i + 1, false));
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
