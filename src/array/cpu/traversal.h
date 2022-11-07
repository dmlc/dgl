/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/traversal.h
 * @brief Graph traversal routines.
 *
 * Traversal routines generate frontiers. Frontiers can be node frontiers or
 * edge frontiers depending on the traversal function. Each frontier is a list
 * of nodes/edges (specified by their ids). An optional tag can be specified for
 * each node/edge (represented by an int value).
 */
#ifndef DGL_ARRAY_CPU_TRAVERSAL_H_
#define DGL_ARRAY_CPU_TRAVERSAL_H_

#include <dgl/graph_interface.h>

#include <stack>
#include <tuple>
#include <vector>

namespace dgl {
namespace aten {
namespace impl {

/**
 * @brief Traverse the graph in a breadth-first-search (BFS) order.
 *
 * The queue object must suffice following interface:
 *   Members:
 *   void push(IdType);  // push one node
 *   IdType top();       // get the first node
 *   void pop();           // pop one node
 *   bool empty();         // return true if the queue is empty
 *   size_t size();        // return the size of the queue
 * For example, std::queue<IdType> is a valid queue type.
 *
 * The visit function must be compatible with following interface:
 *   void (*visit)(IdType );
 *
 * The frontier function must be compatible with following interface:
 *   void (*make_frontier)(void);
 *
 * @param graph The graph.
 * @param sources Source nodes.
 * @param reversed If true, BFS follows the in-edge direction
 * @param queue The queue used to do bfs.
 * @param visit The function to call when a node is visited.
 * @param make_frontier The function to indicate that a new froniter can be
 * made;
 */
template <
    typename IdType, typename Queue, typename VisitFn, typename FrontierFn>
void BFSTraverseNodes(
    const CSRMatrix &csr, IdArray source, Queue *queue, VisitFn visit,
    FrontierFn make_frontier) {
  const int64_t len = source->shape[0];
  const IdType *src_data = static_cast<IdType *>(source->data);

  const IdType *indptr_data = static_cast<IdType *>(csr.indptr->data);
  const IdType *indices_data = static_cast<IdType *>(csr.indices->data);
  const int64_t num_nodes = csr.num_rows;
  std::vector<bool> visited(num_nodes);
  for (int64_t i = 0; i < len; ++i) {
    const IdType u = src_data[i];
    visited[u] = true;
    visit(u);
    queue->push(u);
  }
  make_frontier();

  while (!queue->empty()) {
    const size_t size = queue->size();
    for (size_t i = 0; i < size; ++i) {
      const IdType u = queue->top();
      queue->pop();
      for (auto idx = indptr_data[u]; idx < indptr_data[u + 1]; ++idx) {
        auto v = indices_data[idx];
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
 *   void push(IdType);  // push one node
 *   IdType top();       // get the first node
 *   void pop();           // pop one node
 *   bool empty();         // return true if the queue is empty
 *   size_t size();        // return the size of the queue
 * For example, std::queue<IdType> is a valid queue type.
 *
 * The visit function must be compatible with following interface:
 *   void (*visit)(IdType );
 *
 * The frontier function must be compatible with following interface:
 *   void (*make_frontier)(void);
 *
 * @param graph The graph.
 * @param sources Source nodes.
 * @param reversed If true, BFS follows the in-edge direction
 * @param queue The queue used to do bfs.
 * @param visit The function to call when a node is visited.
 *        The argument would be edge ID.
 * @param make_frontier The function to indicate that a new frontier can be
 * made;
 */
template <
    typename IdType, typename Queue, typename VisitFn, typename FrontierFn>
void BFSTraverseEdges(
    const CSRMatrix &csr, IdArray source, Queue *queue, VisitFn visit,
    FrontierFn make_frontier) {
  const int64_t len = source->shape[0];
  const IdType *src_data = static_cast<IdType *>(source->data);

  const IdType *indptr_data = static_cast<IdType *>(csr.indptr->data);
  const IdType *indices_data = static_cast<IdType *>(csr.indices->data);
  const IdType *eid_data = static_cast<IdType *>(csr.data->data);

  const int64_t num_nodes = csr.num_rows;
  std::vector<bool> visited(num_nodes);
  for (int64_t i = 0; i < len; ++i) {
    const IdType u = src_data[i];
    visited[u] = true;
    queue->push(u);
  }
  make_frontier();

  while (!queue->empty()) {
    const size_t size = queue->size();
    for (size_t i = 0; i < size; ++i) {
      const IdType u = queue->top();
      queue->pop();
      for (auto idx = indptr_data[u]; idx < indptr_data[u + 1]; ++idx) {
        auto e = eid_data ? eid_data[idx] : idx;
        const IdType v = indices_data[idx];
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
 *   void push(IdType);  // push one node
 *   IdType top();       // get the first node
 *   void pop();           // pop one node
 *   bool empty();         // return true if the queue is empty
 *   size_t size();        // return the size of the queue
 * For example, std::queue<IdType> is a valid queue type.
 *
 * The visit function must be compatible with following interface:
 *   void (*visit)(IdType );
 *
 * The frontier function must be compatible with following interface:
 *   void (*make_frontier)(void);
 *
 * @param graph The graph.
 * @param reversed If true, follows the in-edge direction
 * @param queue The queue used to do bfs.
 * @param visit The function to call when a node is visited.
 * @param make_frontier The function to indicate that a new froniter can be
 * made;
 */
template <
    typename IdType, typename Queue, typename VisitFn, typename FrontierFn>
void TopologicalNodes(
    const CSRMatrix &csr, Queue *queue, VisitFn visit,
    FrontierFn make_frontier) {
  int64_t num_visited_nodes = 0;
  const IdType *indptr_data = static_cast<IdType *>(csr.indptr->data);
  const IdType *indices_data = static_cast<IdType *>(csr.indices->data);

  const int64_t num_nodes = csr.num_rows;
  const int64_t num_edges = csr.indices->shape[0];
  std::vector<int64_t> degrees(num_nodes, 0);
  for (int64_t eid = 0; eid < num_edges; ++eid) {
    degrees[indices_data[eid]]++;
  }

  for (int64_t vid = 0; vid < num_nodes; ++vid) {
    if (degrees[vid] == 0) {
      visit(vid);
      queue->push(static_cast<IdType>(vid));
      ++num_visited_nodes;
    }
  }
  make_frontier();

  while (!queue->empty()) {
    const size_t size = queue->size();
    for (size_t i = 0; i < size; ++i) {
      const IdType u = queue->top();
      queue->pop();
      for (auto idx = indptr_data[u]; idx < indptr_data[u + 1]; ++idx) {
        const IdType v = indices_data[idx];
        if (--(degrees[v]) == 0) {
          visit(v);
          queue->push(v);
          ++num_visited_nodes;
        }
      }
    }
    make_frontier();
  }

  if (num_visited_nodes != num_nodes) {
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
 * FORWARD(0), REVERSE(1), NONTREE(2)
 *
 * A FORWARD edge is one in which `u` has been visisted but `v` has not.
 * A REVERSE edge is one in which both `u` and `v` have been visisted and the
 * edge is in the DFS tree. A NONTREE edge is one in which both `u` and `v` have
 * been visisted but the edge is NOT in the DFS tree.
 *
 * @param source Source node.
 * @param reversed If true, DFS follows the in-edge direction
 * @param has_reverse_edge If true, REVERSE edges are included
 * @param has_nontree_edge If true, NONTREE edges are included
 * @param visit The function to call when an edge is visited; the edge id and
 * its tag will be given as the arguments.
 */
template <typename IdType, typename VisitFn>
void DFSLabeledEdges(
    const CSRMatrix &csr, IdType source, bool has_reverse_edge,
    bool has_nontree_edge, VisitFn visit) {
  const int64_t num_nodes = csr.num_rows;
  CHECK_GE(num_nodes, source)
      << "source " << source << " is out of range [0," << num_nodes << "]";
  const IdType *indptr_data = static_cast<IdType *>(csr.indptr->data);
  const IdType *indices_data = static_cast<IdType *>(csr.indices->data);
  const IdType *eid_data = static_cast<IdType *>(csr.data->data);

  if (indptr_data[source + 1] - indptr_data[source] == 0) {
    // no out-going edges from the source node
    return;
  }

  typedef std::tuple<IdType, size_t, bool> StackEntry;
  std::stack<StackEntry> stack;
  std::vector<bool> visited(num_nodes);
  visited[source] = true;
  stack.push(std::make_tuple(source, 0, false));
  IdType u = 0;
  int64_t i = 0;
  bool on_tree = false;

  while (!stack.empty()) {
    std::tie(u, i, on_tree) = stack.top();
    const IdType v = indices_data[indptr_data[u] + i];
    const IdType uv =
        eid_data ? eid_data[indptr_data[u] + i] : indptr_data[u] + i;
    if (visited[v]) {
      if (!on_tree && has_nontree_edge) {
        visit(uv, kNonTree);
      } else if (on_tree && has_reverse_edge) {
        visit(uv, kReverse);
      }
      stack.pop();
      // find next one.
      if (indptr_data[u] + i < indptr_data[u + 1] - 1) {
        stack.push(std::make_tuple(u, i + 1, false));
      }
    } else {
      visited[v] = true;
      std::get<2>(stack.top()) = true;
      visit(uv, kForward);
      // expand
      if (indptr_data[v] < indptr_data[v + 1]) {
        stack.push(std::make_tuple(v, 0, false));
      }
    }
  }
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_TRAVERSAL_H_
