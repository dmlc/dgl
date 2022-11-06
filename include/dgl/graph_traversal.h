/**
 *  Copyright (c) 2020 by Contributors
 * @file dgl/graph_traversal.h
 * @brief common graph traversal operations
 */
#ifndef DGL_GRAPH_TRAVERSAL_H_
#define DGL_GRAPH_TRAVERSAL_H_

#include "array.h"
#include "base_heterograph.h"

namespace dgl {

///////////////////////// Graph Traverse routines //////////////////////////
/**
 * @brief Class for representing frontiers.
 *
 * Each frontier is a list of nodes/edges (specified by their ids).
 * An optional tag can be specified on each node/edge (represented by an int
 * value).
 */
struct Frontiers {
  /** @brief a vector store for the nodes/edges in all the frontiers */
  IdArray ids;

  /**
   * @brief a vector store for node/edge tags. Dtype is int64.
   * Empty if no tags are requested
   */
  IdArray tags;

  /** @brief a section vector to indicate each frontier Dtype is int64. */
  IdArray sections;
};

namespace aten {

/**
 * @brief Traverse the graph in a breadth-first-search (BFS) order.
 *
 * @param csr The input csr matrix.
 * @param sources Source nodes.
 * @return A Frontiers object containing the search result
 */
Frontiers BFSNodesFrontiers(const CSRMatrix& csr, IdArray source);

/**
 * @brief Traverse the graph in a breadth-first-search (BFS) order, returning
 *        the edges of the BFS tree.
 *
 * @param csr The input csr matrix.
 * @param sources Source nodes.
 * @return A Frontiers object containing the search result
 */
Frontiers BFSEdgesFrontiers(const CSRMatrix& csr, IdArray source);

/**
 * @brief Traverse the graph in topological order.
 *
 * @param csr The input csr matrix.
 * @return A Frontiers object containing the search result
 */
Frontiers TopologicalNodesFrontiers(const CSRMatrix& csr);

/**
 * @brief Traverse the graph in a depth-first-search (DFS) order.
 *
 * @param csr The input csr matrix.
 * @param sources Source nodes.
 * @return A Frontiers object containing the search result
 */
Frontiers DGLDFSEdges(const CSRMatrix& csr, IdArray source);

/**
 * @brief Traverse the graph in a depth-first-search (DFS) order and return the
 *        recorded edge tag if return_labels is specified.
 *
 * The traversal visit edges in its DFS order. Edges have three tags:
 * FORWARD(0), REVERSE(1), NONTREE(2)
 *
 * A FORWARD edge is one in which `u` has been visisted but `v` has not.
 * A REVERSE edge is one in which both `u` and `v` have been visisted and the
 * edge is in the DFS tree.
 * A NONTREE edge is one in which both `u` and `v` have been visisted but the
 * edge is NOT in the DFS tree.
 *
 * @param csr The input csr matrix.
 * @param sources Source nodes.
 * @param has_reverse_edge If true, REVERSE edges are included
 * @param has_nontree_edge If true, NONTREE edges are included
 * @param return_labels If true, return the recorded edge tags.
 * @return A Frontiers object containing the search result
 */
Frontiers DGLDFSLabeledEdges(
    const CSRMatrix& csr, IdArray source, const bool has_reverse_edge,
    const bool has_nontree_edge, const bool return_labels);

}  // namespace aten
}  // namespace dgl

#endif  // DGL_GRAPH_TRAVERSAL_H_
