/*!
 *  Copyright (c) 2020 by Contributors
 * \file dgl/graph_traversal.h
 * \brief common graph traversal operations
 */
#ifndef DGL_GRAPH_TRAVERSAL_H_
#define DGL_GRAPH_TRAVERSAL_H_

#include "array.h"
#include "graph_interface.h"

namespace dgl {

///////////////////////// Graph Traverse routines //////////////////////////
/*!
 * \brief Class for representing frontiers.
 *
 * Each frontier is a list of nodes/edges (specified by their ids).
 * An optional tag can be specified on each node/edge (represented by an int value).
 */
struct Frontiers {
  /*!\brief a vector store for the nodes/edges in all the frontiers */
  IdArray ids;

  /*!
   * \brief a vector store for node/edge tags. Dtype is int64.
   * Empty if no tags are requested
   */
  IdArray tags;

  /*!\brief a section vector to indicate each frontier Dtype is int64. */
  IdArray sections;
};

namespace aten {

Frontiers BFSNodesFrontiers(const GraphInterface& graph, IdArray source, const bool reversed);
Frontiers BFSEdgesFrontiers(const GraphInterface& graph, IdArray source, const bool reversed);
Frontiers TopologicalNodesFrontiers(const GraphInterface& graph, const bool reversed);
Frontiers DGLDFSEdges(const GraphInterface& graph, IdArray source, const bool reversed);
Frontiers DGLDFSLabeledEdges(const GraphInterface& graph,
                             IdArray source,
                             const bool reversed,
                             const bool has_reverse_edge,
                             const bool has_nontree_edge,
                             const bool return_labels);

}  // namespace aten
}  // namespace dgl

#endif  // DGL_GRAPH_TRAVERSAL_H_
