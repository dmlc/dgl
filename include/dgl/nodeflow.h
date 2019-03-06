/*!
 *  Copyright (c) 2019 by Contributors
 * \file dgl/nodeflow.h
 * \brief DGL NodeFlow class.
 */
#ifndef DGL_NODEFLOW_H_
#define DGL_NODEFLOW_H_

#include <vector>
#include <string>

#include "graph_interface.h"

namespace dgl {

class ImmutableGraph;

/*!
 * \brief A NodeFlow graph stores the sampling results for a sampler that samples
 * nodes/edges in layers.
 *
 * We store multiple layers of the sampling results in a single graph, which results
 * in a more compact format. We store extra information,
 * such as the node and edge mapping from the NodeFlow graph to the parent graph.
 */
struct NodeFlow {
  /*! \brief The graph. */
  GraphPtr graph;
  /*!
   * \brief the offsets of each layer.
   */
  IdArray layer_offsets;
  /*!
   * \brief the offsets of each flow.
   */
  IdArray flow_offsets;
  /*!
   * \brief The node mapping from the NodeFlow graph to the parent graph.
   */
  IdArray node_mapping;
  /*!
   * \brief The edge mapping from the NodeFlow graph to the parent graph.
   */
  IdArray edge_mapping;
};

/*!
 * \brief Get a slice on a graph that represents a NodeFlow.
 *
 * A row of the returned adjacency matrix represents the destination
 * of an edge and the column represents the source.
 * \param graph An immutable graph.
 * \param fmt the format of the returned adjacency matrix.
 * \param layer0_size the size of the first layer in the block.
 * \param layer1_start the location where the second layer starts.
 * \param layer1_end the location where the secnd layer ends.
 * \param remap Indicates to remap all vertex ids and edge Ids to local Id
 * space.
 * \return a vector of IdArrays.
 */
std::vector<IdArray> GetNodeFlowSlice(const ImmutableGraph &graph, const std::string &fmt,
                                      size_t layer0_size, size_t layer1_start,
                                      size_t layer1_end, bool remap);

}  // namespace dgl

#endif  // DGL_NODEFLOW_H_

