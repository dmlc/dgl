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
  /*!
   * \brief optional tensor data for each node.  The first dimension always has
   * the number of elements the same as number of nodes.
   */
  dgl::runtime::NDArray node_data;
  /*!
   * \brief optional tensor data for each edge.  The first dimension always has
   * the number of elements the same as number of edges.
   */
  dgl::runtime::NDArray edge_data;
};

/*!
 * \brief Get a slice on a graph that represents a NodeFlow.
 *
 * The entire block has to be taken as a slice. Users have to specify the
 * correct starting and ending location of a layer.
 *
 * If remap is false, the returned arrays can be viewed as a sub-matrix slice
 * of the adjmat of the input graph. Let the adjmat of the input graph be A,
 * then the slice is equal to (in numpy syntax):
 *   A[layer1_start:layer1_end, layer0_start:layer0_end]
 *
 * If remap is true,  the returned arrays represents an adjacency matrix
 * of shape NxM, where N is the number of nodes in layer1 and M is
 * the number of nodes in layer0. Nodes in layer0 will be remapped to
 * [0, M) and nodes in layer1 will be remapped to [0, N).
 *
 * A row of the returned adjacency matrix represents the destination
 * of an edge and the column represents the source.
 *
 * If fmt == "csr", the function returns three arrays: indptr, indices, eid.
 * If fmt == "coo", the function returns two arrays: idx, eid. Here, the idx array
 *   is the concatenation of src and dst node id arrays.
 *
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

