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
 * \brief argument structure used in ConstructNodeFlow
 */
struct neighbor_info {
  /*! \brief which node this set of edges is pointing to */
  dgl_id_t id;
  /*! \brief the offset of this edge set in neighbor_list */
  size_t pos;
  /*! \brief the number of edges in this edge set */
  size_t num_edges;

  /*! \brief default ctor */
  neighbor_info(dgl_id_t id, size_t pos, size_t num_edges) {
    this->id = id;
    this->pos = pos;
    this->num_edges = num_edges;
  }
};

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
   * \brief whether the edge mapping to parent graph is available.
   */
  bool edge_mapping_available;
  /*!
   * \brief The node mapping from the NodeFlow graph to the parent graph.
   */
  IdArray node_mapping;
  /*!
   * \brief The edge mapping from the NodeFlow graph to the parent graph.
   */
  IdArray edge_mapping;
  /*!
   * \brief The column name in node frame.  Empty string if no node data is returned.
   */
  std::string node_data_name;
  /*!
   * \brief The column name in edge frame.  Empty string if no edge data is returned.
   */
  std::string edge_data_name;
  /*!
   * \brief Application-specific tensor data on nodes
   */
  runtime::NDArray node_data;
  /*!
   * \brief Application specific tensor data on edges
   */
  runtime::NDArray edge_data;
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

/*!
 * \brief NodeFlow constructor
 * \param sub_vers The flattened list of sampled nodes on the original graph
 * \param layer_offsets The offsets to \a sub_vers for each NodeFlow layer, plus the last element
 * containing the length of \a sub_vers
 * \param edge_list The flattened list of edge IDs on the original graph (-1 if not exist)
 * \param neighbor_list The flattened list of source nodes of \a edge_list
 * \param neigh_pos List of \a neighbor_info keeping track of target nodes of \a edge_list
 * \param edge_type If "out", switches direction of edges
 * \param is_multigraph Whether the nodeflow graph could be a multigraph
 * \param nf The output NodeFlow object
 * \param vertex_mapping Output mapping from vertex IDs of NodeFlow graph to \a sub_vers
 * \param edge_mapping Output mapping from edge IDs of NodeFlow graph to \a edge_list
 */
void ConstructNodeFlow(
    const std::vector<dgl_id_t> &neighbor_list,
    const std::vector<dgl_id_t> &edge_list,
    const std::vector<size_t> &layer_offsets,
    std::vector<dgl_id_t> *sub_vers,
    std::vector<neighbor_info> *neigh_pos,
    const std::string &edge_type,
    bool is_multigraph,
    NodeFlow *nf,
    std::vector<dgl_id_t> *vertex_mapping,
    std::vector<dgl_id_t> *edge_mapping);

}  // namespace dgl

#endif  // DGL_NODEFLOW_H_

