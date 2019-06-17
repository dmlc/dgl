/*!
 *  Copyright (c) 2019 by Contributors
 * \file dgl/heterograph_interface.h
 * \brief DGL heterogeneous graph index class.
 */

#ifndef DGL_HETEROGRAPH_INTERFACE_H_
#define DGL_HETEROGRAPH_INTERFACE_H_

#include <string>
#include <vector>
#include <utility>
#include <algorithm>

#include "array.h"
#include "graph_interface.h"

namespace dgl {

struct HeteroSubgraph;

class HeteroGraphInterface;
typedef std::shared_ptr<HeteroGraphInterface> HeteroGraphPtr;

class HeteroGraphInterface : public GraphInterface {
 public:
  virtual ~HeteroGraphInterface() = default;

  /*!
   * \brief Add vertices to the graph with the given node type.
   *
   * \param ntype the node type
   * \param num_vertices the number of vertices to add
   */
  virtual void AddVertices(dgl_type_t vtype, uint64_t num_vertices);

  /*!
   * \brief Add one edge to the graph with given edge type.
   * \param etype the edge type
   * \param src the type-specific ID of source node
   * \param dst the type-specific ID of destination node
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual void AddEdge(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) = 0;

  /*!
   * \brief Add edges with the same edge type to the graph.
   * \param etype the edge type
   * \param src_ids the type-specific IDs of source nodes
   * \param dst_ids the type-specific IDs of destination nodes
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual void AddEdges(dgl_type_t etype, IdArray src_ids, IdArray dst_ids);

  /*! \return the number of vertices of given type in the graph */
  virtual uint64_t NumVertices(dgl_type_t vtype) const = 0;

  /*! \return the number of edges of given type in the graph */
  virtual uint64_t NumEdges(dgl_type_t etype) const = 0;

  /*!
   * \return true if the vertex exists
   */
  virtual bool HasVertex(dgl_type_t vtype, dgl_id_t vid) const {
    return vid < NumVertices(vtype);
  };

  /*!
   * \return a boolean array of the existence of the nodes
   */
  virtual BoolArray HasVertices(dgl_type_t vtype, IdArray vids) const {
    // FIXME: duplicate code with GraphInterface::HasVertices due to
    // additional argument vtype
    const auto len = vids->shape[0];
    BoolArray rst = NewBoolArray(len);
    const dgl_id_t *vid_data = static_cast<dgl_id_t *>(vids->data);
    dgl_id_t *rst_data = static_cast<dgl_id_t *>(rst->data);
    const uint64_t nverts = NumVertices(vtype);
    for (int64_t i = 0; i < len; ++i) {
      rst_data[i] = (vid_data[i] < nverts) ? 1 : 0;
    }
    return rst;
  }

  /*!
   * \return true if an edge of type exist between type-specific src and dst
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual bool HasEdgeBetween(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const = 0;

  /*!
   * \return a boolean array of existence of the edges between type-specific src and dst
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual BoolArray HasEdgesBetween(dgl_type_t etype, IdArray src, IdArray dst) const {
    // FIXME: duplicate code with GraphInterface::HasVertices due to
    // additional argument etype
    const auto srclen = src_ids->shape[0];
    const auto dstlen = dst_ids->shape[0];
    const auto rstlen = std::max(srclen, dstlen);
    BoolArray rst = BoolArray::Empty({rstlen}, src_ids->dtype, src_ids->ctx);
    dgl_id_t* rst_data = static_cast<dgl_id_t*>(rst->data);
    const dgl_id_t* src_data = static_cast<dgl_id_t*>(src_ids->data);
    const dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst_ids->data);
    if (srclen == 1) {
      // one-many
      for (int64_t i = 0; i < dstlen; ++i) {
        rst_data[i] = HasEdgeBetween(etype, src_data[0], dst_data[i])? 1 : 0;
      }
    } else if (dstlen == 1) {
      // many-one
      for (int64_t i = 0; i < srclen; ++i) {
        rst_data[i] = HasEdgeBetween(etype, src_data[i], dst_data[0])? 1 : 0;
      }
    } else {
      // many-many
      CHECK(srclen == dstlen) << "Invalid src and dst id array.";
      for (int64_t i = 0; i < srclen; ++i) {
        rst_data[i] = HasEdgeBetween(etype, src_data[i], dst_data[i])? 1 : 0;
      }
    }
    return rst;
  }

  /*!
   * \brief Find the predecessors of a vertex given the edge type.
   * \param etype the edge type.
   * \param vid the vertex ID of edge destination type.
   * \return the predecessor ID array of edge source type.
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual IdArray Predecessors(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Find the successors of a vertex given the edge type.
   * \param etype the edge type.
   * \param vid the vertex ID of edge source type.
   * \return the successor ID array of edge destination type.
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual IdArray Successors(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Get all edge IDs of given edge type between two given endpoints.
   * \param etype the edge type.
   * \param src the node ID array of edge source type.
   * \param dst the node ID array of edge destination type.
   * \return the edge ID array
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual IdArray EdgeId(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const = 0;

  /*!
   * \brief Get all edge ids of given edge type between the given endpoint pairs.
   * \note Edges are associated with an integer id start from zero.
   *       The id is assigned when the edge is being added to the graph.
   *       If duplicate pairs exist, the returned edge IDs will also duplicate.
   *       The order of returned edge IDs will follow the order of src-dst pairs
   *       first, and ties are broken by the order of edge ID.
   * \return EdgeArray containing all edges between all pairs.
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual EdgeArray EdgeIds(dgl_type_t etype, IdArray src, IdArray dst) const = 0;

  /*!
   * \brief Find the edge ID of a given edge type, and return the pair of endpoints
   * \param etype The edge type
   * \param eid The edge ID
   * \return a pair whose first element is the source and the second the destination.
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_type_t etype, dgl_id_t eid) const = 0;

  /*!
   * \brief Find the edge IDs of a given edge type, and return their source and target node IDs
   * \param etype The edge type
   * \param eids The edge ID array.
   * \return EdgeArray containing all edges with id in eid.  The order is preserved.
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual EdgeArray FindEdges(dgl_type_t etype, IdArray eids) const = 0;

  /*!
   * \brief Get the in edges of the vertex given the edge type.
   * \param etype The edge type
   * \param vid The vertex ID of edge destination type
   * \return The edges
   * \note The returned dst ID array is filled with vid.
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual EdgeArray InEdges(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Get the in edges of the vertices given the edge type.
   * \param etype The edge type
   * \param vid The vertex ID array of edge destination type
   * \return The edges
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual EdgeArray InEdges(dgl_type_t etype, IdArray vids) const = 0;

  /*!
   * \brief Get the out edges of the vertex given the edge type.
   * \param etype The edge type
   * \param vid The vertex ID of edge source type
   * \return The edges
   * \note The returned src ID array is filled with vid.
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual EdgeArray OutEdges(dgl_id_t vid) const = 0;

  /*!
   * \brief Get the out edges of the vertices given the edge type.
   * \param etype The edge type
   * \param vid The vertex ID array of edge source type
   * \return The edges
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual EdgeArray OutEdges(dgl_type_t etype, IdArray vids) const = 0;

  /*!
   * \brief Get all the edges in the graph given the edge type.
   * \note If order is "srcdst", the returned edges list is sorted by their src and
   *       dst ids. If order is "eid", they are in their edge id order.
   *       Otherwise, in the arbitrary order.
   * \param etype The edge type.
   * \param order The order of the returned edge list.
   * \return the id arrays of the two endpoints of the edges.
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual EdgeArray Edges(dgl_type_t etype, const std::string &order = "") const = 0;

  /*!
   * \brief Get the in degree of the given vertex given the edge type.
   * \param etype The edge type
   * \param vid The vertex ID of edge destination type
   * \return The in degree
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual uint64_t InDegree(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Get the in degrees of the given vertices given the edge type.
   * \param etype The edge type
   * \param vid The vertex ID array of edge destination type
   * \return The in degree array
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual uint64_t InDegrees(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Get the out degree of the given vertex given the edge type.
   * \param etype The edge type
   * \param vid The vertex ID of edge source type
   * \return The out degree
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual uint64_t OutDegree(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Get the out degrees of the given vertices given the edge type.
   * \param etype The edge type
   * \param vid The vertex ID array of edge source type
   * \return The out degree array
   * \note Source node type and destination node type are found according to the
   * metagraph.
   */
  virtual uint64_t OutDegrees(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Construct the induced subgraph of the given vertices.
   *
   * The induced subgraph is a subgraph formed by specifying a set of vertices V' and then
   * selecting all of the edges from the original graph that connect two vertices in V'.
   *
   * Vertices and edges in the original graph will be "reindexed" to local index. The local
   * index of the vertices preserve the order of the given id array, while the local index
   * of the edges preserve the index order in the original graph. Vertices not in the
   * original graph are ignored.
   *
   * The result subgraph is read-only.
   *
   * \param n Number of nodes by node type in vids.
   * \param vids The vertices in the subgraph.
   * \return the induced subgraph
   * \note For heterogeneous graphs, the vertices selected for inducing include:
   *       Vertex ID vids[0:n[0]] with vertex type 0
   *       Vertex ID vids[n[0]:n[0] + n[1]] with vertex type 1
   *       Vertex ID vids[n[0] + n[1]:n[0] + n[1] + n[2]] with vertex type 2
   *       etc.
   */
  virtual HeteroSubgraph VertexSubgraph(IdArray n, IdArray vids) const = 0;

  /*!
   * \brief Construct the induced edge subgraph of the given edges.
   *
   * The induced edges subgraph is a subgraph formed by specifying a set of edges E' and then
   * selecting all of the nodes from the original graph that are endpoints in E'.
   *
   * Vertices and edges in the original graph will be "reindexed" to local index. The local
   * index of the edges preserve the order of the given id array, while the local index
   * of the vertices preserve the index order in the original graph. Edges not in the
   * original graph are ignored.
   *
   * The result subgraph is read-only.
   *
   * \param n Number of edges by edge type in eids.
   * \param eids The edges in the subgraph.
   * \return the induced edge subgraph
   * \note For heterogeneous graphs, the edges selected for inducing include:
   *       Edge ID eids[0:n[0]] with edge type 0
   *       Edge ID eids[n[0]:n[0] + n[1]] with edge type 1
   *       Edge ID eids[n[0] + n[1]:n[0] + n[1] + n[2]] with edge type 2
   *       etc.
   */
  virtual HeteroSubgraph EdgeSubgraph(IdArray n, IdArray eids, bool preserve_nodes = false) const = 0;

 private:
  GraphPtr metagraph_;
};

// TODO
struct HeteroSubgraph : public Subgraph {
};

};  // namespace dgl

#endif  // DGL_HETEROGRAPH_INTERFACE_H_
