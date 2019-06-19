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

class HeteroGraphETypeViewInterface : public GraphInterface {
 public:
  virtual ~HeteroGraphViewInterface() = default;
  HeteroGraphViewInterface(
      HeteroGraphInterface *parent,
      dgl_type_t etype);

  /*
   * Interfaces to be implemented:
   *
   * AddEdge(src, dst)
   * AddEdges(src_ids, dst_ids)
   * NumEdges()
   * HasEdgeBetween(src, dst)
   * HasEdgesBetween(src, dst)
   * Predecessors(vid)
   * Successors(vid)
   * EdgeId(src, dst)
   * EdgeIds(src, dst)
   * FindEdge(eid)
   * FindEdges(eids)
   * InEdges(vid)
   * InEdges(vids)
   * OutEdges(vid)
   * OutEdges(vids)
   * Edges(order = "")
   * InDegree(vid)
   * InDegrees(vids)
   * OutDegree(vid)
   * OutDegrees(vids)
   */
}

class HeteroGraphInterface : public GraphInterface {
 public:
  virtual ~HeteroGraphInterface() = default;

  virtual HeteroGraphETypeViewInterface GetRelationGraph(
      dgl_type_t etype) const = 0;

  /*!
   * \brief Add vertices to the graph with the given node type.
   *
   * \param ntype the node type
   * \param num_vertices the number of vertices to add
   */
  virtual void AddVertices(dgl_type_t vtype, uint64_t num_vertices);

  /*! \return the number of vertices of given type in the graph */
  virtual uint64_t NumVertices(dgl_type_t vtype) const = 0;

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
};

// TODO
struct HeteroSubgraph : public Subgraph {
};

};  // namespace dgl

#endif  // DGL_HETEROGRAPH_INTERFACE_H_
