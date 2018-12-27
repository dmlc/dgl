/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/immutable_graph.cc
 * \brief DGL immutable graph index implementation
 */

BoolArray HasVertices(IdArray vids) const;

bool HasEdgeBetween(dgl_id_t src, dgl_id_t dst) const;

BoolArray HasEdgesBetween(IdArray src_ids, IdArray dst_ids) const;

IdArray Predecessors(dgl_id_t vid, uint64_t radius = 1) const;

IdArray Successors(dgl_id_t vid, uint64_t radius = 1) const;

IdArray EdgeId(dgl_id_t src, dgl_id_t dst) const;

EdgeArray EdgeIds(IdArray src, IdArray dst) const;

EdgeArray FindEdges(IdArray eids) const;

EdgeArray InEdges(dgl_id_t vid) const;

EdgeArray InEdges(IdArray vids) const;

EdgeArray OutEdges(dgl_id_t vid) const;

EdgeArray OutEdges(IdArray vids) const;

EdgeArray Edges(bool sorted = false) const;

DegreeArray InDegrees(IdArray vids) const;

DegreeArray OutDegrees(IdArray vids) const;

Subgraph VertexSubgraph(IdArray vids) const;

Subgraph EdgeSubgraph(IdArray eids) const;

Graph Reverse() const;

  /*!
  * \brief Return the successor vector
  * \param vid The vertex id.
  * \return the successor vector
  */
  const std::vector<dgl_id_t>& SuccVec(dgl_id_t vid) const {
    return adjlist_[vid].succ;
  }

  /*!
  * \brief Return the out edge id vector
  * \param vid The vertex id.
  * \return the out edge id vector
  */
  const std::vector<dgl_id_t>& OutEdgeVec(dgl_id_t vid) const {
    return adjlist_[vid].edge_id;
  }

  /*!
  * \brief Return the predecessor vector
  * \param vid The vertex id.
  * \return the predecessor vector
  */
  const std::vector<dgl_id_t>& PredVec(dgl_id_t vid) const {
    return reverse_adjlist_[vid].succ;
  }

  /*!
  * \brief Return the in edge id vector
  * \param vid The vertex id.
  * \return the in edge id vector
  */
  const std::vector<dgl_id_t>& InEdgeVec(dgl_id_t vid) const {
    return reverse_adjlist_[vid].edge_id;
  }
