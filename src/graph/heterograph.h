/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/heterograph.h
 * \brief Heterograph
 */

#ifndef DGL_GRAPH_HETEROGRAPH_H_
#define DGL_GRAPH_HETEROGRAPH_H_

#include <dgl/base_heterograph.h>
#include <dgl/lazy.h>
#include <utility>
#include <string>
#include <vector>

namespace dgl {

/*! \brief Heterograph */
class HeteroGraph : public BaseHeteroGraph {
 public:
  HeteroGraph(GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& rel_graphs);

  HeteroGraphPtr GetRelationGraph(dgl_type_t etype) const override {
    CHECK_LT(etype, meta_graph_->NumEdges()) << "Invalid edge type: " << etype;
    return relation_graphs_[etype];
  }

  void AddVertices(dgl_type_t vtype, uint64_t num_vertices) override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
  }

  void AddEdge(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
  }

  void AddEdges(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
  }

  void Clear() override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
  }

  DLContext Context() const override {
    return relation_graphs_[0]->Context();
  }

  uint8_t NumBits() const override {
    return relation_graphs_[0]->NumBits();
  }

  bool IsMultigraph() const override;

  bool IsReadonly() const override {
    return true;
  }

  uint64_t NumVertices(dgl_type_t vtype) const override {
    CHECK(meta_graph_->HasVertex(vtype)) << "Invalid vertex type: " << vtype;
    return num_verts_per_type_[vtype];
  }

  uint64_t NumEdges(dgl_type_t etype) const override {
    return GetRelationGraph(etype)->NumEdges(0);
  }

  bool HasVertex(dgl_type_t vtype, dgl_id_t vid) const override {
    return vid < NumVertices(vtype);
  }

  BoolArray HasVertices(dgl_type_t vtype, IdArray vids) const override;

  bool HasEdgeBetween(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const override {
    return GetRelationGraph(etype)->HasEdgeBetween(0, src, dst);
  }

  BoolArray HasEdgesBetween(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) const override {
    return GetRelationGraph(etype)->HasEdgesBetween(0, src_ids, dst_ids);
  }

  IdArray Predecessors(dgl_type_t etype, dgl_id_t dst) const override {
    return GetRelationGraph(etype)->Predecessors(0, dst);
  }

  IdArray Successors(dgl_type_t etype, dgl_id_t src) const override {
    return GetRelationGraph(etype)->Successors(0, src);
  }

  IdArray EdgeId(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const override {
    return GetRelationGraph(etype)->EdgeId(0, src, dst);
  }

  EdgeArray EdgeIds(dgl_type_t etype, IdArray src, IdArray dst) const override {
    return GetRelationGraph(etype)->EdgeIds(0, src, dst);
  }

  std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_type_t etype, dgl_id_t eid) const override {
    return GetRelationGraph(etype)->FindEdge(0, eid);
  }

  EdgeArray FindEdges(dgl_type_t etype, IdArray eids) const override {
    return GetRelationGraph(etype)->FindEdges(0, eids);
  }

  EdgeArray InEdges(dgl_type_t etype, dgl_id_t vid) const override {
    return GetRelationGraph(etype)->InEdges(0, vid);
  }

  EdgeArray InEdges(dgl_type_t etype, IdArray vids) const override {
    return GetRelationGraph(etype)->InEdges(0, vids);
  }

  EdgeArray OutEdges(dgl_type_t etype, dgl_id_t vid) const override {
    return GetRelationGraph(etype)->OutEdges(0, vid);
  }

  EdgeArray OutEdges(dgl_type_t etype, IdArray vids) const override {
    return GetRelationGraph(etype)->OutEdges(0, vids);
  }

  EdgeArray Edges(dgl_type_t etype, const std::string &order = "") const override {
    return GetRelationGraph(etype)->Edges(0, order);
  }

  uint64_t InDegree(dgl_type_t etype, dgl_id_t vid) const override {
    return GetRelationGraph(etype)->InDegree(0, vid);
  }

  DegreeArray InDegrees(dgl_type_t etype, IdArray vids) const override {
    return GetRelationGraph(etype)->InDegrees(0, vids);
  }

  uint64_t OutDegree(dgl_type_t etype, dgl_id_t vid) const override {
    return GetRelationGraph(etype)->OutDegree(0, vid);
  }

  DegreeArray OutDegrees(dgl_type_t etype, IdArray vids) const override {
    return GetRelationGraph(etype)->OutDegrees(0, vids);
  }

  DGLIdIters SuccVec(dgl_type_t etype, dgl_id_t vid) const override {
    return GetRelationGraph(etype)->SuccVec(0, vid);
  }

  DGLIdIters OutEdgeVec(dgl_type_t etype, dgl_id_t vid) const override {
    return GetRelationGraph(etype)->OutEdgeVec(0, vid);
  }

  DGLIdIters PredVec(dgl_type_t etype, dgl_id_t vid) const override {
    return GetRelationGraph(etype)->PredVec(0, vid);
  }

  DGLIdIters InEdgeVec(dgl_type_t etype, dgl_id_t vid) const override {
    return GetRelationGraph(etype)->InEdgeVec(0, vid);
  }

  std::vector<IdArray> GetAdj(
      dgl_type_t etype, bool transpose, const std::string &fmt) const override {
    return GetRelationGraph(etype)->GetAdj(0, transpose, fmt);
  }

  HeteroSubgraph VertexSubgraph(const std::vector<IdArray>& vids) const override;

  HeteroSubgraph EdgeSubgraph(
      const std::vector<IdArray>& eids, bool preserve_nodes = false) const override;

  FlattenedHeteroGraphPtr Flatten(const std::vector<dgl_type_t>& etypes) const override;

 private:
  /*! \brief A map from edge type to unit graph */
  std::vector<HeteroGraphPtr> relation_graphs_;

  /*! \brief A map from vert type to the number of verts in the type */
  std::vector<int64_t> num_verts_per_type_;

  /*! \brief True if the graph is a multigraph */
  Lazy<bool> is_multigraph_;
};

}  // namespace dgl

#endif  // DGL_GRAPH_HETEROGRAPH_H_
