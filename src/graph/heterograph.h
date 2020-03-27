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
#include "./unit_graph.h"

namespace dgl {

/*! \brief Heterograph */
class HeteroGraph : public BaseHeteroGraph {
 public:
  HeteroGraph(
      GraphPtr meta_graph,
      const std::vector<HeteroGraphPtr>& rel_graphs,
      const std::vector<int64_t>& num_nodes_per_type = {});

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

  DLDataType DataType() const override {
    return relation_graphs_[0]->DataType();
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

  inline std::vector<int64_t> NumVerticesPerType() const override {
    return num_verts_per_type_;
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

  aten::COOMatrix GetCOOMatrix(dgl_type_t etype) const override {
    return GetRelationGraph(etype)->GetCOOMatrix(0);
  }

  aten::CSRMatrix GetCSCMatrix(dgl_type_t etype) const override {
    return GetRelationGraph(etype)->GetCSCMatrix(0);
  }

  aten::CSRMatrix GetCSRMatrix(dgl_type_t etype) const override {
    return GetRelationGraph(etype)->GetCSRMatrix(0);
  }

  SparseFormat SelectFormat(dgl_type_t etype, SparseFormat preferred_format) const override {
    return GetRelationGraph(etype)->SelectFormat(0, preferred_format);
  }

  HeteroSubgraph VertexSubgraph(const std::vector<IdArray>& vids) const override;

  HeteroSubgraph EdgeSubgraph(
      const std::vector<IdArray>& eids, bool preserve_nodes = false) const override;

  FlattenedHeteroGraphPtr Flatten(const std::vector<dgl_type_t>& etypes) const override;

  GraphPtr AsImmutableGraph() const override;

  /*! \return Load HeteroGraph from stream, using CSRMatrix*/
  bool Load(dmlc::Stream* fs);

  /*! \return Save HeteroGraph to stream, using CSRMatrix */
  void Save(dmlc::Stream* fs) const;

 private:
  // To create empty class
  friend class Serializer;

  // Empty Constructor, only for serializer
  HeteroGraph() : BaseHeteroGraph() {}

  /*! \brief A map from edge type to unit graph */
  std::vector<UnitGraphPtr> relation_graphs_;

  /*! \brief A map from vert type to the number of verts in the type */
  std::vector<int64_t> num_verts_per_type_;
};

}  // namespace dgl


namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, dgl::HeteroGraph, true);
}  // namespace dmlc


#endif  // DGL_GRAPH_HETEROGRAPH_H_
