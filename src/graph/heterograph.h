/**
 *  Copyright (c) 2019 by Contributors
 * @file graph/heterograph.h
 * @brief Heterograph
 */

#ifndef DGL_GRAPH_HETEROGRAPH_H_
#define DGL_GRAPH_HETEROGRAPH_H_

#include <dgl/base_heterograph.h>
#include <dgl/lazy.h>
#include <dgl/runtime/shared_mem.h>

#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "./unit_graph.h"
#include "shared_mem_manager.h"

namespace dgl {

/** @brief Heterograph */
class HeteroGraph : public BaseHeteroGraph {
 public:
  HeteroGraph(
      GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& rel_graphs,
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

  void Clear() override { LOG(FATAL) << "Bipartite graph is not mutable."; }

  DGLDataType DataType() const override {
    return relation_graphs_[0]->DataType();
  }

  DGLContext Context() const override { return relation_graphs_[0]->Context(); }

  bool IsPinned() const override { return relation_graphs_[0]->IsPinned(); }

  uint8_t NumBits() const override { return relation_graphs_[0]->NumBits(); }

  bool IsMultigraph() const override;

  bool IsReadonly() const override { return true; }

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

  bool HasEdgeBetween(
      dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const override {
    return GetRelationGraph(etype)->HasEdgeBetween(0, src, dst);
  }

  BoolArray HasEdgesBetween(
      dgl_type_t etype, IdArray src_ids, IdArray dst_ids) const override {
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

  EdgeArray EdgeIdsAll(
      dgl_type_t etype, IdArray src, IdArray dst) const override {
    return GetRelationGraph(etype)->EdgeIdsAll(0, src, dst);
  }

  IdArray EdgeIdsOne(
      dgl_type_t etype, IdArray src, IdArray dst) const override {
    return GetRelationGraph(etype)->EdgeIdsOne(0, src, dst);
  }

  std::pair<dgl_id_t, dgl_id_t> FindEdge(
      dgl_type_t etype, dgl_id_t eid) const override {
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

  EdgeArray Edges(
      dgl_type_t etype, const std::string& order = "") const override {
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
      dgl_type_t etype, bool transpose, const std::string& fmt) const override {
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

  SparseFormat SelectFormat(
      dgl_type_t etype, dgl_format_code_t preferred_formats) const override {
    return GetRelationGraph(etype)->SelectFormat(0, preferred_formats);
  }

  dgl_format_code_t GetAllowedFormats() const override {
    return GetRelationGraph(0)->GetAllowedFormats();
  }

  dgl_format_code_t GetCreatedFormats() const override {
    return GetRelationGraph(0)->GetCreatedFormats();
  }

  HeteroSubgraph VertexSubgraph(
      const std::vector<IdArray>& vids) const override;

  HeteroSubgraph EdgeSubgraph(
      const std::vector<IdArray>& eids,
      bool preserve_nodes = false) const override;

  HeteroGraphPtr GetGraphInFormat(dgl_format_code_t formats) const override;

  FlattenedHeteroGraphPtr Flatten(
      const std::vector<dgl_type_t>& etypes) const override;

  GraphPtr AsImmutableGraph() const override;

  /** @return Load HeteroGraph from stream, using CSRMatrix*/
  bool Load(dmlc::Stream* fs);

  /** @return Save HeteroGraph to stream, using CSRMatrix */
  void Save(dmlc::Stream* fs) const;

  /** @brief Convert the graph to use the given number of bits for storage */
  static HeteroGraphPtr AsNumBits(HeteroGraphPtr g, uint8_t bits);

  /** @brief Copy the data to another context */
  static HeteroGraphPtr CopyTo(HeteroGraphPtr g, const DGLContext& ctx);

  /**
   * @brief Pin all relation graphs of the current graph.
   * @note The graph will be pinned inplace. Behavior depends on the current
   * context, kDGLCPU: will be pinned; IsPinned: directly return; kDGLCUDA:
   * invalid, will throw an error. The context check is deferred to pinning the
   * NDArray.
   */
  void PinMemory_() override;

  /**
   * @brief Unpin all relation graphs of the current graph.
   * @note The graph will be unpinned inplace. Behavior depends on the current
   * context, IsPinned: will be unpinned; others: directly return. The context
   * check is deferred to unpinning the NDArray.
   */
  void UnpinMemory_();

  /**
   * @brief Copy the current graph to pinned memory managed by
   *     PyTorch CachingHostAllocator for each relation graph.
   * @note If any of the underlying relation graphs are already pinned, the
   *     function will utilize their existing copies. If all of them are
   *     pinned, the function will return the original input hetero-graph
   *     directly.
   */
  static HeteroGraphPtr PinMemory(HeteroGraphPtr g);

  /**
   * @brief Record stream for this graph.
   * @param stream The stream that is using the graph
   */
  void RecordStream(DGLStreamHandle stream) override;

  /**
   * @brief Copy the data to shared memory.
   *
   * Also save names of node types and edge types of the HeteroGraph object to
   * shared memory
   */
  static HeteroGraphPtr CopyToSharedMem(
      HeteroGraphPtr g, const std::string& name,
      const std::vector<std::string>& ntypes,
      const std::vector<std::string>& etypes,
      const std::set<std::string>& fmts);

  /**
   * @brief Create a heterograph from
   *
   * @return the HeteroGraphPtr, names of node types, names of edge types
   */
  static std::tuple<
      HeteroGraphPtr, std::vector<std::string>, std::vector<std::string>>
  CreateFromSharedMem(const std::string& name);

  /** @brief Creat a LineGraph of self */
  HeteroGraphPtr LineGraph(bool backtracking) const;

  const std::vector<UnitGraphPtr>& relation_graphs() const {
    return relation_graphs_;
  }

 private:
  // To create empty class
  friend class Serializer;

  // Empty Constructor, only for serializer
  HeteroGraph() : BaseHeteroGraph() {}

  /** @brief A map from edge type to unit graph */
  std::vector<UnitGraphPtr> relation_graphs_;

  /** @brief A map from vert type to the number of verts in the type */
  std::vector<int64_t> num_verts_per_type_;

  /** @brief The shared memory object for meta info*/
  std::shared_ptr<runtime::SharedMemory> shared_mem_;

  /**
   * @brief The name of the shared memory. Return empty string if it is not in
   * shared memory.
   */
  std::string SharedMemName() const;

  /**
   * @brief template class for Flatten operation
   *
   * @tparam IdType Graph's index data type, can be int32_t or int64_t
   * @param etypes vector of etypes to be falttened
   * @return pointer of FlattenedHeteroGraphh
   */
  template <class IdType>
  FlattenedHeteroGraphPtr FlattenImpl(
      const std::vector<dgl_type_t>& etypes) const;
};

}  // namespace dgl

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, dgl::HeteroGraph, true);
}  // namespace dmlc

#endif  // DGL_GRAPH_HETEROGRAPH_H_
