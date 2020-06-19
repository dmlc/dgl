/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/heterograph.cc
 * \brief Heterograph implementation
 */
#include "./heterograph.h"
#include <dgl/array.h>
#include <dgl/immutable_graph.h>
#include <dgl/graph_serializer.h>
#include <vector>
#include <tuple>
#include <utility>

using namespace dgl::runtime;

namespace dgl {
namespace {

using dgl::ImmutableGraph;

HeteroSubgraph EdgeSubgraphPreserveNodes(
    const HeteroGraph* hg, const std::vector<IdArray>& eids) {
  CHECK_EQ(eids.size(), hg->NumEdgeTypes())
    << "Invalid input: the input list size must be the same as the number of edge type.";
  HeteroSubgraph ret;
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = eids;
  // When preserve_nodes is true, simply compute EdgeSubgraph for each bipartite
  std::vector<HeteroGraphPtr> subrels(hg->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const auto& rel_vsg = hg->GetRelationGraph(etype)->EdgeSubgraph(
        {eids[etype]}, true);
    subrels[etype] = rel_vsg.graph;
    ret.induced_vertices[src_vtype] = rel_vsg.induced_vertices[0];
    ret.induced_vertices[dst_vtype] = rel_vsg.induced_vertices[1];
  }
  ret.graph = HeteroGraphPtr(new HeteroGraph(
      hg->meta_graph(), subrels, hg->NumVerticesPerType()));
  return ret;
}

HeteroSubgraph EdgeSubgraphNoPreserveNodes(
    const HeteroGraph* hg, const std::vector<IdArray>& eids) {
  CHECK_EQ(eids.size(), hg->NumEdgeTypes())
    << "Invalid input: the input list size must be the same as the number of edge type.";
  HeteroSubgraph ret;
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = eids;
  // NOTE(minjie): EdgeSubgraph when preserve_nodes is false is quite complicated in
  // heterograph. This is because we need to make sure bipartite graphs that incident
  // on the same vertex type must have the same ID space. For example, suppose we have
  // following heterograph:
  //
  // Meta graph: A -> B -> C
  // UnitGraph graphs:
  // * A -> B: (0, 0), (0, 1)
  // * B -> C: (1, 0), (1, 1)
  //
  // Suppose for A->B, we only keep edge (0, 0), while for B->C we only keep (1, 0). We need
  // to make sure that in the result subgraph, node type B still has two nodes. This means
  // we cannot simply compute EdgeSubgraph for B->C which will relabel node#1 of type B to be
  // node #0.
  //
  // One implementation is as follows:
  // (1) For each bipartite graph, slice out the edges using the given eids.
  // (2) Make a dictionary map<vtype, vector<IdArray>>, where the key is the vertex type
  //     and the value is the incident nodes from the bipartite graphs that has the vertex
  //     type as either srctype or dsttype.
  // (3) Then for each vertex type, use aten::Relabel_ on its vector<IdArray>.
  //     aten::Relabel_ computes the union of the vertex sets and relabel
  //     the unique elements from zero. The returned mapping array is the final induced
  //     vertex set for that vertex type.
  // (4) Use the relabeled edges to construct the bipartite graph.
  // step (1) & (2)
  std::vector<EdgeArray> subedges(hg->NumEdgeTypes());
  std::vector<std::vector<IdArray>> vtype2incnodes(hg->NumVertexTypes());
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    auto earray = hg->GetRelationGraph(etype)->FindEdges(0, eids[etype]);
    vtype2incnodes[src_vtype].push_back(earray.src);
    vtype2incnodes[dst_vtype].push_back(earray.dst);
    subedges[etype] = earray;
  }
  // step (3)
  std::vector<int64_t> num_vertices_per_type(hg->NumVertexTypes());
  for (dgl_type_t vtype = 0; vtype < hg->NumVertexTypes(); ++vtype) {
    ret.induced_vertices[vtype] = aten::Relabel_(vtype2incnodes[vtype]);
    num_vertices_per_type[vtype] = ret.induced_vertices[vtype]->shape[0];
  }
  // step (4)
  std::vector<HeteroGraphPtr> subrels(hg->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    subrels[etype] = UnitGraph::CreateFromCOO(
      (src_vtype == dst_vtype)? 1 : 2,
      ret.induced_vertices[src_vtype]->shape[0],
      ret.induced_vertices[dst_vtype]->shape[0],
      subedges[etype].src,
      subedges[etype].dst);
  }
  ret.graph = HeteroGraphPtr(new HeteroGraph(
      hg->meta_graph(), subrels, std::move(num_vertices_per_type)));
  return ret;
}

void HeteroGraphSanityCheck(GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& rel_graphs) {
  // Sanity check
  CHECK_EQ(meta_graph->NumEdges(), rel_graphs.size());
  CHECK(!rel_graphs.empty()) << "Empty heterograph is not allowed.";
  // all relation graphs must have only one edge type
  for (const auto &rg : rel_graphs) {
    CHECK_EQ(rg->NumEdgeTypes(), 1) << "Each relation graph must have only one edge type.";
  }
}

std::vector<int64_t>
InferNumVerticesPerType(GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& rel_graphs) {
  // create num verts per type
  std::vector<int64_t> num_verts_per_type(meta_graph->NumVertices(), -1);

  EdgeArray etype_array = meta_graph->Edges();
  dgl_type_t *srctypes = static_cast<dgl_type_t *>(etype_array.src->data);
  dgl_type_t *dsttypes = static_cast<dgl_type_t *>(etype_array.dst->data);
  dgl_type_t *etypes = static_cast<dgl_type_t *>(etype_array.id->data);

  for (size_t i = 0; i < meta_graph->NumEdges(); ++i) {
    dgl_type_t srctype = srctypes[i];
    dgl_type_t dsttype = dsttypes[i];
    dgl_type_t etype = etypes[i];
    const auto& rg = rel_graphs[etype];
    const auto sty = 0;
    const auto dty = rg->NumVertexTypes() == 1? 0 : 1;
    size_t nv;

    // # nodes of source type
    nv = rg->NumVertices(sty);
    if (num_verts_per_type[srctype] < 0)
      num_verts_per_type[srctype] = nv;
    else
      CHECK_EQ(num_verts_per_type[srctype], nv)
        << "Mismatch number of vertices for vertex type " << srctype;
    // # nodes of destination type
    nv = rg->NumVertices(dty);
    if (num_verts_per_type[dsttype] < 0)
      num_verts_per_type[dsttype] = nv;
    else
      CHECK_EQ(num_verts_per_type[dsttype], nv)
        << "Mismatch number of vertices for vertex type " << dsttype;
  }
  return num_verts_per_type;
}

std::vector<UnitGraphPtr> CastToUnitGraphs(const std::vector<HeteroGraphPtr>& rel_graphs) {
  std::vector<UnitGraphPtr> relation_graphs(rel_graphs.size());
  for (size_t i = 0; i < rel_graphs.size(); ++i) {
    HeteroGraphPtr relg = rel_graphs[i];
    if (std::dynamic_pointer_cast<UnitGraph>(relg)) {
      relation_graphs[i] = std::dynamic_pointer_cast<UnitGraph>(relg);
    } else {
      relation_graphs[i] = CHECK_NOTNULL(
          std::dynamic_pointer_cast<UnitGraph>(relg->GetRelationGraph(0)));
    }
  }
  return relation_graphs;
}

}  // namespace

HeteroGraph::HeteroGraph(
    GraphPtr meta_graph,
    const std::vector<HeteroGraphPtr>& rel_graphs,
    const std::vector<int64_t>& num_nodes_per_type) : BaseHeteroGraph(meta_graph) {
  if (num_nodes_per_type.size() == 0)
    num_verts_per_type_ = InferNumVerticesPerType(meta_graph, rel_graphs);
  else
    num_verts_per_type_ = num_nodes_per_type;

  HeteroGraphSanityCheck(meta_graph, rel_graphs);
  relation_graphs_ = CastToUnitGraphs(rel_graphs);
}

bool HeteroGraph::IsMultigraph() const {
  for (const auto &hg : relation_graphs_) {
    if (hg->IsMultigraph()) {
      return true;
    }
  }
  return false;
}

BoolArray HeteroGraph::HasVertices(dgl_type_t vtype, IdArray vids) const {
  CHECK(aten::IsValidIdArray(vids)) << "Invalid id array input";
  return aten::LT(vids, NumVertices(vtype));
}

HeteroSubgraph HeteroGraph::VertexSubgraph(const std::vector<IdArray>& vids) const {
  CHECK_EQ(vids.size(), NumVertexTypes())
    << "Invalid input: the input list size must be the same as the number of vertex types.";
  HeteroSubgraph ret;
  ret.induced_vertices = vids;
  std::vector<int64_t> num_vertices_per_type(NumVertexTypes());
  for (dgl_type_t vtype = 0; vtype < NumVertexTypes(); ++vtype)
    num_vertices_per_type[vtype] = vids[vtype]->shape[0];
  ret.induced_edges.resize(NumEdgeTypes());
  std::vector<HeteroGraphPtr> subrels(NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < NumEdgeTypes(); ++etype) {
    auto pair = meta_graph_->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const std::vector<IdArray> rel_vids = (src_vtype == dst_vtype) ?
      std::vector<IdArray>({vids[src_vtype]}) :
      std::vector<IdArray>({vids[src_vtype], vids[dst_vtype]});
    const auto& rel_vsg = GetRelationGraph(etype)->VertexSubgraph(rel_vids);
    subrels[etype] = rel_vsg.graph;
    ret.induced_edges[etype] = rel_vsg.induced_edges[0];
  }
  ret.graph = HeteroGraphPtr(new HeteroGraph(
      meta_graph_, subrels, std::move(num_vertices_per_type)));
  return ret;
}

HeteroSubgraph HeteroGraph::EdgeSubgraph(
    const std::vector<IdArray>& eids, bool preserve_nodes) const {
  if (preserve_nodes) {
    return EdgeSubgraphPreserveNodes(this, eids);
  } else {
    return EdgeSubgraphNoPreserveNodes(this, eids);
  }
}

HeteroGraphPtr HeteroGraph::AsNumBits(HeteroGraphPtr g, uint8_t bits) {
  auto hgindex = std::dynamic_pointer_cast<HeteroGraph>(g);
  CHECK_NOTNULL(hgindex);
  std::vector<HeteroGraphPtr> rel_graphs;
  for (auto g : hgindex->relation_graphs_) {
    rel_graphs.push_back(UnitGraph::AsNumBits(g, bits));
  }
  return HeteroGraphPtr(new HeteroGraph(hgindex->meta_graph_, rel_graphs,
                                        hgindex->num_verts_per_type_));
}

HeteroGraphPtr HeteroGraph::CopyTo(HeteroGraphPtr g, const DLContext& ctx) {
  if (ctx == g->Context()) {
    return g;
  }
  auto hgindex = std::dynamic_pointer_cast<HeteroGraph>(g);
  CHECK_NOTNULL(hgindex);
  std::vector<HeteroGraphPtr> rel_graphs;
  for (auto g : hgindex->relation_graphs_) {
    rel_graphs.push_back(UnitGraph::CopyTo(g, ctx));
  }
  return HeteroGraphPtr(new HeteroGraph(hgindex->meta_graph_, rel_graphs,
                                        hgindex->num_verts_per_type_));
}

HeteroGraphPtr HeteroGraph::GetGraphInFormat(SparseFormat restrict_format) const {
  std::vector<HeteroGraphPtr> format_rels(NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < NumEdgeTypes(); ++etype) {
    auto relgraph = std::dynamic_pointer_cast<UnitGraph>(GetRelationGraph(etype));
    format_rels[etype] = relgraph->GetGraphInFormat(restrict_format);
  }
  return HeteroGraphPtr(new HeteroGraph(
    meta_graph_, format_rels, NumVerticesPerType()));
}

FlattenedHeteroGraphPtr HeteroGraph::Flatten(
    const std::vector<dgl_type_t>& etypes) const {
  const int64_t bits = NumBits();
  if (bits == 32) {
    return FlattenImpl<int32_t>(etypes);
  } else {
    return FlattenImpl<int64_t>(etypes);
  }
}

template <class IdType>
FlattenedHeteroGraphPtr HeteroGraph::FlattenImpl(const std::vector<dgl_type_t>& etypes) const {
  std::unordered_map<dgl_type_t, size_t> srctype_offsets, dsttype_offsets;
  size_t src_nodes = 0, dst_nodes = 0;
  std::vector<IdType> result_src, result_dst;
  std::vector<dgl_type_t> induced_srctype, induced_etype, induced_dsttype;
  std::vector<IdType> induced_srcid, induced_eid, induced_dstid;
  std::vector<dgl_type_t> srctype_set, dsttype_set;

  // XXXtype_offsets contain the mapping from node type and number of nodes after this
  // loop.
  for (dgl_type_t etype : etypes) {
    auto src_dsttype = meta_graph_->FindEdge(etype);
    dgl_type_t srctype = src_dsttype.first;
    dgl_type_t dsttype = src_dsttype.second;
    size_t num_srctype_nodes = NumVertices(srctype);
    size_t num_dsttype_nodes = NumVertices(dsttype);

    if (srctype_offsets.count(srctype) == 0) {
      srctype_offsets[srctype] = num_srctype_nodes;
      srctype_set.push_back(srctype);
    }
    if (dsttype_offsets.count(dsttype) == 0) {
      dsttype_offsets[dsttype] = num_dsttype_nodes;
      dsttype_set.push_back(dsttype);
    }
  }
  // Sort the node types so that we can compare the sets and decide whether a homograph
  // should be returned.
  std::sort(srctype_set.begin(), srctype_set.end());
  std::sort(dsttype_set.begin(), dsttype_set.end());
  bool homograph = (srctype_set.size() == dsttype_set.size()) &&
    std::equal(srctype_set.begin(), srctype_set.end(), dsttype_set.begin());

  // XXXtype_offsets contain the mapping from node type to node ID offsets after these
  // two loops.
  for (size_t i = 0; i < srctype_set.size(); ++i) {
    dgl_type_t ntype = srctype_set[i];
    size_t num_nodes = srctype_offsets[ntype];
    srctype_offsets[ntype] = src_nodes;
    src_nodes += num_nodes;
    for (size_t j = 0; j < num_nodes; ++j) {
      induced_srctype.push_back(ntype);
      induced_srcid.push_back(j);
    }
  }
  for (size_t i = 0; i < dsttype_set.size(); ++i) {
    dgl_type_t ntype = dsttype_set[i];
    size_t num_nodes = dsttype_offsets[ntype];
    dsttype_offsets[ntype] = dst_nodes;
    dst_nodes += num_nodes;
    for (size_t j = 0; j < num_nodes; ++j) {
      induced_dsttype.push_back(ntype);
      induced_dstid.push_back(j);
    }
  }

  for (dgl_type_t etype : etypes) {
    auto src_dsttype = meta_graph_->FindEdge(etype);
    dgl_type_t srctype = src_dsttype.first;
    dgl_type_t dsttype = src_dsttype.second;
    size_t srctype_offset = srctype_offsets[srctype];
    size_t dsttype_offset = dsttype_offsets[dsttype];

    EdgeArray edges = Edges(etype);
    size_t num_edges = NumEdges(etype);
    const IdType* edges_src_data = static_cast<const IdType*>(edges.src->data);
    const IdType* edges_dst_data = static_cast<const IdType*>(edges.dst->data);
    const IdType* edges_eid_data = static_cast<const IdType*>(edges.id->data);
    // TODO(gq) Use concat?
    for (size_t i = 0; i < num_edges; ++i) {
      result_src.push_back(edges_src_data[i] + srctype_offset);
      result_dst.push_back(edges_dst_data[i] + dsttype_offset);
      induced_etype.push_back(etype);
      induced_eid.push_back(edges_eid_data[i]);
    }
  }

  HeteroGraphPtr gptr = UnitGraph::CreateFromCOO(
      homograph ? 1 : 2,
      src_nodes,
      dst_nodes,
      aten::VecToIdArray(result_src),
      aten::VecToIdArray(result_dst));

  FlattenedHeteroGraph* result = new FlattenedHeteroGraph;
  result->graph = HeteroGraphRef(gptr);
  result->induced_srctype = aten::VecToIdArray(induced_srctype);
  result->induced_srctype_set = aten::VecToIdArray(srctype_set);
  result->induced_srcid = aten::VecToIdArray(induced_srcid);
  result->induced_etype = aten::VecToIdArray(induced_etype);
  result->induced_etype_set = aten::VecToIdArray(etypes);
  result->induced_eid = aten::VecToIdArray(induced_eid);
  result->induced_dsttype = aten::VecToIdArray(induced_dsttype);
  result->induced_dsttype_set = aten::VecToIdArray(dsttype_set);
  result->induced_dstid = aten::VecToIdArray(induced_dstid);
  return FlattenedHeteroGraphPtr(result);
}

constexpr uint64_t kDGLSerialize_HeteroGraph = 0xDD589FBE35224ABF;

bool HeteroGraph::Load(dmlc::Stream* fs) {
  uint64_t magicNum;
  CHECK(fs->Read(&magicNum)) << "Invalid Magic Number";
  CHECK_EQ(magicNum, kDGLSerialize_HeteroGraph) << "Invalid HeteroGraph Data";
  auto meta_imgraph = Serializer::make_shared<ImmutableGraph>();
  CHECK(fs->Read(&meta_imgraph)) << "Invalid meta graph";
  meta_graph_ = meta_imgraph;
  CHECK(fs->Read(&relation_graphs_)) << "Invalid relation_graphs_";
  CHECK(fs->Read(&num_verts_per_type_)) << "Invalid num_verts_per_type_";
  return true;
}

void HeteroGraph::Save(dmlc::Stream* fs) const {
  fs->Write(kDGLSerialize_HeteroGraph);
  auto meta_graph_ptr = ImmutableGraph::ToImmutable(meta_graph());
  fs->Write(meta_graph_ptr);
  fs->Write(relation_graphs_);
  fs->Write(num_verts_per_type_);
}

GraphPtr HeteroGraph::AsImmutableGraph() const {
  CHECK(NumVertexTypes() == 1) << "graph has more than one node types";
  CHECK(NumEdgeTypes() == 1) << "graph has more than one edge types";
  auto unit_graph = CHECK_NOTNULL(
      std::dynamic_pointer_cast<UnitGraph>(GetRelationGraph(0)));
  return unit_graph->AsImmutableGraph();
}

}  // namespace dgl
