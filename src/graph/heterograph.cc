/**
 *  Copyright (c) 2019 by Contributors
 * @file graph/heterograph.cc
 * @brief Heterograph implementation
 */
#include "./heterograph.h"

#include <dgl/array.h>
#include <dgl/graph_serializer.h>
#include <dgl/immutable_graph.h>
#include <dmlc/memory_io.h>

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

using namespace dgl::runtime;

namespace dgl {
namespace {

using dgl::ImmutableGraph;

HeteroSubgraph EdgeSubgraphPreserveNodes(
    const HeteroGraph* hg, const std::vector<IdArray>& eids) {
  CHECK_EQ(eids.size(), hg->NumEdgeTypes())
      << "Invalid input: the input list size must be the same as the number of "
         "edge type.";
  HeteroSubgraph ret;
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = eids;
  // When preserve_nodes is true, simply compute EdgeSubgraph for each bipartite
  std::vector<HeteroGraphPtr> subrels(hg->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const auto& rel_vsg =
        hg->GetRelationGraph(etype)->EdgeSubgraph({eids[etype]}, true);
    subrels[etype] = rel_vsg.graph;
    ret.induced_vertices[src_vtype] = rel_vsg.induced_vertices[0];
    ret.induced_vertices[dst_vtype] = rel_vsg.induced_vertices[1];
  }
  ret.graph = HeteroGraphPtr(
      new HeteroGraph(hg->meta_graph(), subrels, hg->NumVerticesPerType()));
  return ret;
}

HeteroSubgraph EdgeSubgraphNoPreserveNodes(
    const HeteroGraph* hg, const std::vector<IdArray>& eids) {
  // TODO(minjie): In general, all relabeling should be separated with subgraph
  //   operations.
  CHECK_EQ(eids.size(), hg->NumEdgeTypes())
      << "Invalid input: the input list size must be the same as the number of "
         "edge type.";
  HeteroSubgraph ret;
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = eids;
  // NOTE(minjie): EdgeSubgraph when preserve_nodes is false is quite
  // complicated in heterograph. This is because we need to make sure bipartite
  // graphs that incident on the same vertex type must have the same ID space.
  // For example, suppose we have following heterograph:
  //
  // Meta graph: A -> B -> C
  // UnitGraph graphs:
  // * A -> B: (0, 0), (0, 1)
  // * B -> C: (1, 0), (1, 1)
  //
  // Suppose for A->B, we only keep edge (0, 0), while for B->C we only keep (1,
  // 0). We need to make sure that in the result subgraph, node type B still has
  // two nodes. This means we cannot simply compute EdgeSubgraph for B->C which
  // will relabel node#1 of type B to be node #0.
  //
  // One implementation is as follows:
  // (1) For each bipartite graph, slice out the edges using the given eids.
  // (2) Make a dictionary map<vtype, vector<IdArray>>, where the key is the
  // vertex type
  //     and the value is the incident nodes from the bipartite graphs that has
  //     the vertex type as either srctype or dsttype.
  // (3) Then for each vertex type, use aten::Relabel_ on its vector<IdArray>.
  //     aten::Relabel_ computes the union of the vertex sets and relabel
  //     the unique elements from zero. The returned mapping array is the final
  //     induced vertex set for that vertex type.
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
        (src_vtype == dst_vtype) ? 1 : 2,
        ret.induced_vertices[src_vtype]->shape[0],
        ret.induced_vertices[dst_vtype]->shape[0], subedges[etype].src,
        subedges[etype].dst);
  }
  ret.graph = HeteroGraphPtr(new HeteroGraph(
      hg->meta_graph(), subrels, std::move(num_vertices_per_type)));
  return ret;
}

void HeteroGraphSanityCheck(
    GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& rel_graphs) {
  // Sanity check
  CHECK_EQ(meta_graph->NumEdges(), rel_graphs.size());
  CHECK(!rel_graphs.empty()) << "Empty heterograph is not allowed.";
  // all relation graphs must have only one edge type
  for (const auto& rg : rel_graphs) {
    CHECK_EQ(rg->NumEdgeTypes(), 1)
        << "Each relation graph must have only one edge type.";
  }
  auto ctx = rel_graphs[0]->Context();
  for (const auto& rg : rel_graphs) {
    CHECK_EQ(rg->Context(), ctx)
        << "Each relation graph must have the same context.";
  }
}

std::vector<int64_t> InferNumVerticesPerType(
    GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& rel_graphs) {
  // create num verts per type
  std::vector<int64_t> num_verts_per_type(meta_graph->NumVertices(), -1);

  EdgeArray etype_array = meta_graph->Edges();
  dgl_type_t* srctypes = static_cast<dgl_type_t*>(etype_array.src->data);
  dgl_type_t* dsttypes = static_cast<dgl_type_t*>(etype_array.dst->data);
  dgl_type_t* etypes = static_cast<dgl_type_t*>(etype_array.id->data);
  for (size_t i = 0; i < meta_graph->NumEdges(); ++i) {
    dgl_type_t srctype = srctypes[i];
    dgl_type_t dsttype = dsttypes[i];
    dgl_type_t etype = etypes[i];
    const auto& rg = rel_graphs[etype];
    const auto sty = 0;
    const auto dty = rg->NumVertexTypes() == 1 ? 0 : 1;
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

std::vector<UnitGraphPtr> CastToUnitGraphs(
    const std::vector<HeteroGraphPtr>& rel_graphs) {
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
    GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& rel_graphs,
    const std::vector<int64_t>& num_nodes_per_type)
    : BaseHeteroGraph(meta_graph) {
  if (num_nodes_per_type.size() == 0)
    num_verts_per_type_ = InferNumVerticesPerType(meta_graph, rel_graphs);
  else
    num_verts_per_type_ = num_nodes_per_type;
  HeteroGraphSanityCheck(meta_graph, rel_graphs);
  relation_graphs_ = CastToUnitGraphs(rel_graphs);
}

bool HeteroGraph::IsMultigraph() const {
  for (const auto& hg : relation_graphs_) {
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

HeteroSubgraph HeteroGraph::VertexSubgraph(
    const std::vector<IdArray>& vids) const {
  CHECK_EQ(vids.size(), NumVertexTypes())
      << "Invalid input: the input list size must be the same as the number of "
         "vertex types.";
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
    const std::vector<IdArray> rel_vids =
        (src_vtype == dst_vtype)
            ? std::vector<IdArray>({vids[src_vtype]})
            : std::vector<IdArray>({vids[src_vtype], vids[dst_vtype]});
    const auto& rel_vsg = GetRelationGraph(etype)->VertexSubgraph(rel_vids);
    subrels[etype] = rel_vsg.graph;
    ret.induced_edges[etype] = rel_vsg.induced_edges[0];
  }
  ret.graph = HeteroGraphPtr(
      new HeteroGraph(meta_graph_, subrels, std::move(num_vertices_per_type)));
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
  return HeteroGraphPtr(new HeteroGraph(
      hgindex->meta_graph_, rel_graphs, hgindex->num_verts_per_type_));
}

HeteroGraphPtr HeteroGraph::CopyTo(HeteroGraphPtr g, const DGLContext& ctx) {
  if (ctx == g->Context()) {
    return g;
  }
  auto hgindex = std::dynamic_pointer_cast<HeteroGraph>(g);
  CHECK_NOTNULL(hgindex);
  std::vector<HeteroGraphPtr> rel_graphs;
  for (auto g : hgindex->relation_graphs_) {
    rel_graphs.push_back(UnitGraph::CopyTo(g, ctx));
  }
  return HeteroGraphPtr(new HeteroGraph(
      hgindex->meta_graph_, rel_graphs, hgindex->num_verts_per_type_));
}

HeteroGraphPtr HeteroGraph::PinMemory(HeteroGraphPtr g) {
  auto casted_ptr = std::dynamic_pointer_cast<HeteroGraph>(g);
  CHECK_NOTNULL(casted_ptr);
  auto relation_graphs = casted_ptr->relation_graphs_;

  auto it = std::find_if_not(
      relation_graphs.begin(), relation_graphs.end(),
      [](auto& underlying_g) { return underlying_g->IsPinned(); });
  // All underlying relation graphs are pinned, return the input hetero-graph
  // directly.
  if (it == relation_graphs.end()) return g;

  std::vector<HeteroGraphPtr> pinned_relation_graphs(relation_graphs.size());
  for (size_t i = 0; i < pinned_relation_graphs.size(); ++i) {
    if (!relation_graphs[i]->IsPinned()) {
      pinned_relation_graphs[i] = relation_graphs[i]->PinMemory();
    } else {
      pinned_relation_graphs[i] = relation_graphs[i];
    }
  }
  return HeteroGraphPtr(new HeteroGraph(
      casted_ptr->meta_graph_, pinned_relation_graphs,
      casted_ptr->num_verts_per_type_));
}

void HeteroGraph::PinMemory_() {
  for (auto g : relation_graphs_) g->PinMemory_();
}

void HeteroGraph::UnpinMemory_() {
  for (auto g : relation_graphs_) g->UnpinMemory_();
}

void HeteroGraph::RecordStream(DGLStreamHandle stream) {
  for (auto g : relation_graphs_) g->RecordStream(stream);
}

std::string HeteroGraph::SharedMemName() const {
  return shared_mem_ ? shared_mem_->GetName() : "";
}

HeteroGraphPtr HeteroGraph::CopyToSharedMem(
    HeteroGraphPtr g, const std::string& name,
    const std::vector<std::string>& ntypes,
    const std::vector<std::string>& etypes, const std::set<std::string>& fmts) {
  // TODO(JJ): Raise error when calling shared_memory if graph index is on gpu
  auto hg = std::dynamic_pointer_cast<HeteroGraph>(g);
  CHECK_NOTNULL(hg);
  if (hg->SharedMemName() == name) return g;

  // Copy buffer to share memory
  auto mem = std::make_shared<SharedMemory>(name);
  auto mem_buf = mem->CreateNew(SHARED_MEM_METAINFO_SIZE_MAX);
  dmlc::MemoryFixedSizeStream strm(mem_buf, SHARED_MEM_METAINFO_SIZE_MAX);
  SharedMemManager shm(name, &strm);

  bool has_coo = fmts.find("coo") != fmts.end();
  bool has_csr = fmts.find("csr") != fmts.end();
  bool has_csc = fmts.find("csc") != fmts.end();
  shm.Write(g->NumBits());
  shm.Write(has_coo);
  shm.Write(has_csr);
  shm.Write(has_csc);
  shm.Write(ImmutableGraph::ToImmutable(hg->meta_graph_));
  shm.Write(hg->num_verts_per_type_);

  std::vector<HeteroGraphPtr> relgraphs(g->NumEdgeTypes());

  for (dgl_type_t etype = 0; etype < g->NumEdgeTypes(); ++etype) {
    auto src_dst_type = g->GetEndpointTypes(etype);
    int num_vtypes = (src_dst_type.first == src_dst_type.second ? 1 : 2);
    aten::COOMatrix coo;
    aten::CSRMatrix csr, csc;
    std::string prefix = name + "_" + std::to_string(etype);
    if (has_coo) {
      coo = shm.CopyToSharedMem(hg->GetCOOMatrix(etype), prefix + "_coo");
    }
    if (has_csr) {
      csr = shm.CopyToSharedMem(hg->GetCSRMatrix(etype), prefix + "_csr");
    }
    if (has_csc) {
      csc = shm.CopyToSharedMem(hg->GetCSCMatrix(etype), prefix + "_csc");
    }
    relgraphs[etype] = UnitGraph::CreateUnitGraphFrom(
        num_vtypes, csc, csr, coo, has_csc, has_csr, has_coo);
  }

  auto ret = std::shared_ptr<HeteroGraph>(
      new HeteroGraph(hg->meta_graph_, relgraphs, hg->num_verts_per_type_));
  ret->shared_mem_ = mem;

  shm.Write(ntypes);
  shm.Write(etypes);
  return ret;
}

std::tuple<HeteroGraphPtr, std::vector<std::string>, std::vector<std::string>>
HeteroGraph::CreateFromSharedMem(const std::string& name) {
  bool exist = SharedMemory::Exist(name);
  if (!exist) {
    return std::make_tuple(
        nullptr, std::vector<std::string>(), std::vector<std::string>());
  }
  auto mem = std::make_shared<SharedMemory>(name);
  auto mem_buf = mem->Open(SHARED_MEM_METAINFO_SIZE_MAX);
  dmlc::MemoryFixedSizeStream strm(mem_buf, SHARED_MEM_METAINFO_SIZE_MAX);
  SharedMemManager shm(name, &strm);

  uint8_t nbits;
  CHECK(shm.Read(&nbits)) << "invalid nbits (unit8_t)";

  bool has_coo, has_csr, has_csc;
  CHECK(shm.Read(&has_coo)) << "invalid nbits (unit8_t)";
  CHECK(shm.Read(&has_csr)) << "invalid csr (unit8_t)";
  CHECK(shm.Read(&has_csc)) << "invalid csc (unit8_t)";

  auto meta_imgraph = Serializer::make_shared<ImmutableGraph>();
  CHECK(shm.Read(&meta_imgraph)) << "Invalid meta graph";
  GraphPtr metagraph = meta_imgraph;

  std::vector<int64_t> num_verts_per_type;
  CHECK(shm.Read(&num_verts_per_type)) << "Invalid number of vertices per type";

  std::vector<HeteroGraphPtr> relgraphs(metagraph->NumEdges());
  for (dgl_type_t etype = 0; etype < metagraph->NumEdges(); ++etype) {
    auto src_dst = metagraph->FindEdge(etype);
    int num_vtypes = (src_dst.first == src_dst.second) ? 1 : 2;
    aten::COOMatrix coo;
    aten::CSRMatrix csr, csc;
    std::string prefix = name + "_" + std::to_string(etype);
    if (has_coo) {
      shm.CreateFromSharedMem(&coo, prefix + "_coo");
    }
    if (has_csr) {
      shm.CreateFromSharedMem(&csr, prefix + "_csr");
    }
    if (has_csc) {
      shm.CreateFromSharedMem(&csc, prefix + "_csc");
    }

    relgraphs[etype] = UnitGraph::CreateUnitGraphFrom(
        num_vtypes, csc, csr, coo, has_csc, has_csr, has_coo);
  }

  auto ret =
      std::make_shared<HeteroGraph>(metagraph, relgraphs, num_verts_per_type);
  ret->shared_mem_ = mem;

  std::vector<std::string> ntypes;
  std::vector<std::string> etypes;
  CHECK(shm.Read(&ntypes)) << "invalid ntypes";
  CHECK(shm.Read(&etypes)) << "invalid etypes";
  return std::make_tuple(ret, ntypes, etypes);
}

HeteroGraphPtr HeteroGraph::GetGraphInFormat(dgl_format_code_t formats) const {
  std::vector<HeteroGraphPtr> format_rels(NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < NumEdgeTypes(); ++etype) {
    auto relgraph =
        std::dynamic_pointer_cast<UnitGraph>(GetRelationGraph(etype));
    format_rels[etype] = relgraph->GetGraphInFormat(formats);
  }
  return HeteroGraphPtr(
      new HeteroGraph(meta_graph_, format_rels, NumVerticesPerType()));
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
FlattenedHeteroGraphPtr HeteroGraph::FlattenImpl(
    const std::vector<dgl_type_t>& etypes) const {
  std::unordered_map<dgl_type_t, size_t> srctype_offsets, dsttype_offsets;
  size_t src_nodes = 0, dst_nodes = 0;
  std::vector<dgl_type_t> induced_srctype, induced_dsttype;
  std::vector<IdType> induced_srcid, induced_dstid;
  std::vector<dgl_type_t> srctype_set, dsttype_set;

  // XXXtype_offsets contain the mapping from node type and number of nodes
  // after this loop.
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
  // Sort the node types so that we can compare the sets and decide whether a
  // homogeneous graph should be returned.
  std::sort(srctype_set.begin(), srctype_set.end());
  std::sort(dsttype_set.begin(), dsttype_set.end());
  bool homograph =
      (srctype_set.size() == dsttype_set.size()) &&
      std::equal(srctype_set.begin(), srctype_set.end(), dsttype_set.begin());

  // XXXtype_offsets contain the mapping from node type to node ID offsets after
  // these two loops.
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

  // TODO(minjie): Using concat operations cause many fragmented memory.
  //   Need to optimize it in the future.
  std::vector<IdArray> src_arrs, dst_arrs, eid_arrs, induced_etypes;
  src_arrs.reserve(etypes.size());
  dst_arrs.reserve(etypes.size());
  eid_arrs.reserve(etypes.size());
  induced_etypes.reserve(etypes.size());
  for (dgl_type_t etype : etypes) {
    auto src_dsttype = meta_graph_->FindEdge(etype);
    dgl_type_t srctype = src_dsttype.first;
    dgl_type_t dsttype = src_dsttype.second;
    size_t srctype_offset = srctype_offsets[srctype];
    size_t dsttype_offset = dsttype_offsets[dsttype];

    EdgeArray edges = Edges(etype);
    size_t num_edges = NumEdges(etype);
    src_arrs.push_back(edges.src + srctype_offset);
    dst_arrs.push_back(edges.dst + dsttype_offset);
    eid_arrs.push_back(edges.id);
    induced_etypes.push_back(
        aten::Full(etype, num_edges, NumBits(), Context()));
  }

  HeteroGraphPtr gptr = UnitGraph::CreateFromCOO(
      homograph ? 1 : 2, src_nodes, dst_nodes, aten::Concat(src_arrs),
      aten::Concat(dst_arrs));

  // Sanity check
  CHECK_EQ(gptr->Context(), Context());
  CHECK_EQ(gptr->NumBits(), NumBits());

  FlattenedHeteroGraph* result = new FlattenedHeteroGraph;
  result->graph = HeteroGraphRef(
      HeteroGraphPtr(new HeteroGraph(gptr->meta_graph(), {gptr})));
  result->induced_srctype =
      aten::VecToIdArray(induced_srctype).CopyTo(Context());
  result->induced_srctype_set =
      aten::VecToIdArray(srctype_set).CopyTo(Context());
  result->induced_srcid = aten::VecToIdArray(induced_srcid).CopyTo(Context());
  result->induced_etype = aten::Concat(induced_etypes);
  result->induced_etype_set = aten::VecToIdArray(etypes).CopyTo(Context());
  result->induced_eid = aten::Concat(eid_arrs);
  result->induced_dsttype =
      aten::VecToIdArray(induced_dsttype).CopyTo(Context());
  result->induced_dsttype_set =
      aten::VecToIdArray(dsttype_set).CopyTo(Context());
  result->induced_dstid = aten::VecToIdArray(induced_dstid).CopyTo(Context());
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
  auto unit_graph =
      CHECK_NOTNULL(std::dynamic_pointer_cast<UnitGraph>(GetRelationGraph(0)));
  return unit_graph->AsImmutableGraph();
}

HeteroGraphPtr HeteroGraph::LineGraph(bool backtracking) const {
  CHECK_EQ(1, meta_graph_->NumEdges())
      << "Only support Homogeneous graph now (one edge type)";
  CHECK_EQ(1, meta_graph_->NumVertices())
      << "Only support Homogeneous graph now (one node type)";
  CHECK_EQ(1, relation_graphs_.size()) << "Only support Homogeneous graph now";
  UnitGraphPtr ug = relation_graphs_[0];

  const auto& ulg = ug->LineGraph(backtracking);
  std::vector<HeteroGraphPtr> rel_graph = {ulg};
  std::vector<int64_t> num_nodes_per_type = {
      static_cast<int64_t>(ulg->NumVertices(0))};
  return HeteroGraphPtr(
      new HeteroGraph(meta_graph_, rel_graph, std::move(num_nodes_per_type)));
}

}  // namespace dgl
