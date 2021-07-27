/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/sampling/neighbor.cc
 * \brief Definition of neighborhood-based sampler APIs.
 */

#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>
#include <dgl/array.h>
#include <dgl/sampling/neighbor.h>
#include "../../../c_api_common.h"
#include "../../unit_graph.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {
namespace sampling {

HeteroSubgraph SampleNeighbors(
    const HeteroGraphPtr hg,
    const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& fanouts,
    EdgeDir dir,
    const std::vector<FloatArray>& prob,
    bool replace) {

  // sanity check
  CHECK_EQ(nodes.size(), hg->NumVertexTypes())
    << "Number of node ID tensors must match the number of node types.";
  CHECK_EQ(fanouts.size(), hg->NumEdgeTypes())
    << "Number of fanout values must match the number of edge types.";
  CHECK_EQ(prob.size(), hg->NumEdgeTypes())
    << "Number of probability tensors must match the number of edge types.";

  std::vector<HeteroGraphPtr> subrels(hg->NumEdgeTypes());
  std::vector<IdArray> induced_edges(hg->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const IdArray nodes_ntype = nodes[(dir == EdgeDir::kOut)? src_vtype : dst_vtype];
    const int64_t num_nodes = nodes_ntype->shape[0];
    if (num_nodes == 0 || fanouts[etype] == 0) {
      // Nothing to sample for this etype, create a placeholder relation graph
      subrels[etype] = UnitGraph::Empty(
        hg->GetRelationGraph(etype)->NumVertexTypes(),
        hg->NumVertices(src_vtype),
        hg->NumVertices(dst_vtype),
        hg->DataType(), hg->Context());
      induced_edges[etype] = aten::NullArray();
    } else if (fanouts[etype] == -1) {
      const auto &earr = (dir == EdgeDir::kOut) ?
        hg->OutEdges(etype, nodes_ntype) :
        hg->InEdges(etype, nodes_ntype);
      subrels[etype] = UnitGraph::CreateFromCOO(
        hg->GetRelationGraph(etype)->NumVertexTypes(),
        hg->NumVertices(src_vtype),
        hg->NumVertices(dst_vtype),
        earr.src,
        earr.dst);
      induced_edges[etype] = earr.id;
    } else {
      // sample from one relation graph
      auto req_fmt = (dir == EdgeDir::kOut)? CSR_CODE : CSC_CODE;
      auto avail_fmt = hg->SelectFormat(etype, req_fmt);
      COOMatrix sampled_coo;
      switch (avail_fmt) {
        case SparseFormat::kCOO:
          if (dir == EdgeDir::kIn) {
            sampled_coo = aten::COOTranspose(aten::COORowWiseSampling(
              aten::COOTranspose(hg->GetCOOMatrix(etype)),
              nodes_ntype, fanouts[etype], prob[etype], replace));
          } else {
            sampled_coo = aten::COORowWiseSampling(
              hg->GetCOOMatrix(etype), nodes_ntype, fanouts[etype], prob[etype], replace);
          }
          break;
        case SparseFormat::kCSR:
          CHECK(dir == EdgeDir::kOut) << "Cannot sample out edges on CSC matrix.";
          sampled_coo = aten::CSRRowWiseSampling(
            hg->GetCSRMatrix(etype), nodes_ntype, fanouts[etype], prob[etype], replace);
          break;
        case SparseFormat::kCSC:
          CHECK(dir == EdgeDir::kIn) << "Cannot sample in edges on CSR matrix.";
          sampled_coo = aten::CSRRowWiseSampling(
            hg->GetCSCMatrix(etype), nodes_ntype, fanouts[etype], prob[etype], replace);
          sampled_coo = aten::COOTranspose(sampled_coo);
          break;
        default:
          LOG(FATAL) << "Unsupported sparse format.";
      }
      subrels[etype] = UnitGraph::CreateFromCOO(
        hg->GetRelationGraph(etype)->NumVertexTypes(), sampled_coo.num_rows, sampled_coo.num_cols,
        sampled_coo.row, sampled_coo.col);
      induced_edges[etype] = sampled_coo.data;
    }
  }

  HeteroSubgraph ret;
  ret.graph = CreateHeteroGraph(hg->meta_graph(), subrels, hg->NumVerticesPerType());
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = std::move(induced_edges);
  return ret;
}

HeteroSubgraph SampleNeighborsEType(
    const HeteroGraphPtr hg,
    const IdArray nodes,
    const IdArray etypes,
    const int64_t fanout,
    EdgeDir dir,
    const IdArray prob,
    bool replace) {

  CHECK_EQ(1, hg->NumVertexTypes())
    << "SampleNeighborsEType only work with homogeneous graph";
  CHECK_EQ(1, hg->NumEdgeTypes())
    << "SampleNeighborsEType only work with homogeneous graph";

  std::vector<HeteroGraphPtr> subrels(1);
  std::vector<IdArray> induced_edges(1);
  const int64_t num_nodes = nodes->shape[0];
  dgl_type_t etype = 0;
  const dgl_type_t src_vtype = 0;
  const dgl_type_t dst_vtype = 0;
  if (num_nodes == 0 || fanout == 0) {
    subrels[etype] = UnitGraph::Empty(1,
      hg->NumVertices(src_vtype),
      hg->NumVertices(dst_vtype),
      hg->DataType(), hg->Context());
    induced_edges[etype] = aten::NullArray();
  } else if (fanout == -1) {
    const auto &earr = (dir == EdgeDir::kOut) ?
      hg->OutEdges(etype, nodes) :
      hg->InEdges(etype, nodes);
    subrels[etype] = UnitGraph::CreateFromCOO(
      1,
      hg->NumVertices(src_vtype),
      hg->NumVertices(dst_vtype),
      earr.src,
      earr.dst);
      induced_edges[etype] = earr.id;
  } else {
    // sample from graph
    // the edge type is stored in etypes
    auto req_fmt = (dir == EdgeDir::kOut)? CSR_CODE : CSC_CODE;
    auto avail_fmt = hg->SelectFormat(etype, req_fmt);
    COOMatrix sampled_coo;
    switch (avail_fmt) {
      case SparseFormat::kCOO:
        if (dir == EdgeDir::kIn) {
          sampled_coo = aten::COOTranspose(aten::COORowWisePerEtypeSampling(
            aten::COOTranspose(hg->GetCOOMatrix(etype)),
            nodes, etypes, fanout, prob, replace));
        } else {
          sampled_coo = aten::COORowWisePerEtypeSampling(
            hg->GetCOOMatrix(etype), nodes, etypes, fanout, prob, replace);
        }
        break;
      case SparseFormat::kCSR:
        CHECK(dir == EdgeDir::kOut) << "Cannot sample out edges on CSC matrix.";
        sampled_coo = aten::CSRRowWisePerEtypeSampling(
            hg->GetCSRMatrix(etype), nodes, etypes, fanout, prob, replace);
          break;
      case SparseFormat::kCSC:
        CHECK(dir == EdgeDir::kIn) << "Cannot sample in edges on CSR matrix.";
        sampled_coo = aten::CSRRowWisePerEtypeSampling(
            hg->GetCSCMatrix(etype), nodes, etypes, fanout, prob, replace);
        sampled_coo = aten::COOTranspose(sampled_coo);
        break;
      default:
        LOG(FATAL) << "Unsupported sparse format.";
    }

    subrels[etype] = UnitGraph::CreateFromCOO(
      1, sampled_coo.num_rows, sampled_coo.num_cols,
      sampled_coo.row, sampled_coo.col);
    induced_edges[etype] = sampled_coo.data;
  }

  HeteroSubgraph ret;
  ret.graph = CreateHeteroGraph(hg->meta_graph(), subrels, hg->NumVerticesPerType());
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = std::move(induced_edges);
  return ret;
}

HeteroSubgraph SampleNeighborsTopk(
    const HeteroGraphPtr hg,
    const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& k,
    EdgeDir dir,
    const std::vector<FloatArray>& weight,
    bool ascending) {
  // sanity check
  CHECK_EQ(nodes.size(), hg->NumVertexTypes())
    << "Number of node ID tensors must match the number of node types.";
  CHECK_EQ(k.size(), hg->NumEdgeTypes())
    << "Number of k values must match the number of edge types.";
  CHECK_EQ(weight.size(), hg->NumEdgeTypes())
    << "Number of weight tensors must match the number of edge types.";

  std::vector<HeteroGraphPtr> subrels(hg->NumEdgeTypes());
  std::vector<IdArray> induced_edges(hg->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const IdArray nodes_ntype = nodes[(dir == EdgeDir::kOut)? src_vtype : dst_vtype];
    const int64_t num_nodes = nodes_ntype->shape[0];
    if (num_nodes == 0 || k[etype] == 0) {
      // Nothing to sample for this etype, create a placeholder relation graph
      subrels[etype] = UnitGraph::Empty(
        hg->GetRelationGraph(etype)->NumVertexTypes(),
        hg->NumVertices(src_vtype),
        hg->NumVertices(dst_vtype),
        hg->DataType(), hg->Context());
      induced_edges[etype] = aten::NullArray();
    } else if (k[etype] == -1) {
      const auto &earr = (dir == EdgeDir::kOut) ?
        hg->OutEdges(etype, nodes_ntype) :
        hg->InEdges(etype, nodes_ntype);
      subrels[etype] = UnitGraph::CreateFromCOO(
        hg->GetRelationGraph(etype)->NumVertexTypes(),
        hg->NumVertices(src_vtype),
        hg->NumVertices(dst_vtype),
        earr.src,
        earr.dst);
      induced_edges[etype] = earr.id;
    } else {
      // sample from one relation graph
      auto req_fmt = (dir == EdgeDir::kOut)? CSR_CODE : CSC_CODE;
      auto avail_fmt = hg->SelectFormat(etype, req_fmt);
      COOMatrix sampled_coo;
      switch (avail_fmt) {
        case SparseFormat::kCOO:
          if (dir == EdgeDir::kIn) {
            sampled_coo = aten::COOTranspose(aten::COORowWiseTopk(
              aten::COOTranspose(hg->GetCOOMatrix(etype)),
              nodes_ntype, k[etype], weight[etype], ascending));
          } else {
            sampled_coo = aten::COORowWiseTopk(
              hg->GetCOOMatrix(etype), nodes_ntype, k[etype], weight[etype], ascending);
          }
          break;
        case SparseFormat::kCSR:
          CHECK(dir == EdgeDir::kOut) << "Cannot sample out edges on CSC matrix.";
          sampled_coo = aten::CSRRowWiseTopk(
            hg->GetCSRMatrix(etype), nodes_ntype, k[etype], weight[etype], ascending);
          break;
        case SparseFormat::kCSC:
          CHECK(dir == EdgeDir::kIn) << "Cannot sample in edges on CSR matrix.";
          sampled_coo = aten::CSRRowWiseTopk(
            hg->GetCSCMatrix(etype), nodes_ntype, k[etype], weight[etype], ascending);
          sampled_coo = aten::COOTranspose(sampled_coo);
          break;
        default:
          LOG(FATAL) << "Unsupported sparse format.";
      }
      subrels[etype] = UnitGraph::CreateFromCOO(
        hg->GetRelationGraph(etype)->NumVertexTypes(), sampled_coo.num_rows, sampled_coo.num_cols,
        sampled_coo.row, sampled_coo.col);
      induced_edges[etype] = sampled_coo.data;
    }
  }

  HeteroSubgraph ret;
  ret.graph = CreateHeteroGraph(hg->meta_graph(), subrels, hg->NumVerticesPerType());
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = std::move(induced_edges);
  return ret;
}

HeteroSubgraph SampleNeighborsBiased(
    const HeteroGraphPtr hg,
    const IdArray& nodes,
    const int64_t fanout,
    const NDArray& bias,
    const NDArray& tag_offset,
    const EdgeDir dir,
    const bool replace
) {
  CHECK_EQ(hg->NumEdgeTypes(), 1) << "Only homogeneous or bipartite graphs are supported";
  auto pair = hg->meta_graph()->FindEdge(0);
  const dgl_type_t src_vtype = pair.first;
  const dgl_type_t dst_vtype = pair.second;
  const dgl_type_t nodes_ntype = (dir == EdgeDir::kOut) ? src_vtype : dst_vtype;

  // sanity check
  CHECK_EQ(tag_offset->ndim, 2) << "The shape of tag_offset should be [num_nodes, num_tags + 1]";
  CHECK_EQ(tag_offset->shape[0], hg->NumVertices(nodes_ntype))
    << "The shape of tag_offset should be [num_nodes, num_tags + 1]";
  CHECK_EQ(tag_offset->shape[1], bias->shape[0] + 1)
    << "The sizes of tag_offset and bias are inconsistent";

  const int64_t num_nodes = nodes->shape[0];
  HeteroGraphPtr subrel;
  IdArray induced_edges;
  const dgl_type_t etype = 0;
  if (num_nodes == 0 || fanout == 0) {
      // Nothing to sample for this etype, create a placeholder relation graph
      subrel = UnitGraph::Empty(
        hg->GetRelationGraph(etype)->NumVertexTypes(),
        hg->NumVertices(src_vtype),
        hg->NumVertices(dst_vtype),
        hg->DataType(), hg->Context());
      induced_edges = aten::NullArray();
    } else if (fanout == -1) {
      const auto &earr = (dir == EdgeDir::kOut) ?
        hg->OutEdges(etype, nodes_ntype) :
        hg->InEdges(etype, nodes_ntype);
      subrel = UnitGraph::CreateFromCOO(
        hg->GetRelationGraph(etype)->NumVertexTypes(),
        hg->NumVertices(src_vtype),
        hg->NumVertices(dst_vtype),
        earr.src,
        earr.dst);
      induced_edges = earr.id;
    } else {
      // sample from one relation graph
      const auto req_fmt = (dir == EdgeDir::kOut)? CSR_CODE : CSC_CODE;
      const auto created_fmt = hg->GetCreatedFormats();
      COOMatrix sampled_coo;

      switch (req_fmt) {
        case CSR_CODE:
          CHECK(created_fmt & CSR_CODE) << "A sorted CSR Matrix is required.";
          sampled_coo = aten::CSRRowWiseSamplingBiased(
            hg->GetCSRMatrix(etype), nodes, fanout, tag_offset, bias, replace);
          break;
        case CSC_CODE:
          CHECK(created_fmt & CSC_CODE) << "A sorted CSC Matrix is required.";
          sampled_coo = aten::CSRRowWiseSamplingBiased(
            hg->GetCSCMatrix(etype), nodes, fanout, tag_offset, bias, replace);
          sampled_coo = aten::COOTranspose(sampled_coo);
          break;
        default:
          LOG(FATAL) << "Unsupported sparse format.";
      }
      subrel = UnitGraph::CreateFromCOO(
        hg->GetRelationGraph(etype)->NumVertexTypes(), sampled_coo.num_rows, sampled_coo.num_cols,
        sampled_coo.row, sampled_coo.col);
      induced_edges = sampled_coo.data;
    }

  HeteroSubgraph ret;
  ret.graph = CreateHeteroGraph(hg->meta_graph(), {subrel}, hg->NumVerticesPerType());
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = {induced_edges};
  return ret;
}

DGL_REGISTER_GLOBAL("sampling.neighbor._CAPI_DGLSampleNeighborsEType")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    IdArray nodes = args[1];
    IdArray etypes = args[2];
    const int64_t fanout = args[3];
    const std::string dir_str = args[4];
    IdArray prob = args[5];
    const bool replace = args[6];

    CHECK(dir_str == "in" || dir_str == "out")
      << "Invalid edge direction. Must be \"in\" or \"out\".";
    EdgeDir dir = (dir_str == "in")? EdgeDir::kIn : EdgeDir::kOut;

    std::shared_ptr<HeteroSubgraph> subg(new HeteroSubgraph);
    *subg = sampling::SampleNeighborsEType(
        hg.sptr(), nodes, etypes, fanout, dir, prob, replace);

    *rv = HeteroSubgraphRef(subg);
  });

DGL_REGISTER_GLOBAL("sampling.neighbor._CAPI_DGLSampleNeighbors")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    const auto& nodes = ListValueToVector<IdArray>(args[1]);
    IdArray fanouts_array = args[2];
    const auto& fanouts = fanouts_array.ToVector<int64_t>();
    const std::string dir_str = args[3];
    const auto& prob = ListValueToVector<FloatArray>(args[4]);
    const bool replace = args[5];

    CHECK(dir_str == "in" || dir_str == "out")
      << "Invalid edge direction. Must be \"in\" or \"out\".";
    EdgeDir dir = (dir_str == "in")? EdgeDir::kIn : EdgeDir::kOut;

    std::shared_ptr<HeteroSubgraph> subg(new HeteroSubgraph);
    *subg = sampling::SampleNeighbors(
        hg.sptr(), nodes, fanouts, dir, prob, replace);

    *rv = HeteroSubgraphRef(subg);
  });

DGL_REGISTER_GLOBAL("sampling.neighbor._CAPI_DGLSampleNeighborsTopk")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    const auto& nodes = ListValueToVector<IdArray>(args[1]);
    IdArray k_array = args[2];
    const auto& k = k_array.ToVector<int64_t>();
    const std::string dir_str = args[3];
    const auto& weight = ListValueToVector<FloatArray>(args[4]);
    const bool ascending = args[5];

    CHECK(dir_str == "in" || dir_str == "out")
      << "Invalid edge direction. Must be \"in\" or \"out\".";
      EdgeDir dir = (dir_str == "in")? EdgeDir::kIn : EdgeDir::kOut;

    std::shared_ptr<HeteroSubgraph> subg(new HeteroSubgraph);
    *subg = sampling::SampleNeighborsTopk(
        hg.sptr(), nodes, k, dir, weight, ascending);

    *rv = HeteroGraphRef(subg);
  });

DGL_REGISTER_GLOBAL("sampling.neighbor._CAPI_DGLSampleNeighborsBiased")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    const IdArray nodes = args[1];
    const int64_t fanout = args[2];
    const NDArray bias = args[3];
    const NDArray tag_offset = args[4];
    const std::string dir_str = args[5];
    const bool replace = args[6];

    CHECK(dir_str == "in" || dir_str == "out")
      << "Invalid edge direction. Must be \"in\" or \"out\".";
      EdgeDir dir = (dir_str == "in")? EdgeDir::kIn : EdgeDir::kOut;

    std::shared_ptr<HeteroSubgraph> subg(new HeteroSubgraph);
    *subg = sampling::SampleNeighborsBiased(
        hg.sptr(), nodes, fanout, bias, tag_offset, dir, replace);

    *rv = HeteroGraphRef(subg);
  });

}  // namespace sampling
}  // namespace dgl
