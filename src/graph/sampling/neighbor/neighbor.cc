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
    if (num_nodes == 0) {
      // No node provided in the type, create a placeholder relation graph
      subrels[etype] = UnitGraph::Empty(
        hg->GetRelationGraph(etype)->NumVertexTypes(),
        hg->NumVertices(src_vtype),
        hg->NumVertices(dst_vtype),
        hg->DataType(), hg->Context());
      induced_edges[etype] = IdArray::Empty({0}, hg->DataType(), hg->Context());
    } else {
      // sample from one relation graph
      auto req_fmt = (dir == EdgeDir::kOut)? SparseFormat::CSR : SparseFormat::CSC;
      auto avail_fmt = hg->SelectFormat(etype, req_fmt);
      COOMatrix sampled_coo;
      switch (avail_fmt) {
        case SparseFormat::COO:
          if (dir == EdgeDir::kIn) {
            sampled_coo = aten::COOTranspose(aten::COORowWiseSampling(
              aten::COOTranspose(hg->GetCOOMatrix(etype)),
              nodes_ntype, fanouts[etype], prob[etype], replace));
          } else {
            sampled_coo = aten::COORowWiseSampling(
              hg->GetCOOMatrix(etype), nodes_ntype, fanouts[etype], prob[etype], replace);
          }
          break;
        case SparseFormat::CSR:
          CHECK(dir == EdgeDir::kOut) << "Cannot sample out edges on CSC matrix.";
          sampled_coo = aten::CSRRowWiseSampling(
            hg->GetCSRMatrix(etype), nodes_ntype, fanouts[etype], prob[etype], replace);
          break;
        case SparseFormat::CSC:
          CHECK(dir == EdgeDir::kIn) << "Cannot sample in edges on CSR matrix.";
          sampled_coo = aten::CSRRowWiseSampling(
            hg->GetCSCMatrix(etype), nodes_ntype, fanouts[etype], prob[etype], replace);
          sampled_coo = aten::COOTranspose(sampled_coo);
          break;
        default:
          LOG(FATAL) << "Unsupported sparse format.";
      }
      subrels[etype] = UnitGraph::CreateFromCOO(
        hg->GetRelationGraph(etype)->NumVertexTypes(), sampled_coo);
      if (sampled_coo.data.defined()) {
        induced_edges[etype] = sampled_coo.data;
      } else {
        induced_edges[etype] = IdArray::Empty({0}, hg->DataType(), hg->Context());
      }
    }
  }

  HeteroSubgraph ret;
  ret.graph = CreateHeteroGraph(hg->meta_graph(), subrels);
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
    if (num_nodes == 0) {
      // No node provided in the type, create a placeholder relation graph
      subrels[etype] = UnitGraph::Empty(
        hg->GetRelationGraph(etype)->NumVertexTypes(),
        hg->NumVertices(src_vtype),
        hg->NumVertices(dst_vtype),
        hg->DataType(), hg->Context());
      induced_edges[etype] = IdArray::Empty({0}, hg->DataType(), hg->Context());
    } else {
      // sample from one relation graph
      auto req_fmt = (dir == EdgeDir::kOut)? SparseFormat::CSR : SparseFormat::CSC;
      auto avail_fmt = hg->SelectFormat(etype, req_fmt);
      COOMatrix sampled_coo;
      switch (avail_fmt) {
        case SparseFormat::COO:
          if (dir == EdgeDir::kIn) {
            sampled_coo = aten::COOTranspose(aten::COORowWiseTopk(
              aten::COOTranspose(hg->GetCOOMatrix(etype)),
              nodes_ntype, k[etype], weight[etype], ascending));
          } else {
            sampled_coo = aten::COORowWiseTopk(
              hg->GetCOOMatrix(etype), nodes_ntype, k[etype], weight[etype], ascending);
          }
          break;
        case SparseFormat::CSR:
          CHECK(dir == EdgeDir::kOut) << "Cannot sample out edges on CSC matrix.";
          sampled_coo = aten::CSRRowWiseTopk(
            hg->GetCSRMatrix(etype), nodes_ntype, k[etype], weight[etype], ascending);
          break;
        case SparseFormat::CSC:
          CHECK(dir == EdgeDir::kIn) << "Cannot sample in edges on CSR matrix.";
          sampled_coo = aten::CSRRowWiseTopk(
            hg->GetCSCMatrix(etype), nodes_ntype, k[etype], weight[etype], ascending);
          sampled_coo = aten::COOTranspose(sampled_coo);
          break;
        default:
          LOG(FATAL) << "Unsupported sparse format.";
      }
      subrels[etype] = UnitGraph::CreateFromCOO(
        hg->GetRelationGraph(etype)->NumVertexTypes(), sampled_coo);
      if (sampled_coo.data.defined()) {
        induced_edges[etype] = sampled_coo.data;
      } else {
        induced_edges[etype] = IdArray::Empty({0}, hg->DataType(), hg->Context());
      }
    }
  }

  HeteroSubgraph ret;
  ret.graph = CreateHeteroGraph(hg->meta_graph(), subrels);
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = std::move(induced_edges);
  return ret;
}

DGL_REGISTER_GLOBAL("sampling.neighbor._CAPI_DGLSampleNeighbors")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    const auto& nodes = ListValueToVector<IdArray>(args[1]);
    const auto& fanouts = ListValueToVector<int64_t>(args[2]);
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
    const auto& k = ListValueToVector<int64_t>(args[2]);
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

}  // namespace sampling
}  // namespace dgl
