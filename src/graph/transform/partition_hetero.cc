/**
 *  Copyright (c) 2020 by Contributors
 * @file graph/metis_partition.cc
 * @brief Call Metis partitioning
 */

#include <dgl/base_heterograph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/parallel_for.h>

#include "../heterograph.h"
#include "../unit_graph.h"

#if !defined(_WIN32)
#include <GKlib.h>
#endif  // !defined(_WIN32)

using namespace dgl::runtime;

namespace dgl {

#if !defined(_WIN32)
gk_csr_t *Convert2GKCsr(const aten::CSRMatrix mat, bool is_row);
aten::CSRMatrix Convert2DGLCsr(gk_csr_t *gk_csr, bool is_row);
#endif  // !defined(_WIN32)

namespace transform {

class HaloHeteroSubgraph : public HeteroSubgraph {
 public:
  std::vector<IdArray> inner_nodes;
};

HeteroGraphPtr ReorderUnitGraph(UnitGraphPtr ug, IdArray new_order) {
  auto format = ug->GetCreatedFormats();
  // We only need to reorder one of the graph structure.
  if (format & CSC_CODE) {
    auto cscmat = ug->GetCSCMatrix(0);
    auto new_cscmat = aten::CSRReorder(cscmat, new_order, new_order);
    return UnitGraph::CreateFromCSC(
        ug->NumVertexTypes(), new_cscmat, ug->GetAllowedFormats());
  } else if (format & CSR_CODE) {
    auto csrmat = ug->GetCSRMatrix(0);
    auto new_csrmat = aten::CSRReorder(csrmat, new_order, new_order);
    return UnitGraph::CreateFromCSR(
        ug->NumVertexTypes(), new_csrmat, ug->GetAllowedFormats());
  } else {
    auto coomat = ug->GetCOOMatrix(0);
    auto new_coomat = aten::COOReorder(coomat, new_order, new_order);
    return UnitGraph::CreateFromCOO(
        ug->NumVertexTypes(), new_coomat, ug->GetAllowedFormats());
  }
}

HaloHeteroSubgraph GetSubgraphWithHalo(
    std::shared_ptr<HeteroGraph> hg, IdArray nodes, int num_hops) {
  CHECK_EQ(hg->NumBits(), 64) << "halo subgraph only supports 64bits graph";
  CHECK_EQ(hg->relation_graphs().size(), 1)
      << "halo subgraph only supports homogeneous graph";
  CHECK_EQ(nodes->dtype.bits, 64)
      << "halo subgraph only supports 64bits nodes tensor";
  const dgl_id_t *nid = static_cast<dgl_id_t *>(nodes->data);
  const auto id_len = nodes->shape[0];
  // A map contains all nodes in the subgraph.
  // The key is the old node Ids, the value indicates whether a node is a inner
  // node.
  std::unordered_map<dgl_id_t, bool> all_nodes;
  // The old Ids of all nodes. We want to preserve the order of the nodes in the
  // vector. The first few nodes are the inner nodes in the subgraph.
  std::vector<dgl_id_t> old_node_ids(nid, nid + id_len);
  std::vector<std::vector<dgl_id_t>> outer_nodes(num_hops);
  for (int64_t i = 0; i < id_len; i++) all_nodes[nid[i]] = true;
  auto orig_nodes = all_nodes;

  std::vector<dgl_id_t> edge_src, edge_dst, edge_eid;

  // When we deal with in-edges, we need to do two things:
  // * find the edges inside the partition and the edges between partitions.
  // * find the nodes outside the partition that connect the partition.
  EdgeArray in_edges = hg->InEdges(0, nodes);
  auto src = in_edges.src;
  auto dst = in_edges.dst;
  auto eid = in_edges.id;
  auto num_edges = eid->shape[0];
  const dgl_id_t *src_data = static_cast<dgl_id_t *>(src->data);
  const dgl_id_t *dst_data = static_cast<dgl_id_t *>(dst->data);
  const dgl_id_t *eid_data = static_cast<dgl_id_t *>(eid->data);
  for (int64_t i = 0; i < num_edges; i++) {
    // We check if the source node is in the original node.
    auto it1 = orig_nodes.find(src_data[i]);
    if (it1 != orig_nodes.end() || num_hops > 0) {
      edge_src.push_back(src_data[i]);
      edge_dst.push_back(dst_data[i]);
      edge_eid.push_back(eid_data[i]);
    }
    // We need to expand only if the node hasn't been seen before.
    auto it = all_nodes.find(src_data[i]);
    if (it == all_nodes.end() && num_hops > 0) {
      all_nodes[src_data[i]] = false;
      old_node_ids.push_back(src_data[i]);
      outer_nodes[0].push_back(src_data[i]);
    }
  }

  // Now we need to traverse the graph with the in-edges to access nodes
  // and edges more hops away.
  for (int k = 1; k < num_hops; k++) {
    const std::vector<dgl_id_t> &nodes = outer_nodes[k - 1];
    EdgeArray in_edges = hg->InEdges(0, aten::VecToIdArray(nodes));
    auto src = in_edges.src;
    auto dst = in_edges.dst;
    auto eid = in_edges.id;
    auto num_edges = eid->shape[0];
    const dgl_id_t *src_data = static_cast<dgl_id_t *>(src->data);
    const dgl_id_t *dst_data = static_cast<dgl_id_t *>(dst->data);
    const dgl_id_t *eid_data = static_cast<dgl_id_t *>(eid->data);
    for (int64_t i = 0; i < num_edges; i++) {
      auto it1 = orig_nodes.find(src_data[i]);
      // If the source node is in the partition, we have got this edge when we
      // iterate over the out-edges above.
      if (it1 == orig_nodes.end()) {
        edge_src.push_back(src_data[i]);
        edge_dst.push_back(dst_data[i]);
        edge_eid.push_back(eid_data[i]);
      }
      // If we haven't seen this node.
      auto it = all_nodes.find(src_data[i]);
      if (it == all_nodes.end()) {
        all_nodes[src_data[i]] = false;
        old_node_ids.push_back(src_data[i]);
        outer_nodes[k].push_back(src_data[i]);
      }
    }
  }

  if (num_hops > 0) {
    EdgeArray out_edges = hg->OutEdges(0, nodes);
    auto src = out_edges.src;
    auto dst = out_edges.dst;
    auto eid = out_edges.id;
    auto num_edges = eid->shape[0];
    const dgl_id_t *src_data = static_cast<dgl_id_t *>(src->data);
    const dgl_id_t *dst_data = static_cast<dgl_id_t *>(dst->data);
    const dgl_id_t *eid_data = static_cast<dgl_id_t *>(eid->data);
    for (int64_t i = 0; i < num_edges; i++) {
      // If the outer edge isn't in the partition.
      auto it1 = orig_nodes.find(dst_data[i]);
      if (it1 == orig_nodes.end()) {
        edge_src.push_back(src_data[i]);
        edge_dst.push_back(dst_data[i]);
        edge_eid.push_back(eid_data[i]);
      }
      // We don't expand along the out-edges.
      auto it = all_nodes.find(dst_data[i]);
      if (it == all_nodes.end()) {
        all_nodes[dst_data[i]] = false;
        old_node_ids.push_back(dst_data[i]);
      }
    }
  }

  // We assign new Ids to the nodes in the subgraph. We ensure that the HALO
  // nodes are behind the input nodes.
  std::unordered_map<dgl_id_t, dgl_id_t> old2new;
  for (size_t i = 0; i < old_node_ids.size(); i++) {
    old2new[old_node_ids[i]] = i;
  }

  num_edges = edge_src.size();
  IdArray new_src = IdArray::Empty(
      {num_edges}, DGLDataType{kDGLInt, 64, 1}, DGLContext{kDGLCPU, 0});
  IdArray new_dst = IdArray::Empty(
      {num_edges}, DGLDataType{kDGLInt, 64, 1}, DGLContext{kDGLCPU, 0});
  dgl_id_t *new_src_data = static_cast<dgl_id_t *>(new_src->data);
  dgl_id_t *new_dst_data = static_cast<dgl_id_t *>(new_dst->data);
  for (size_t i = 0; i < edge_src.size(); i++) {
    new_src_data[i] = old2new[edge_src[i]];
    new_dst_data[i] = old2new[edge_dst[i]];
  }

  std::vector<int> inner_nodes(old_node_ids.size());
  for (size_t i = 0; i < old_node_ids.size(); i++) {
    dgl_id_t old_nid = old_node_ids[i];
    inner_nodes[i] = all_nodes[old_nid];
  }
  aten::COOMatrix coo(
      old_node_ids.size(), old_node_ids.size(), new_src, new_dst);
  HeteroGraphPtr ugptr = UnitGraph::CreateFromCOO(1, coo);
  HeteroGraphPtr subg = CreateHeteroGraph(hg->meta_graph(), {ugptr});
  HaloHeteroSubgraph halo_subg;
  halo_subg.graph = subg;
  halo_subg.induced_vertices = {aten::VecToIdArray(old_node_ids)};
  halo_subg.induced_edges = {aten::VecToIdArray(edge_eid)};
  // TODO(zhengda) we need to switch to 8 bytes afterwards.
  halo_subg.inner_nodes = {aten::VecToIdArray<int>(inner_nodes, 32)};
  return halo_subg;
}

DGL_REGISTER_GLOBAL("partition._CAPI_DGLReorderGraph_Hetero")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphRef g = args[0];
      auto hgptr = std::dynamic_pointer_cast<HeteroGraph>(g.sptr());
      CHECK(hgptr) << "Invalid HeteroGraph object";
      CHECK_EQ(hgptr->relation_graphs().size(), 1)
          << "Reorder only supports HomoGraph";
      auto ugptr = hgptr->relation_graphs()[0];
      const IdArray new_order = args[1];
      auto reorder_ugptr = ReorderUnitGraph(ugptr, new_order);
      std::vector<HeteroGraphPtr> rel_graphs = {reorder_ugptr};
      *rv = HeteroGraphRef(std::make_shared<HeteroGraph>(
          hgptr->meta_graph(), rel_graphs, hgptr->NumVerticesPerType()));
    });

DGL_REGISTER_GLOBAL("partition._CAPI_DGLPartitionWithHalo_Hetero")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphRef g = args[0];
      auto hgptr = std::dynamic_pointer_cast<HeteroGraph>(g.sptr());
      CHECK(hgptr) << "Invalid HeteroGraph object";
      CHECK_EQ(hgptr->relation_graphs().size(), 1)
          << "Metis partition only supports HomoGraph";
      auto ugptr = hgptr->relation_graphs()[0];

      IdArray node_parts = args[1];
      int num_hops = args[2];

      CHECK_EQ(node_parts->dtype.bits, 64)
          << "Only supports 64bits tensor for now";

      const int64_t *part_data = static_cast<int64_t *>(node_parts->data);
      int64_t num_nodes = node_parts->shape[0];
      std::unordered_map<int, std::vector<int64_t>> part_map;
      for (int64_t i = 0; i < num_nodes; i++) {
        dgl_id_t part_id = part_data[i];
        auto it = part_map.find(part_id);
        if (it == part_map.end()) {
          std::vector<int64_t> vec;
          vec.push_back(i);
          part_map[part_id] = vec;
        } else {
          it->second.push_back(i);
        }
      }
      std::vector<int> part_ids;
      std::vector<std::vector<int64_t>> part_nodes;
      int max_part_id = 0;
      for (auto it = part_map.begin(); it != part_map.end(); it++) {
        max_part_id = std::max(it->first, max_part_id);
        part_ids.push_back(it->first);
        part_nodes.push_back(it->second);
      }
      // When we construct subgraphs, we need to access both in-edges and
      // out-edges. We need to make sure the in-CSR and out-CSR exist.
      // Otherwise, we'll try to construct in-CSR and out-CSR in openmp for
      // loop, which will lead to some unexpected results.
      ugptr->GetInCSR();
      ugptr->GetOutCSR();
      std::vector<std::shared_ptr<HaloHeteroSubgraph>> subgs(max_part_id + 1);
      int num_partitions = part_nodes.size();
      runtime::parallel_for(0, num_partitions, [&](int b, int e) {
        for (auto i = b; i < e; i++) {
          auto nodes = aten::VecToIdArray(part_nodes[i]);
          HaloHeteroSubgraph subg = GetSubgraphWithHalo(hgptr, nodes, num_hops);
          std::shared_ptr<HaloHeteroSubgraph> subg_ptr(
              new HaloHeteroSubgraph(subg));
          int part_id = part_ids[i];
          subgs[part_id] = subg_ptr;
        }
      });
      List<HeteroSubgraphRef> ret_list;
      for (size_t i = 0; i < subgs.size(); i++) {
        ret_list.push_back(HeteroSubgraphRef(subgs[i]));
      }
      *rv = ret_list;
    });

template <class IdType>
struct EdgeProperty {
  IdType eid;
  int64_t idx;
  int part_id;
};

// Reassign edge IDs so that all edges in a partition have contiguous edge IDs.
// The original edge IDs are returned.
DGL_REGISTER_GLOBAL("partition._CAPI_DGLReassignEdges_Hetero")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphRef g = args[0];
      auto hgptr = std::dynamic_pointer_cast<HeteroGraph>(g.sptr());
      CHECK(hgptr) << "Invalid HeteroGraph object";
      CHECK_EQ(hgptr->relation_graphs().size(), 1)
          << "Reorder only supports HomoGraph";
      auto ugptr = hgptr->relation_graphs()[0];
      IdArray etype = args[1];
      IdArray part_id = args[2];
      bool is_incsr = args[3];
      auto csrmat = is_incsr ? ugptr->GetCSCMatrix(0) : ugptr->GetCSRMatrix(0);
      int64_t num_edges = csrmat.data->shape[0];
      int64_t num_rows = csrmat.indptr->shape[0] - 1;
      IdArray new_data =
          IdArray::Empty({num_edges}, csrmat.data->dtype, csrmat.data->ctx);
      // Return the original edge Ids.
      *rv = new_data;

      // Generate new edge Ids.
      ATEN_ID_TYPE_SWITCH(new_data->dtype, IdType, {
        CHECK(etype->dtype.bits == sizeof(IdType) * 8);
        CHECK(part_id->dtype.bits == sizeof(IdType) * 8);
        const IdType *part_id_data = static_cast<IdType *>(part_id->data);
        const IdType *etype_data = static_cast<IdType *>(etype->data);
        const IdType *indptr_data = static_cast<IdType *>(csrmat.indptr->data);
        IdType *typed_data = static_cast<IdType *>(csrmat.data->data);
        IdType *typed_new_data = static_cast<IdType *>(new_data->data);
        std::vector<EdgeProperty<IdType>> indexed_eids(num_edges);
        for (int64_t i = 0; i < num_rows; i++) {
          for (int64_t j = indptr_data[i]; j < indptr_data[i + 1]; j++) {
            indexed_eids[j].eid = typed_data[j];
            indexed_eids[j].idx = j;
            indexed_eids[j].part_id = part_id_data[i];
          }
        }
        auto comp = [etype_data](
                        const EdgeProperty<IdType> &a,
                        const EdgeProperty<IdType> &b) {
          if (a.part_id == b.part_id) {
            return etype_data[a.eid] < etype_data[b.eid];
          } else {
            return a.part_id < b.part_id;
          }
        };
        // We only need to sort the edges if the input graph has multiple
        // relations. If it's a homogeneous grap, we'll just assign edge Ids
        // based on its previous order.
        if (etype->shape[0] > 0) {
          std::sort(indexed_eids.begin(), indexed_eids.end(), comp);
        }
        for (int64_t new_eid = 0; new_eid < num_edges; new_eid++) {
          int64_t orig_idx = indexed_eids[new_eid].idx;
          typed_new_data[new_eid] = typed_data[orig_idx];
          typed_data[orig_idx] = new_eid;
        }
      });
      ugptr->InvalidateCSR();
      ugptr->InvalidateCOO();
    });

DGL_REGISTER_GLOBAL("partition._CAPI_GetHaloSubgraphInnerNodes_Hetero")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroSubgraphRef g = args[0];
      auto gptr = std::dynamic_pointer_cast<HaloHeteroSubgraph>(g.sptr());
      CHECK(gptr) << "The input graph has to be HaloHeteroSubgraph";
      *rv = gptr->inner_nodes[0];
    });

DGL_REGISTER_GLOBAL("partition._CAPI_DGLMakeSymmetric_Hetero")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphRef g = args[0];
      auto hgptr = std::dynamic_pointer_cast<HeteroGraph>(g.sptr());
      CHECK(hgptr) << "Invalid HeteroGraph object";
      CHECK_EQ(hgptr->relation_graphs().size(), 1)
          << "Metis partition only supports homogeneous graph";
      auto ugptr = hgptr->relation_graphs()[0];

#if !defined(_WIN32)
      // TODO(zhengda) should we get whatever CSR exists in the graph.
      gk_csr_t *gk_csr = Convert2GKCsr(ugptr->GetCSCMatrix(0), true);
      gk_csr_t *sym_gk_csr = gk_csr_MakeSymmetric(gk_csr, GK_CSR_SYM_SUM);
      auto mat = Convert2DGLCsr(sym_gk_csr, true);
      gk_csr_Free(&gk_csr);
      gk_csr_Free(&sym_gk_csr);

      auto new_ugptr = UnitGraph::CreateFromCSC(
          ugptr->NumVertexTypes(), mat, ugptr->GetAllowedFormats());
      std::vector<HeteroGraphPtr> rel_graphs = {new_ugptr};
      *rv = HeteroGraphRef(std::make_shared<HeteroGraph>(
          hgptr->meta_graph(), rel_graphs, hgptr->NumVerticesPerType()));
#else
      LOG(FATAL) << "The fast version of making symmetric graph is not "
                    "supported in Windows.";
#endif  // !defined(_WIN32)
    });

}  // namespace transform
}  // namespace dgl
