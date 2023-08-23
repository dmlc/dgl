/**
 *  Copyright (c) 2020-2022 by Contributors
 * @file graph/sampling/neighbor.cc
 * @brief Definition of neighborhood-based sampler APIs.
 */

#include <dgl/array.h>
#include <dgl/aten/macro.h>
#include <dgl/immutable_graph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/parallel_for.h>
#include <dgl/sampling/neighbor.h>

#include <tuple>
#include <utility>

#include "../../../array/cpu/concurrent_id_hash_map.h"
#include "../../../c_api_common.h"
#include "../../unit_graph.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {
namespace sampling {

template <typename IdType>
void ExcludeCertainEdgesFused(
    std::vector<CSRMatrix>* sampled_graphs, std::vector<IdArray>* induced_edges,
    std::vector<IdArray>* sampled_coo_rows,
    const std::vector<IdArray>& exclude_edges,
    std::vector<FloatArray>* weights = nullptr) {
  int etypes = (*sampled_graphs).size();
  std::vector<IdArray> remain_induced_edges(etypes);
  std::vector<IdArray> remain_indptrs(etypes);
  std::vector<IdArray> remain_indices(etypes);
  std::vector<IdArray> remain_coo_rows(etypes);
  std::vector<FloatArray> remain_weights(etypes);
  for (int etype = 0; etype < etypes; ++etype) {
    if (exclude_edges[etype].GetSize() == 0 ||
        (*sampled_graphs)[etype].num_rows == 0) {
      remain_induced_edges[etype] = (*induced_edges)[etype];
      if (weights) remain_weights[etype] = (*weights)[etype];
      continue;
    }
    const auto dtype = weights && (*weights)[etype]->shape[0]
                           ? (*weights)[etype]->dtype
                           : DGLDataType{kDGLFloat, 8 * sizeof(float), 1};
    ATEN_FLOAT_TYPE_SWITCH(dtype, FloatType, "weights", {
      IdType* indptr = (*sampled_graphs)[etype].indptr.Ptr<IdType>();
      IdType* indices = (*sampled_graphs)[etype].indices.Ptr<IdType>();
      IdType* coo_rows = (*sampled_coo_rows)[etype].Ptr<IdType>();
      IdType* induced_edges_data = (*induced_edges)[etype].Ptr<IdType>();
      FloatType* weights_data = weights && (*weights)[etype]->shape[0]
                                    ? (*weights)[etype].Ptr<FloatType>()
                                    : nullptr;
      const IdType exclude_edges_len = exclude_edges[etype]->shape[0];
      std::sort(
          exclude_edges[etype].Ptr<IdType>(),
          exclude_edges[etype].Ptr<IdType>() + exclude_edges_len);
      const IdType* exclude_edges_data = exclude_edges[etype].Ptr<IdType>();
      IdType outIndices = 0;
      for (IdType row = 0; row < (*sampled_graphs)[etype].indptr->shape[0] - 1;
           ++row) {
        auto tmp_row = indptr[row];
        if (outIndices != indptr[row]) indptr[row] = outIndices;
        for (IdType col = tmp_row; col < indptr[row + 1]; ++col) {
          if (!std::binary_search(
                  exclude_edges_data, exclude_edges_data + exclude_edges_len,
                  induced_edges_data[col])) {
            indices[outIndices] = indices[col];
            induced_edges_data[outIndices] = induced_edges_data[col];
            coo_rows[outIndices] = coo_rows[col];
            if (weights_data) weights_data[outIndices] = weights_data[col];
            ++outIndices;
          }
        }
      }
      indptr[(*sampled_graphs)[etype].indptr->shape[0] - 1] = outIndices;
      remain_induced_edges[etype] =
          aten::IndexSelect((*induced_edges)[etype], 0, outIndices);
      remain_weights[etype] =
          weights_data ? aten::IndexSelect((*weights)[etype], 0, outIndices)
                       : NullArray();
      remain_indices[etype] =
          aten::IndexSelect((*sampled_graphs)[etype].indices, 0, outIndices);
      (*sampled_coo_rows)[etype] =
          aten::IndexSelect((*sampled_coo_rows)[etype], 0, outIndices);
      (*sampled_graphs)[etype] = CSRMatrix(
          (*sampled_graphs)[etype].num_rows, outIndices,
          (*sampled_graphs)[etype].indptr, remain_indices[etype],
          remain_induced_edges[etype]);
    });
  }
}

std::pair<HeteroSubgraph, std::vector<FloatArray>> ExcludeCertainEdges(
    const HeteroSubgraph& sg, const std::vector<IdArray>& exclude_edges,
    const std::vector<FloatArray>* weights = nullptr) {
  HeteroGraphPtr hg_view = HeteroGraphRef(sg.graph).sptr();
  std::vector<IdArray> remain_induced_edges(hg_view->NumEdgeTypes());
  std::vector<IdArray> remain_edges(hg_view->NumEdgeTypes());
  std::vector<FloatArray> remain_weights(hg_view->NumEdgeTypes());

  for (dgl_type_t etype = 0; etype < hg_view->NumEdgeTypes(); ++etype) {
    IdArray edge_ids = Range(
        0, sg.induced_edges[etype]->shape[0],
        sg.induced_edges[etype]->dtype.bits, sg.induced_edges[etype]->ctx);
    if (exclude_edges[etype].GetSize() == 0 || edge_ids.GetSize() == 0) {
      remain_edges[etype] = edge_ids;
      remain_induced_edges[etype] = sg.induced_edges[etype];
      if (weights) remain_weights[etype] = (*weights)[etype];
      continue;
    }
    ATEN_ID_TYPE_SWITCH(hg_view->DataType(), IdType, {
      const auto dtype = weights && (*weights)[etype]->shape[0]
                             ? (*weights)[etype]->dtype
                             : DGLDataType{kDGLFloat, 8 * sizeof(float), 1};
      ATEN_FLOAT_TYPE_SWITCH(dtype, FloatType, "weights", {
        IdType* idx_data = edge_ids.Ptr<IdType>();
        IdType* induced_edges_data = sg.induced_edges[etype].Ptr<IdType>();
        FloatType* weights_data = weights && (*weights)[etype]->shape[0]
                                      ? (*weights)[etype].Ptr<FloatType>()
                                      : nullptr;
        const IdType exclude_edges_len = exclude_edges[etype]->shape[0];
        std::sort(
            exclude_edges[etype].Ptr<IdType>(),
            exclude_edges[etype].Ptr<IdType>() + exclude_edges_len);
        const IdType* exclude_edges_data = exclude_edges[etype].Ptr<IdType>();
        IdType outId = 0;
        for (IdType i = 0; i != sg.induced_edges[etype]->shape[0]; ++i) {
          // the following binary search is the bottleneck, excluding weights
          // together with edges should almost be free.
          if (!std::binary_search(
                  exclude_edges_data, exclude_edges_data + exclude_edges_len,
                  induced_edges_data[i])) {
            induced_edges_data[outId] = induced_edges_data[i];
            idx_data[outId] = idx_data[i];
            if (weights_data) weights_data[outId] = weights_data[i];
            ++outId;
          }
        }
        remain_edges[etype] = aten::IndexSelect(edge_ids, 0, outId);
        remain_induced_edges[etype] =
            aten::IndexSelect(sg.induced_edges[etype], 0, outId);
        remain_weights[etype] =
            weights_data ? aten::IndexSelect((*weights)[etype], 0, outId)
                         : NullArray();
      });
    });
  }
  HeteroSubgraph subg = hg_view->EdgeSubgraph(remain_edges, true);
  subg.induced_edges = std::move(remain_induced_edges);
  return std::make_pair(subg, remain_weights);
}

std::pair<HeteroSubgraph, std::vector<FloatArray>> SampleLabors(
    const HeteroGraphPtr hg, const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& fanouts, EdgeDir dir,
    const std::vector<FloatArray>& prob,
    const std::vector<IdArray>& exclude_edges, const int importance_sampling,
    const IdArray random_seed, const float seed2_contribution,
    const std::vector<IdArray>& NIDs) {
  // sanity check
  CHECK_EQ(nodes.size(), hg->NumVertexTypes())
      << "Number of node ID tensors must match the number of node types.";
  CHECK_EQ(fanouts.size(), hg->NumEdgeTypes())
      << "Number of fanout values must match the number of edge types.";

  DGLContext ctx = aten::GetContextOf(nodes);

  std::vector<HeteroGraphPtr> subrels(hg->NumEdgeTypes());
  std::vector<FloatArray> subimportances(hg->NumEdgeTypes());
  std::vector<IdArray> induced_edges(hg->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const IdArray nodes_ntype =
        nodes[(dir == EdgeDir::kOut) ? src_vtype : dst_vtype];
    const IdArray NIDs_ntype =
        NIDs[(dir == EdgeDir::kIn) ? src_vtype : dst_vtype];
    const int64_t num_nodes = nodes_ntype->shape[0];
    if (num_nodes == 0 || fanouts[etype] == 0) {
      // Nothing to sample for this etype, create a placeholder relation graph
      subrels[etype] = UnitGraph::Empty(
          hg->GetRelationGraph(etype)->NumVertexTypes(),
          hg->NumVertices(src_vtype), hg->NumVertices(dst_vtype),
          hg->DataType(), ctx);
      induced_edges[etype] = aten::NullArray(hg->DataType(), ctx);
      subimportances[etype] = NullArray();
    } else {
      // sample from one relation graph
      auto req_fmt = (dir == EdgeDir::kOut) ? CSR_CODE : CSC_CODE;
      auto avail_fmt = hg->SelectFormat(etype, req_fmt);
      COOMatrix sampled_coo;
      FloatArray importances;
      const int64_t fanout =
          fanouts[etype] >= 0
              ? fanouts[etype]
              : std::max(
                    hg->NumVertices(dst_vtype), hg->NumVertices(src_vtype));
      switch (avail_fmt) {
        case SparseFormat::kCOO:
          if (dir == EdgeDir::kIn) {
            auto fs = aten::COOLaborSampling(
                aten::COOTranspose(hg->GetCOOMatrix(etype)), nodes_ntype,
                fanout, prob[etype], importance_sampling, random_seed,
                seed2_contribution, NIDs_ntype);
            sampled_coo = aten::COOTranspose(fs.first);
            importances = fs.second;
          } else {
            std::tie(sampled_coo, importances) = aten::COOLaborSampling(
                hg->GetCOOMatrix(etype), nodes_ntype, fanout, prob[etype],
                importance_sampling, random_seed, seed2_contribution,
                NIDs_ntype);
          }
          break;
        case SparseFormat::kCSR:
          CHECK(dir == EdgeDir::kOut)
              << "Cannot sample out edges on CSC matrix.";
          std::tie(sampled_coo, importances) = aten::CSRLaborSampling(
              hg->GetCSRMatrix(etype), nodes_ntype, fanout, prob[etype],
              importance_sampling, random_seed, seed2_contribution, NIDs_ntype);
          break;
        case SparseFormat::kCSC:
          CHECK(dir == EdgeDir::kIn) << "Cannot sample in edges on CSR matrix.";
          std::tie(sampled_coo, importances) = aten::CSRLaborSampling(
              hg->GetCSCMatrix(etype), nodes_ntype, fanout, prob[etype],
              importance_sampling, random_seed, seed2_contribution, NIDs_ntype);
          sampled_coo = aten::COOTranspose(sampled_coo);
          break;
        default:
          LOG(FATAL) << "Unsupported sparse format.";
      }
      subrels[etype] = UnitGraph::CreateFromCOO(
          hg->GetRelationGraph(etype)->NumVertexTypes(), sampled_coo.num_rows,
          sampled_coo.num_cols, sampled_coo.row, sampled_coo.col);
      subimportances[etype] = importances;
      induced_edges[etype] = sampled_coo.data;
    }
  }

  HeteroSubgraph ret;
  ret.graph =
      CreateHeteroGraph(hg->meta_graph(), subrels, hg->NumVerticesPerType());
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = std::move(induced_edges);

  if (!exclude_edges.empty())
    return ExcludeCertainEdges(ret, exclude_edges, &subimportances);

  return std::make_pair(ret, std::move(subimportances));
}

HeteroSubgraph SampleNeighbors(
    const HeteroGraphPtr hg, const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& fanouts, EdgeDir dir,
    const std::vector<NDArray>& prob_or_mask,
    const std::vector<IdArray>& exclude_edges, bool replace) {
  // sanity check
  CHECK_EQ(nodes.size(), hg->NumVertexTypes())
      << "Number of node ID tensors must match the number of node types.";
  CHECK_EQ(fanouts.size(), hg->NumEdgeTypes())
      << "Number of fanout values must match the number of edge types.";
  CHECK_EQ(prob_or_mask.size(), hg->NumEdgeTypes())
      << "Number of probability tensors must match the number of edge types.";

  DGLContext ctx = aten::GetContextOf(nodes);

  std::vector<HeteroGraphPtr> subrels(hg->NumEdgeTypes());
  std::vector<IdArray> induced_edges(hg->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const IdArray nodes_ntype =
        nodes[(dir == EdgeDir::kOut) ? src_vtype : dst_vtype];
    const int64_t num_nodes = nodes_ntype->shape[0];

    if (num_nodes == 0 || fanouts[etype] == 0) {
      // Nothing to sample for this etype, create a placeholder relation graph
      subrels[etype] = UnitGraph::Empty(
          hg->GetRelationGraph(etype)->NumVertexTypes(),
          hg->NumVertices(src_vtype), hg->NumVertices(dst_vtype),
          hg->DataType(), ctx);
      induced_edges[etype] = aten::NullArray(hg->DataType(), ctx);
    } else {
      COOMatrix sampled_coo;
      // sample from one relation graph
      auto req_fmt = (dir == EdgeDir::kOut) ? CSR_CODE : CSC_CODE;
      auto avail_fmt = hg->SelectFormat(etype, req_fmt);
      switch (avail_fmt) {
        case SparseFormat::kCOO:
          if (dir == EdgeDir::kIn) {
            sampled_coo = aten::COOTranspose(aten::COORowWiseSampling(
                aten::COOTranspose(hg->GetCOOMatrix(etype)), nodes_ntype,
                fanouts[etype], prob_or_mask[etype], replace));
          } else {
            sampled_coo = aten::COORowWiseSampling(
                hg->GetCOOMatrix(etype), nodes_ntype, fanouts[etype],
                prob_or_mask[etype], replace);
          }
          break;
        case SparseFormat::kCSR:
          CHECK(dir == EdgeDir::kOut)
              << "Cannot sample out edges on CSC matrix.";
          sampled_coo = aten::CSRRowWiseSampling(
              hg->GetCSRMatrix(etype), nodes_ntype, fanouts[etype],
              prob_or_mask[etype], replace);
          break;
        case SparseFormat::kCSC:
          CHECK(dir == EdgeDir::kIn) << "Cannot sample in edges on CSR matrix.";
          sampled_coo = aten::CSRRowWiseSampling(
              hg->GetCSCMatrix(etype), nodes_ntype, fanouts[etype],
              prob_or_mask[etype], replace);
          sampled_coo = aten::COOTranspose(sampled_coo);
          break;
        default:
          LOG(FATAL) << "Unsupported sparse format.";
      }

      subrels[etype] = UnitGraph::CreateFromCOO(
          hg->GetRelationGraph(etype)->NumVertexTypes(), sampled_coo.num_rows,
          sampled_coo.num_cols, sampled_coo.row, sampled_coo.col);
      induced_edges[etype] = sampled_coo.data;
    }
  }

  HeteroSubgraph ret;
  ret.graph =
      CreateHeteroGraph(hg->meta_graph(), subrels, hg->NumVerticesPerType());
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = std::move(induced_edges);
  if (!exclude_edges.empty()) {
    return ExcludeCertainEdges(ret, exclude_edges).first;
  }
  return ret;
}

template <typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
SampleNeighborsFused(
    const HeteroGraphPtr hg, const std::vector<IdArray>& nodes,
    const std::vector<IdArray>& mapping, const std::vector<int64_t>& fanouts,
    EdgeDir dir, const std::vector<NDArray>& prob_or_mask,
    const std::vector<IdArray>& exclude_edges, bool replace) {
  CHECK_EQ(nodes.size(), hg->NumVertexTypes())
      << "Number of node ID tensors must match the number of node types.";
  CHECK_EQ(fanouts.size(), hg->NumEdgeTypes())
      << "Number of fanout values must match the number of edge types.";
  CHECK_EQ(prob_or_mask.size(), hg->NumEdgeTypes())
      << "Number of probability tensors must match the number of edge types.";

  DGLContext ctx = aten::GetContextOf(nodes);

  std::vector<CSRMatrix> sampled_graphs;
  std::vector<IdArray> sampled_coo_rows;
  std::vector<IdArray> induced_edges;
  std::vector<IdArray> induced_vertices;
  std::vector<int64_t> num_nodes_per_type;
  std::vector<std::vector<IdType>> new_nodes_vec(hg->NumVertexTypes());
  std::vector<int> seed_nodes_mapped(hg->NumVertexTypes(), 0);

  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const dgl_type_t rhs_node_type =
        (dir == EdgeDir::kOut) ? src_vtype : dst_vtype;
    const IdArray nodes_ntype = nodes[rhs_node_type];
    const int64_t num_nodes = nodes_ntype->shape[0];

    if (num_nodes == 0 || fanouts[etype] == 0) {
      // Nothing to sample for this etype, create a placeholder
      sampled_graphs.push_back(CSRMatrix());
      sampled_coo_rows.push_back(IdArray());
      induced_edges.push_back(aten::NullArray(hg->DataType(), ctx));
    } else {
      bool map_seed_nodes = !seed_nodes_mapped[rhs_node_type];
      // sample from one relation graph
      std::pair<CSRMatrix, IdArray> sampled_graph;
      auto sampling_fn = map_seed_nodes
                             ? aten::CSRRowWiseSamplingFused<IdType, true>
                             : aten::CSRRowWiseSamplingFused<IdType, false>;
      auto req_fmt = (dir == EdgeDir::kOut) ? CSR_CODE : CSC_CODE;
      auto avail_fmt = hg->SelectFormat(etype, req_fmt);
      switch (avail_fmt) {
        case SparseFormat::kCSR:
          CHECK(dir == EdgeDir::kOut)
              << "Cannot sample out edges on CSC matrix.";
          // In heterographs nodes of two diffrent types can be connected
          // therefore two diffrent mappings and node vectors are needed
          sampled_graph = sampling_fn(
              hg->GetCSRMatrix(etype), nodes_ntype, mapping[src_vtype],
              &new_nodes_vec[src_vtype], fanouts[etype], prob_or_mask[etype],
              replace);
          break;
        case SparseFormat::kCSC:
          CHECK(dir == EdgeDir::kIn) << "Cannot sample in edges on CSR matrix.";
          sampled_graph = sampling_fn(
              hg->GetCSCMatrix(etype), nodes_ntype, mapping[dst_vtype],
              &new_nodes_vec[dst_vtype], fanouts[etype], prob_or_mask[etype],
              replace);
          break;
        default:
          LOG(FATAL) << "Unsupported sparse format.";
      }
      seed_nodes_mapped[rhs_node_type]++;
      sampled_graphs.push_back(sampled_graph.first);
      if (sampled_graph.first.data.defined())
        induced_edges.push_back(sampled_graph.first.data);
      else
        induced_edges.push_back(
            aten::NullArray(DGLDataType{kDGLInt, sizeof(IdType) * 8, 1}, ctx));
      sampled_coo_rows.push_back(sampled_graph.second);
    }
  }

  if (!exclude_edges.empty()) {
    ExcludeCertainEdgesFused<IdType>(
        &sampled_graphs, &induced_edges, &sampled_coo_rows, exclude_edges);
    for (size_t i = 0; i < hg->NumEdgeTypes(); i++) {
      if (sampled_graphs[i].data.defined())
        induced_edges[i] = std::move(sampled_graphs[i].data);
      else
        induced_edges[i] =
            aten::NullArray(DGLDataType{kDGLInt, sizeof(IdType) * 8, 1}, ctx);
    }
  }

  // map indices
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const dgl_type_t lhs_node_type =
        (dir == EdgeDir::kIn) ? src_vtype : dst_vtype;
    if (sampled_graphs[etype].num_cols != 0) {
      auto num_cols = sampled_graphs[etype].num_cols;
      int num_threads_col = runtime::compute_num_threads(0, num_cols, 1);
      std::vector<IdType> global_prefix_col(num_threads_col + 1, 0);
      std::vector<std::vector<IdType>> src_nodes_local(num_threads_col);
      IdType* mapping_data_dst = mapping[lhs_node_type].Ptr<IdType>();
      IdType* cdata = sampled_graphs[etype].indices.Ptr<IdType>();
#pragma omp parallel num_threads(num_threads_col)
      {
        const int thread_id = omp_get_thread_num();
        num_threads_col = omp_get_num_threads();

        const int64_t start_i =
            thread_id * (num_cols / num_threads_col) +
            std::min(
                static_cast<int64_t>(thread_id), num_cols % num_threads_col);
        const int64_t end_i = (thread_id + 1) * (num_cols / num_threads_col) +
                              std::min(
                                  static_cast<int64_t>(thread_id + 1),
                                  num_cols % num_threads_col);
        assert(thread_id + 1 < num_threads_col || end_i == num_cols);
        for (int64_t i = start_i; i < end_i; ++i) {
          int64_t picked_idx = cdata[i];
          bool spot_claimed =
              BoolCompareAndSwap<IdType>(&mapping_data_dst[picked_idx]);
          if (spot_claimed) src_nodes_local[thread_id].push_back(picked_idx);
        }
        global_prefix_col[thread_id + 1] = src_nodes_local[thread_id].size();

#pragma omp barrier
#pragma omp master
        {
          global_prefix_col[0] = new_nodes_vec[lhs_node_type].size();
          for (int t = 0; t < num_threads_col; ++t) {
            global_prefix_col[t + 1] += global_prefix_col[t];
          }
        }

#pragma omp barrier
        int64_t mapping_shift = global_prefix_col[thread_id];
        for (size_t i = 0; i < src_nodes_local[thread_id].size(); ++i)
          mapping_data_dst[src_nodes_local[thread_id][i]] = mapping_shift + i;

#pragma omp barrier
        for (int64_t i = start_i; i < end_i; ++i) {
          IdType picked_idx = cdata[i];
          IdType mapped_idx = mapping_data_dst[picked_idx];
          cdata[i] = mapped_idx;
        }
      }
      IdType offset = new_nodes_vec[lhs_node_type].size();
      new_nodes_vec[lhs_node_type].resize(global_prefix_col.back());
      for (int thread_id = 0; thread_id < num_threads_col; ++thread_id) {
        memcpy(
            new_nodes_vec[lhs_node_type].data() + offset,
            &src_nodes_local[thread_id][0],
            src_nodes_local[thread_id].size() * sizeof(IdType));
        offset += src_nodes_local[thread_id].size();
      }
    }
  }

  // counting how many nodes of each ntype were sampled
  num_nodes_per_type.resize(2 * hg->NumVertexTypes());
  for (size_t i = 0; i < hg->NumVertexTypes(); i++) {
    num_nodes_per_type[i] = new_nodes_vec[i].size();
    num_nodes_per_type[hg->NumVertexTypes() + i] = nodes[i]->shape[0];
    induced_vertices.push_back(
        VecToIdArray(new_nodes_vec[i], sizeof(IdType) * 8));
  }

  std::vector<HeteroGraphPtr> subrels(hg->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    if (sampled_graphs[etype].num_rows == 0) {
      subrels[etype] = UnitGraph::Empty(
          2, new_nodes_vec[src_vtype].size(), nodes[dst_vtype]->shape[0],
          hg->DataType(), ctx);
    } else {
      CSRMatrix graph = sampled_graphs[etype];
      if (dir == EdgeDir::kOut) {
        subrels[etype] = UnitGraph::CreateFromCSRAndCOO(
            2,
            CSRMatrix(
                nodes[src_vtype]->shape[0], new_nodes_vec[dst_vtype].size(),
                graph.indptr, graph.indices,
                Range(
                    0, graph.indices->shape[0], graph.indices->dtype.bits,
                    ctx)),
            COOMatrix(
                nodes[src_vtype]->shape[0], new_nodes_vec[dst_vtype].size(),
                sampled_coo_rows[etype], graph.indices),
            ALL_CODE);
      } else {
        subrels[etype] = UnitGraph::CreateFromCSCAndCOO(
            2,
            CSRMatrix(
                nodes[dst_vtype]->shape[0], new_nodes_vec[src_vtype].size(),
                graph.indptr, graph.indices,
                Range(
                    0, graph.indices->shape[0], graph.indices->dtype.bits,
                    ctx)),
            COOMatrix(
                new_nodes_vec[src_vtype].size(), nodes[dst_vtype]->shape[0],
                graph.indices, sampled_coo_rows[etype]),
            ALL_CODE);
      }
    }
  }

  HeteroSubgraph ret;

  const auto meta_graph = hg->meta_graph();
  const EdgeArray etypes = meta_graph->Edges("eid");
  const IdArray new_dst = Add(etypes.dst, hg->NumVertexTypes());

  const auto new_meta_graph = ImmutableGraph::CreateFromCOO(
      hg->NumVertexTypes() * 2, etypes.src, new_dst);

  HeteroGraphPtr new_graph =
      CreateHeteroGraph(new_meta_graph, subrels, num_nodes_per_type);
  return std::make_tuple(new_graph, induced_edges, induced_vertices);
}

template std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
SampleNeighborsFused<int64_t>(
    const HeteroGraphPtr, const std::vector<IdArray>&,
    const std::vector<IdArray>&, const std::vector<int64_t>&, EdgeDir,
    const std::vector<NDArray>&, const std::vector<IdArray>&, bool);

template std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
SampleNeighborsFused<int32_t>(
    const HeteroGraphPtr, const std::vector<IdArray>&,
    const std::vector<IdArray>&, const std::vector<int64_t>&, EdgeDir,
    const std::vector<NDArray>&, const std::vector<IdArray>&, bool);

HeteroSubgraph SampleNeighborsEType(
    const HeteroGraphPtr hg, const IdArray nodes,
    const std::vector<int64_t>& eid2etype_offset,
    const std::vector<int64_t>& fanouts, EdgeDir dir,
    const std::vector<FloatArray>& prob, bool replace,
    bool rowwise_etype_sorted) {
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

  bool same_fanout = true;
  int64_t fanout_value = fanouts[0];
  for (auto fanout : fanouts) {
    if (fanout != fanout_value) {
      same_fanout = false;
      break;
    }
  }

  if (num_nodes == 0 || (same_fanout && fanout_value == 0)) {
    subrels[etype] = UnitGraph::Empty(
        1, hg->NumVertices(src_vtype), hg->NumVertices(dst_vtype),
        hg->DataType(), hg->Context());
    induced_edges[etype] = aten::NullArray();
  } else {
    COOMatrix sampled_coo;
    // sample from graph
    // the edge type is stored in etypes
    auto req_fmt = (dir == EdgeDir::kOut) ? CSR_CODE : CSC_CODE;
    auto avail_fmt = hg->SelectFormat(etype, req_fmt);
    switch (avail_fmt) {
      case SparseFormat::kCOO:
        if (dir == EdgeDir::kIn) {
          sampled_coo = aten::COOTranspose(aten::COORowWisePerEtypeSampling(
              aten::COOTranspose(hg->GetCOOMatrix(etype)), nodes,
              eid2etype_offset, fanouts, prob, replace));
        } else {
          sampled_coo = aten::COORowWisePerEtypeSampling(
              hg->GetCOOMatrix(etype), nodes, eid2etype_offset, fanouts, prob,
              replace);
        }
        break;
      case SparseFormat::kCSR:
        CHECK(dir == EdgeDir::kOut) << "Cannot sample out edges on CSC matrix.";
        sampled_coo = aten::CSRRowWisePerEtypeSampling(
            hg->GetCSRMatrix(etype), nodes, eid2etype_offset, fanouts, prob,
            replace, rowwise_etype_sorted);
        break;
      case SparseFormat::kCSC:
        CHECK(dir == EdgeDir::kIn) << "Cannot sample in edges on CSR matrix.";
        sampled_coo = aten::CSRRowWisePerEtypeSampling(
            hg->GetCSCMatrix(etype), nodes, eid2etype_offset, fanouts, prob,
            replace, rowwise_etype_sorted);
        sampled_coo = aten::COOTranspose(sampled_coo);
        break;
      default:
        LOG(FATAL) << "Unsupported sparse format.";
    }

    subrels[etype] = UnitGraph::CreateFromCOO(
        1, sampled_coo.num_rows, sampled_coo.num_cols, sampled_coo.row,
        sampled_coo.col);
    induced_edges[etype] = sampled_coo.data;
  }

  HeteroSubgraph ret;
  ret.graph =
      CreateHeteroGraph(hg->meta_graph(), subrels, hg->NumVerticesPerType());
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = std::move(induced_edges);
  return ret;
}

HeteroSubgraph SampleNeighborsTopk(
    const HeteroGraphPtr hg, const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& k, EdgeDir dir,
    const std::vector<FloatArray>& weight, bool ascending) {
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
    const IdArray nodes_ntype =
        nodes[(dir == EdgeDir::kOut) ? src_vtype : dst_vtype];
    const int64_t num_nodes = nodes_ntype->shape[0];
    if (num_nodes == 0 || k[etype] == 0) {
      // Nothing to sample for this etype, create a placeholder relation graph
      subrels[etype] = UnitGraph::Empty(
          hg->GetRelationGraph(etype)->NumVertexTypes(),
          hg->NumVertices(src_vtype), hg->NumVertices(dst_vtype),
          hg->DataType(), hg->Context());
      induced_edges[etype] = aten::NullArray();
    } else {
      // sample from one relation graph
      auto req_fmt = (dir == EdgeDir::kOut) ? CSR_CODE : CSC_CODE;
      auto avail_fmt = hg->SelectFormat(etype, req_fmt);
      COOMatrix sampled_coo;
      switch (avail_fmt) {
        case SparseFormat::kCOO:
          if (dir == EdgeDir::kIn) {
            sampled_coo = aten::COOTranspose(aten::COORowWiseTopk(
                aten::COOTranspose(hg->GetCOOMatrix(etype)), nodes_ntype,
                k[etype], weight[etype], ascending));
          } else {
            sampled_coo = aten::COORowWiseTopk(
                hg->GetCOOMatrix(etype), nodes_ntype, k[etype], weight[etype],
                ascending);
          }
          break;
        case SparseFormat::kCSR:
          CHECK(dir == EdgeDir::kOut)
              << "Cannot sample out edges on CSC matrix.";
          sampled_coo = aten::CSRRowWiseTopk(
              hg->GetCSRMatrix(etype), nodes_ntype, k[etype], weight[etype],
              ascending);
          break;
        case SparseFormat::kCSC:
          CHECK(dir == EdgeDir::kIn) << "Cannot sample in edges on CSR matrix.";
          sampled_coo = aten::CSRRowWiseTopk(
              hg->GetCSCMatrix(etype), nodes_ntype, k[etype], weight[etype],
              ascending);
          sampled_coo = aten::COOTranspose(sampled_coo);
          break;
        default:
          LOG(FATAL) << "Unsupported sparse format.";
      }
      subrels[etype] = UnitGraph::CreateFromCOO(
          hg->GetRelationGraph(etype)->NumVertexTypes(), sampled_coo.num_rows,
          sampled_coo.num_cols, sampled_coo.row, sampled_coo.col);
      induced_edges[etype] = sampled_coo.data;
    }
  }

  HeteroSubgraph ret;
  ret.graph =
      CreateHeteroGraph(hg->meta_graph(), subrels, hg->NumVerticesPerType());
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = std::move(induced_edges);
  return ret;
}

HeteroSubgraph SampleNeighborsBiased(
    const HeteroGraphPtr hg, const IdArray& nodes, const int64_t fanout,
    const NDArray& bias, const NDArray& tag_offset, const EdgeDir dir,
    const bool replace) {
  CHECK_EQ(hg->NumEdgeTypes(), 1)
      << "Only homogeneous or bipartite graphs are supported";
  auto pair = hg->meta_graph()->FindEdge(0);
  const dgl_type_t src_vtype = pair.first;
  const dgl_type_t dst_vtype = pair.second;
  const dgl_type_t nodes_ntype = (dir == EdgeDir::kOut) ? src_vtype : dst_vtype;

  // sanity check
  CHECK_EQ(tag_offset->ndim, 2)
      << "The shape of tag_offset should be [num_nodes, num_tags + 1]";
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
        hg->NumVertices(src_vtype), hg->NumVertices(dst_vtype), hg->DataType(),
        hg->Context());
    induced_edges = aten::NullArray();
  } else {
    // sample from one relation graph
    const auto req_fmt = (dir == EdgeDir::kOut) ? CSR_CODE : CSC_CODE;
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
        hg->GetRelationGraph(etype)->NumVertexTypes(), sampled_coo.num_rows,
        sampled_coo.num_cols, sampled_coo.row, sampled_coo.col);
    induced_edges = sampled_coo.data;
  }

  HeteroSubgraph ret;
  ret.graph =
      CreateHeteroGraph(hg->meta_graph(), {subrel}, hg->NumVerticesPerType());
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = {induced_edges};
  return ret;
}

DGL_REGISTER_GLOBAL("sampling.neighbor._CAPI_DGLSampleNeighborsEType")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef hg = args[0];
      IdArray nodes = args[1];
      const std::vector<int64_t>& eid2etype_offset =
          ListValueToVector<int64_t>(args[2]);
      IdArray fanout = args[3];
      const std::string dir_str = args[4];
      const auto& prob = ListValueToVector<FloatArray>(args[5]);
      const bool replace = args[6];
      const bool rowwise_etype_sorted = args[7];

      CHECK(dir_str == "in" || dir_str == "out")
          << "Invalid edge direction. Must be \"in\" or \"out\".";
      EdgeDir dir = (dir_str == "in") ? EdgeDir::kIn : EdgeDir::kOut;
      CHECK_INT64(fanout, "fanout");
      std::vector<int64_t> fanout_vec = fanout.ToVector<int64_t>();

      std::shared_ptr<HeteroSubgraph> subg(new HeteroSubgraph);
      *subg = sampling::SampleNeighborsEType(
          hg.sptr(), nodes, eid2etype_offset, fanout_vec, dir, prob, replace,
          rowwise_etype_sorted);
      *rv = HeteroSubgraphRef(subg);
    });

DGL_REGISTER_GLOBAL("sampling.labor._CAPI_DGLSampleLabors")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef hg = args[0];
      const auto& nodes = ListValueToVector<IdArray>(args[1]);
      IdArray fanouts_array = args[2];
      const auto& fanouts = fanouts_array.ToVector<int64_t>();
      const std::string dir_str = args[3];
      const auto& prob = ListValueToVector<FloatArray>(args[4]);
      const auto& exclude_edges = ListValueToVector<IdArray>(args[5]);
      const int importance_sampling = args[6];
      const IdArray random_seed = args[7];
      const double seed2_contribution = args[8];
      const auto& NIDs = ListValueToVector<IdArray>(args[9]);

      CHECK(dir_str == "in" || dir_str == "out")
          << "Invalid edge direction. Must be \"in\" or \"out\".";
      EdgeDir dir = (dir_str == "in") ? EdgeDir::kIn : EdgeDir::kOut;

      std::shared_ptr<HeteroSubgraph> subg_ptr(new HeteroSubgraph);

      auto&& subg_importances = sampling::SampleLabors(
          hg.sptr(), nodes, fanouts, dir, prob, exclude_edges,
          importance_sampling, random_seed, seed2_contribution, NIDs);
      *subg_ptr = subg_importances.first;
      List<Value> ret_val;
      ret_val.push_back(Value(subg_ptr));
      for (auto& imp : subg_importances.second)
        ret_val.push_back(Value(MakeValue(imp)));

      *rv = ret_val;
    });

DGL_REGISTER_GLOBAL("sampling.neighbor._CAPI_DGLSampleNeighbors")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef hg = args[0];
      const auto& nodes = ListValueToVector<IdArray>(args[1]);
      IdArray fanouts_array = args[2];
      const auto& fanouts = fanouts_array.ToVector<int64_t>();
      const std::string dir_str = args[3];
      const auto& prob_or_mask = ListValueToVector<NDArray>(args[4]);
      const auto& exclude_edges = ListValueToVector<IdArray>(args[5]);
      const bool replace = args[6];

      CHECK(dir_str == "in" || dir_str == "out")
          << "Invalid edge direction. Must be \"in\" or \"out\".";
      EdgeDir dir = (dir_str == "in") ? EdgeDir::kIn : EdgeDir::kOut;

      std::shared_ptr<HeteroSubgraph> subg(new HeteroSubgraph);
      *subg = sampling::SampleNeighbors(
          hg.sptr(), nodes, fanouts, dir, prob_or_mask, exclude_edges, replace);

      *rv = HeteroSubgraphRef(subg);
    });

DGL_REGISTER_GLOBAL("sampling.neighbor._CAPI_DGLSampleNeighborsFused")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef hg = args[0];
      const auto& nodes = ListValueToVector<IdArray>(args[1]);
      auto mapping = ListValueToVector<IdArray>(args[2]);
      IdArray fanouts_array = args[3];
      const auto& fanouts = fanouts_array.ToVector<int64_t>();
      const std::string dir_str = args[4];
      const auto& prob_or_mask = ListValueToVector<NDArray>(args[5]);
      const auto& exclude_edges = ListValueToVector<IdArray>(args[6]);
      const bool replace = args[7];

      CHECK(dir_str == "in" || dir_str == "out")
          << "Invalid edge direction. Must be \"in\" or \"out\".";
      EdgeDir dir = (dir_str == "in") ? EdgeDir::kIn : EdgeDir::kOut;

      HeteroGraphPtr new_graph;
      std::vector<IdArray> induced_edges;
      std::vector<IdArray> induced_vertices;

      ATEN_ID_TYPE_SWITCH(hg->DataType(), IdType, {
        std::tie(new_graph, induced_edges, induced_vertices) =
            SampleNeighborsFused<IdType>(
                hg.sptr(), nodes, mapping, fanouts, dir, prob_or_mask,
                exclude_edges, replace);
      });

      List<Value> lhs_nodes_ref;
      for (IdArray& array : induced_vertices)
        lhs_nodes_ref.push_back(Value(MakeValue(array)));
      List<Value> induced_edges_ref;
      for (IdArray& array : induced_edges)
        induced_edges_ref.push_back(Value(MakeValue(array)));
      List<ObjectRef> ret;
      ret.push_back(HeteroGraphRef(new_graph));
      ret.push_back(lhs_nodes_ref);
      ret.push_back(induced_edges_ref);

      *rv = ret;
    });

DGL_REGISTER_GLOBAL("sampling.neighbor._CAPI_DGLSampleNeighborsTopk")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef hg = args[0];
      const auto& nodes = ListValueToVector<IdArray>(args[1]);
      IdArray k_array = args[2];
      const auto& k = k_array.ToVector<int64_t>();
      const std::string dir_str = args[3];
      const auto& weight = ListValueToVector<FloatArray>(args[4]);
      const bool ascending = args[5];

      CHECK(dir_str == "in" || dir_str == "out")
          << "Invalid edge direction. Must be \"in\" or \"out\".";
      EdgeDir dir = (dir_str == "in") ? EdgeDir::kIn : EdgeDir::kOut;

      std::shared_ptr<HeteroSubgraph> subg(new HeteroSubgraph);
      *subg = sampling::SampleNeighborsTopk(
          hg.sptr(), nodes, k, dir, weight, ascending);

      *rv = HeteroGraphRef(subg);
    });

DGL_REGISTER_GLOBAL("sampling.neighbor._CAPI_DGLSampleNeighborsBiased")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef hg = args[0];
      const IdArray nodes = args[1];
      const int64_t fanout = args[2];
      const NDArray bias = args[3];
      const NDArray tag_offset = args[4];
      const std::string dir_str = args[5];
      const bool replace = args[6];

      CHECK(dir_str == "in" || dir_str == "out")
          << "Invalid edge direction. Must be \"in\" or \"out\".";
      EdgeDir dir = (dir_str == "in") ? EdgeDir::kIn : EdgeDir::kOut;

      std::shared_ptr<HeteroSubgraph> subg(new HeteroSubgraph);
      *subg = sampling::SampleNeighborsBiased(
          hg.sptr(), nodes, fanout, bias, tag_offset, dir, replace);

      *rv = HeteroGraphRef(subg);
    });

}  // namespace sampling
}  // namespace dgl
