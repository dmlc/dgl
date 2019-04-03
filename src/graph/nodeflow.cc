/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/nodeflow.cc
 * \brief DGL NodeFlow related functions.
 */

#include <dgl/immutable_graph.h>
#include <dgl/nodeflow.h>
#include <dgl/sampler.h>

#include <string>
#include <numeric>

#include "../c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::PackedFunc;
using dgl::runtime::NDArray;

namespace dgl {

std::vector<IdArray> GetNodeFlowSlice(const ImmutableGraph &graph, const std::string &fmt,
                                      size_t layer0_size, size_t layer1_start,
                                      size_t layer1_end, bool remap) {
  CHECK_GE(layer1_start, layer0_size);
  if (fmt == "csr") {
    dgl_id_t first_vid = layer1_start - layer0_size;
    ImmutableGraph::CSRArray arrs = graph.GetInCSRArray(layer1_start, layer1_end);
    if (remap) {
      dgl_id_t *indices_data = static_cast<dgl_id_t*>(arrs.indices->data);
      dgl_id_t *eid_data = static_cast<dgl_id_t*>(arrs.id->data);
      const size_t len = arrs.indices->shape[0];
      dgl_id_t first_eid = eid_data[0];
      for (size_t i = 0; i < len; i++) {
        CHECK_GE(indices_data[i], first_vid);
        indices_data[i] -= first_vid;
        CHECK_GE(eid_data[i], first_eid);
        eid_data[i] -= first_eid;
      }
    }
    return std::vector<IdArray>{arrs.indptr, arrs.indices, arrs.id};
  } else if (fmt == "coo") {
    ImmutableGraph::CSR::Ptr csr = graph.GetInCSR();
    int64_t nnz = csr->indptr[layer1_end] - csr->indptr[layer1_start];
    IdArray idx = IdArray::Empty({2 * nnz}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
    IdArray eid = IdArray::Empty({nnz}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
    int64_t *idx_data = static_cast<int64_t*>(idx->data);
    dgl_id_t *eid_data = static_cast<dgl_id_t*>(eid->data);
    size_t num_edges = 0;
    for (size_t i = layer1_start; i < layer1_end; i++) {
      for (int64_t j = csr->indptr[i]; j < csr->indptr[i + 1]; j++) {
        // These nodes are all in a layer. We need to remap them to the node id
        // local to the layer.
        idx_data[num_edges] = remap ? i - layer1_start : i;
        num_edges++;
      }
    }
    CHECK_EQ(num_edges, nnz);
    if (remap) {
      size_t edge_start = csr->indptr[layer1_start];
      dgl_id_t first_eid = csr->edge_ids[edge_start];
      dgl_id_t first_vid = layer1_start - layer0_size;
      for (int64_t i = 0; i < nnz; i++) {
        CHECK_GE(csr->indices[edge_start + i], first_vid);
        idx_data[nnz + i] = csr->indices[edge_start + i] - first_vid;
        eid_data[i] = csr->edge_ids[edge_start + i] - first_eid;
      }
    } else {
      std::copy(csr->indices.begin() + csr->indptr[layer1_start],
                csr->indices.begin() + csr->indptr[layer1_end], idx_data + nnz);
      std::copy(csr->edge_ids.begin() + csr->indptr[layer1_start],
                csr->edge_ids.begin() + csr->indptr[layer1_end], eid_data);
    }
    return std::vector<IdArray>{idx, eid};
  } else {
    LOG(FATAL) << "unsupported adjacency matrix format";
    return std::vector<IdArray>();
  }
}

void ConstructNodeFlow(
    const std::vector<dgl_id_t> &neighbor_list,
    const std::vector<dgl_id_t> &edge_list,
    const std::vector<size_t> &layer_offsets,
    std::vector<dgl_id_t> *sub_vers,
    std::vector<neighbor_info> *neigh_pos,
    const std::string &edge_type,
    bool is_multigraph,
    NodeFlow *nf,
    std::vector<dgl_id_t> *vertex_mapping,
    std::vector<dgl_id_t> *edge_mapping) {
  uint64_t num_vertices = sub_vers->size();
  int64_t num_edges = static_cast<int64_t>(edge_list.size());
  bool edges_available = (num_edges > 0);
  int num_hops = layer_offsets.size() - 1;
  nf->node_mapping = IdArray::Empty({static_cast<int64_t>(num_vertices)},
                                   DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  nf->edge_mapping_available = edges_available;
  if (edges_available)
    nf->edge_mapping = IdArray::Empty({static_cast<int64_t>(num_edges)},
                                      DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  nf->layer_offsets = IdArray::Empty({static_cast<int64_t>(num_hops + 1)},
                                     DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  nf->flow_offsets = IdArray::Empty({static_cast<int64_t>(num_hops)},
                                    DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  vertex_mapping->resize(num_vertices);
  edge_mapping->resize(num_edges);

  dgl_id_t *node_map_data = static_cast<dgl_id_t *>(nf->node_mapping->data);
  dgl_id_t *layer_off_data = static_cast<dgl_id_t *>(nf->layer_offsets->data);
  dgl_id_t *flow_off_data = static_cast<dgl_id_t *>(nf->flow_offsets->data);
  dgl_id_t *edge_map_data = nullptr;
  if (edges_available)
    edge_map_data = static_cast<dgl_id_t *>(nf->edge_mapping->data);

  // Construct sub_csr_graph
  auto subg_csr = std::make_shared<ImmutableGraph::CSR>(num_vertices, num_edges);
  subg_csr->indices.resize(num_edges);
  subg_csr->edge_ids.resize(num_edges);
  dgl_id_t* col_list_out = subg_csr->indices.data();
  int64_t* indptr_out = subg_csr->indptr.data();
  size_t collected_nedges = 0;

  // The data from the previous steps:
  // * node data: sub_vers (vid, layer), neigh_pos,
  // * edge data: neighbor_list, edge_list, probability.
  // * layer_offsets: the offset in sub_vers.
  dgl_id_t ver_id = 0;
  std::vector<std::unordered_map<dgl_id_t, dgl_id_t>> layer_ver_maps;
  layer_ver_maps.resize(num_hops);
  size_t out_node_idx = 0;
  for (int layer_id = num_hops - 1; layer_id >= 0; layer_id--) {
    // We sort the vertices in a layer so that we don't need to sort the neighbor Ids
    // after remap to a subgraph.
    std::sort(sub_vers->begin() + layer_offsets[layer_id],
              sub_vers->begin() + layer_offsets[layer_id + 1]);

    // Save the sampled vertices and its layer Id.
    for (size_t i = layer_offsets[layer_id]; i < layer_offsets[layer_id + 1]; i++) {
      (*vertex_mapping)[out_node_idx] = i;
      node_map_data[out_node_idx] = sub_vers->at(i);
      layer_ver_maps[layer_id].insert(std::pair<dgl_id_t, dgl_id_t>(sub_vers->at(i),
                                                                    ver_id++));
      ++out_node_idx;
    }
  }
  CHECK(out_node_idx == num_vertices);

  // sampling algorithms have to start from the seed nodes, so the seed nodes are
  // in the first layer and the input nodes are in the last layer.
  // When we expose the sampled graph to a Python user, we say the input nodes
  // are in the first layer and the seed nodes are in the last layer.
  // Thus, when we copy sampled results to a CSR, we need to reverse the order of layers.
  size_t row_idx = 0;
  for (size_t i = layer_offsets[num_hops - 1]; i < layer_offsets[num_hops]; i++) {
    indptr_out[row_idx++] = 0;
  }
  layer_off_data[0] = 0;
  layer_off_data[1] = layer_offsets[num_hops] - layer_offsets[num_hops - 1];
  int out_layer_idx = 1;
  for (int layer_id = num_hops - 2; layer_id >= 0; layer_id--) {
    std::sort(neigh_pos->begin() + layer_offsets[layer_id],
              neigh_pos->begin() + layer_offsets[layer_id + 1],
              [](const neighbor_info &a1, const neighbor_info &a2) {
                return a1.id < a2.id;
              });

    for (size_t i = layer_offsets[layer_id]; i < layer_offsets[layer_id + 1]; i++) {
      dgl_id_t dst_id = sub_vers->at(i);
      CHECK_EQ(dst_id, neigh_pos->at(i).id);
      size_t pos = neigh_pos->at(i).pos;
      CHECK_LE(pos, neighbor_list.size());
      size_t num_edges = neigh_pos->at(i).num_edges;
      if (neighbor_list.empty()) CHECK_EQ(num_edges, 0);

      // We need to map the Ids of the neighbors to the subgraph.
      auto neigh_it = neighbor_list.begin() + pos;
      for (size_t i = 0; i < num_edges; i++) {
        dgl_id_t neigh = *(neigh_it + i);
        CHECK(layer_ver_maps[layer_id + 1].find(neigh) != layer_ver_maps[layer_id + 1].end());
        col_list_out[collected_nedges + i] = layer_ver_maps[layer_id + 1][neigh];
      }
      // We can simply copy the edge Ids.
      if (edges_available)
        std::copy_n(edge_list.begin() + pos,
                    num_edges, edge_map_data + collected_nedges);
      else
        std::fill(edge_map_data + collected_nedges,
                  edge_map_data + collected_nedges + num_edges,
                  -1);
      std::iota(
          edge_mapping->begin() + collected_nedges,
          edge_mapping->begin() + collected_nedges + num_edges,
          pos);
      collected_nedges += num_edges;
      indptr_out[row_idx+1] = indptr_out[row_idx] + num_edges;
      row_idx++;
    }
    layer_off_data[out_layer_idx + 1] = layer_off_data[out_layer_idx]
        + layer_offsets[layer_id + 1] - layer_offsets[layer_id];
    out_layer_idx++;
  }
  CHECK(row_idx == num_vertices);
  CHECK(indptr_out[row_idx] == num_edges);
  CHECK(out_layer_idx == num_hops);
  CHECK(layer_off_data[out_layer_idx] == num_vertices);

  // Copy flow offsets.
  flow_off_data[0] = 0;
  int out_flow_idx = 0;
  for (size_t i = 0; i < layer_offsets.size() - 2; i++) {
    size_t num_edges = subg_csr->GetDegree(layer_off_data[i + 1], layer_off_data[i + 2]);
    flow_off_data[out_flow_idx + 1] = flow_off_data[out_flow_idx] + num_edges;
    out_flow_idx++;
  }
  CHECK(out_flow_idx == num_hops - 1);
  CHECK(flow_off_data[num_hops - 1] == static_cast<uint64_t>(num_edges));

  for (size_t i = 0; i < subg_csr->edge_ids.size(); i++) {
    subg_csr->edge_ids[i] = i;
  }

  if (edge_type == "in") {
    nf->graph = GraphPtr(new ImmutableGraph(subg_csr, nullptr, is_multigraph));
  } else {
    nf->graph = GraphPtr(new ImmutableGraph(nullptr, subg_csr, is_multigraph));
  }

  nf->node_data_name = "";
  nf->edge_data_name = "";
}

DGL_REGISTER_GLOBAL("nodeflow._CAPI_NodeFlowGetGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* ptr = args[0];
    const NodeFlow* nflow = static_cast<NodeFlow*>(ptr);
    GraphInterface* gptr = nflow->graph->Reset();
    *rv = gptr;
  });

DGL_REGISTER_GLOBAL("nodeflow._CAPI_NodeFlowGetNodeMapping")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* ptr = args[0];
    const NodeFlow* nflow = static_cast<NodeFlow*>(ptr);
    *rv = nflow->node_mapping;
  });

DGL_REGISTER_GLOBAL("nodeflow._CAPI_NodeFlowEdgeMappingIsAvailable")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* ptr = args[0];
    const NodeFlow* nflow = static_cast<NodeFlow*>(ptr);
    *rv = nflow->edge_mapping_available;
  });

DGL_REGISTER_GLOBAL("nodeflow._CAPI_NodeFlowGetEdgeMapping")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* ptr = args[0];
    const NodeFlow* nflow = static_cast<NodeFlow*>(ptr);
    *rv = nflow->edge_mapping;
  });

DGL_REGISTER_GLOBAL("nodeflow._CAPI_NodeFlowGetLayerOffsets")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* ptr = args[0];
    const NodeFlow* nflow = static_cast<NodeFlow*>(ptr);
    *rv = nflow->layer_offsets;
  });

DGL_REGISTER_GLOBAL("nodeflow._CAPI_NodeFlowGetBlockOffsets")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* ptr = args[0];
    const NodeFlow* nflow = static_cast<NodeFlow*>(ptr);
    *rv = nflow->flow_offsets;
  });

DGL_REGISTER_GLOBAL("nodeflow._CAPI_NodeFlowFree")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* ptr = args[0];
    NodeFlow* nflow = static_cast<NodeFlow*>(ptr);
    delete nflow;
  });

DGL_REGISTER_GLOBAL("nodeflow._CAPI_NodeFlowNodeDataName")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* ptr = args[0];
    const NodeFlow* nflow = static_cast<NodeFlow*>(ptr);
    *rv = nflow->node_data_name;
  });

DGL_REGISTER_GLOBAL("nodeflow._CAPI_NodeFlowEdgeDataName")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* ptr = args[0];
    const NodeFlow* nflow = static_cast<NodeFlow*>(ptr);
    *rv = nflow->edge_data_name;
  });

DGL_REGISTER_GLOBAL("nodeflow._CAPI_NodeFlowGetNodeData")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* ptr = args[0];
    const NodeFlow* nflow = static_cast<NodeFlow*>(ptr);
    if (nflow->node_data_name.empty()) {
      LOG(FATAL) << "node data unavailable";
      *rv = nullptr;
    } else {
      *rv = nflow->node_data;
    }
  });

DGL_REGISTER_GLOBAL("nodeflow._CAPI_NodeFlowGetEdgeData")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* ptr = args[0];
    const NodeFlow* nflow = static_cast<NodeFlow*>(ptr);
    if (nflow->edge_data_name.empty()) {
      LOG(FATAL) << "edge data unavailable";
      *rv = nullptr;
    } else {
      *rv = nflow->edge_data;
    }
  });

}  // namespace dgl
