/*!
 *  Copyright (c) 2019 by Contributors
 * \file serialize.cc
 * \brief Serialization for DGL distributed training.
 */
#include "serialize.h"

#include <dmlc/logging.h>
#include <dgl/immutable_graph.h>

#include <cstring>

#include "../network.h"

using dgl::runtime::NDArray;

namespace dgl {
namespace network {

const int kNumTensor = 7;  // We need to serialize 7 conponents (tensor) here

/*!
 * Round x + 1 up to \c roundto
 * Example:
 * roundup(1, 8) = 8
 * roundup(7, 8) = 8
 * roundup(8, 8) = 16
 */
inline int64_t roundup(int64_t x, int64_t roundto) {
  return (x / roundto + 1) * roundto;
}

int64_t SerializeNodeFlow(char* data,
                          const ImmutableGraph *graph,
                          const NodeFlow &nf) {
  const auto csr = graph->GetInCSR();
  int64_t total_size = 0;

  // First check the size of the data to prevent overflowing the buffer.

  // node_mapping
  int64_t node_mapping_size = nf.node_mapping->shape[0] * sizeof(dgl_id_t);
  total_size += sizeof(node_mapping_size);
  total_size += node_mapping_size;

  // layer_offsets
  int64_t layer_offsets_size = nf.layer_offsets->shape[0] * sizeof(dgl_id_t);
  total_size += sizeof(layer_offsets_size);
  total_size += layer_offsets_size;

  // flow_offsets
  int64_t flow_offsets_size = nf.flow_offsets->shape[0] * sizeof(dgl_id_t);
  total_size += sizeof(flow_offsets_size);
  total_size += flow_offsets_size;

  // edge_mapping
  int64_t edge_mapping_size =
    nf.edge_mapping_available ? nf.edge_mapping->shape[0] * sizeof(dgl_id_t) : 0;
  total_size += sizeof(edge_mapping_size);
  total_size += edge_mapping_size;

  // node_data, serialized as follows:
  // ndim
  // dtype (8-bytes aligned)
  // node data name (8-bytes aligned, null-terminated)
  // shape array
  // content
  int64_t node_ndim = !nf.node_data_name.empty() ? nf.node_data->ndim : 0;
  total_size += sizeof(node_ndim);
  int64_t node_data_size = 0, node_data_name_size = 0;
  if (!nf.node_data_name.empty()) {
    total_size += sizeof(int64_t);  // DLDataType
    node_data_name_size = roundup(nf.node_data_name.length(), 8);
    total_size += node_data_name_size;
    node_data_size = nf.node_data->dtype.bits / 8;

    for (int i = 0; i < node_ndim; ++i) {
      node_data_size *= nf.node_data->shape[i];
      total_size += sizeof(int64_t);
    }
  }
  total_size += node_data_size;

  // edge_data
  int64_t edge_ndim = !nf.edge_data_name.empty() ? nf.edge_data->ndim : 0;
  total_size += sizeof(edge_ndim);
  int64_t edge_data_size = 0, edge_data_name_size = 0;
  if (!nf.edge_data_name.empty()) {
    total_size += sizeof(int64_t);  // DLDataType
    edge_data_name_size = roundup(nf.edge_data_name.length(), 8);
    total_size += edge_data_name_size;
    edge_data_size = nf.edge_data->dtype.bits / 8;

    for (int i = 0; i < edge_ndim; ++i) {
      edge_data_size *= nf.edge_data->shape[i];
      total_size += sizeof(int64_t);
    }
  }
  total_size += edge_data_size;

  // graph
  int64_t indptr_size = csr->indptr.size() * sizeof(int64_t);
  total_size += sizeof(int64_t);
  total_size += indptr_size;
  int64_t indices_size = csr->indices.size() * sizeof(dgl_id_t);
  total_size += sizeof(int64_t);
  total_size += indices_size;
  int64_t edge_ids_size = csr->edge_ids.size() * sizeof(dgl_id_t);
  total_size += sizeof(int64_t);
  total_size += edge_ids_size;
  total_size += sizeof(int64_t);    // multigraph

  if (total_size > kMaxBufferSize) {
    LOG(FATAL) << "Message size: (" << total_size
               << ") is larger than buffer size: ("
               << kMaxBufferSize << ")";
  }

  // Then write the data into the buffer
  char *data_ptr = data;

  // node_mapping
  *(reinterpret_cast<int64_t *>(data_ptr)) = node_mapping_size;
  data_ptr += sizeof(node_mapping_size);
  dgl_id_t* node_map_data = static_cast<dgl_id_t*>(nf.node_mapping->data);
  memcpy(data_ptr, node_map_data, node_mapping_size);
  data_ptr += node_mapping_size;

  // layer_offsets
  *(reinterpret_cast<int64_t*>(data_ptr)) = layer_offsets_size;
  data_ptr += sizeof(layer_offsets_size);
  dgl_id_t* layer_off_data = static_cast<dgl_id_t*>(nf.layer_offsets->data);
  memcpy(data_ptr, layer_off_data, layer_offsets_size);
  data_ptr += layer_offsets_size;

  // flow_offsets
  *(reinterpret_cast<int64_t*>(data_ptr)) = flow_offsets_size;
  data_ptr += sizeof(flow_offsets_size);
  dgl_id_t* flow_off_data = static_cast<dgl_id_t*>(nf.flow_offsets->data);
  memcpy(data_ptr, flow_off_data, flow_offsets_size);
  data_ptr += flow_offsets_size;

  // edge_mapping
  *(reinterpret_cast<int64_t*>(data_ptr)) = edge_mapping_size;
  data_ptr += sizeof(edge_mapping_size);
  if (nf.edge_mapping_available) {
    dgl_id_t* edge_map_data = static_cast<dgl_id_t*>(nf.edge_mapping->data);
    memcpy(data_ptr, edge_map_data, edge_mapping_size);
  }
  data_ptr += edge_mapping_size;

  // node_data
  *(reinterpret_cast<int64_t*>(data_ptr)) = node_ndim;
  data_ptr += sizeof(node_ndim);
  if (!nf.node_data_name.empty()) {
    *(reinterpret_cast<DLDataType*>(data_ptr)) = nf.node_data->dtype;
    data_ptr += sizeof(int64_t);
    strncpy(data_ptr, nf.node_data_name.c_str(), node_data_name_size);
    data_ptr += node_data_name_size;
    for (int i = 0; i < node_ndim; ++i) {
      *(reinterpret_cast<int64_t*>(data_ptr)) = nf.node_data->shape[i];
      data_ptr += sizeof(int64_t);
    }
    memcpy(data_ptr, nf.node_data->data, node_data_size);
    data_ptr += node_data_size;
  }

  // edge_data
  *(reinterpret_cast<int64_t*>(data_ptr)) = edge_ndim;
  data_ptr += sizeof(edge_ndim);
  if (!nf.edge_data_name.empty()) {
    *(reinterpret_cast<DLDataType*>(data_ptr)) = nf.edge_data->dtype;
    data_ptr += sizeof(int64_t);
    strncpy(data_ptr, nf.edge_data_name.c_str(), edge_data_name_size);
    data_ptr += edge_data_name_size;
    for (int i = 0; i < edge_ndim; ++i) {
      *(reinterpret_cast<int64_t*>(data_ptr)) = nf.edge_data->shape[i];
      data_ptr += sizeof(int64_t);
    }
    memcpy(data_ptr, nf.edge_data->data, edge_data_size);
    data_ptr += edge_data_size;
  }

  // graph
  int64_t* indptr = static_cast<int64_t*>(csr->indptr.data());
  dgl_id_t* indices = static_cast<dgl_id_t*>(csr->indices.data());
  dgl_id_t* edge_ids = static_cast<dgl_id_t*>(csr->edge_ids.data());
  // indices (CSR)
  *(reinterpret_cast<int64_t*>(data_ptr)) = indices_size;
  data_ptr += sizeof(int64_t);
  memcpy(data_ptr, indices, indices_size);
  data_ptr += indices_size;
  // edge_ids (CSR)
  *(reinterpret_cast<int64_t*>(data_ptr)) = edge_ids_size;
  data_ptr += sizeof(int64_t);
  memcpy(data_ptr, edge_ids, edge_ids_size);
  data_ptr += edge_ids_size;
  // indptr (CSR)
  *(reinterpret_cast<int64_t*>(data_ptr)) = indptr_size;
  data_ptr += sizeof(int64_t);
  memcpy(data_ptr, indptr, indptr_size);
  data_ptr += indptr_size;
  // multigraph
  *(reinterpret_cast<int64_t*>(data_ptr)) = (graph->IsMultigraph() ? 1 : 0);
  data_ptr += sizeof(int64_t);
  return total_size;
}

void DeserializeNodeFlow(char* data, NodeFlow *nf) {
  // For each component, we first read its size at the
  // begining of the buffer and then read its binary data
  char* data_ptr = data;
  // node_mapping
  int64_t tensor_size = *(reinterpret_cast<int64_t*>(data_ptr));
  int64_t num_vertices = tensor_size / sizeof(int64_t);
  int64_t num_edges;

  data_ptr += sizeof(int64_t);
  nf->node_mapping = IdArray::Empty(
      {static_cast<int64_t>(num_vertices)},
      DLDataType{kDLInt, 64, 1},
      DLContext{kDLCPU, 0});
  dgl_id_t* node_map_data = static_cast<dgl_id_t*>(nf->node_mapping->data);
  memcpy(node_map_data, data_ptr, tensor_size);
  data_ptr += tensor_size;

  // layer offsets
  tensor_size = *(reinterpret_cast<int64_t*>(data_ptr));
  int64_t num_hops_add_one = tensor_size / sizeof(int64_t);
  data_ptr += sizeof(int64_t);
  nf->layer_offsets = IdArray::Empty(
      {static_cast<int64_t>(num_hops_add_one)},
      DLDataType{kDLInt, 64, 1},
      DLContext{kDLCPU, 0});
  dgl_id_t* layer_off_data = static_cast<dgl_id_t*>(nf->layer_offsets->data);
  memcpy(layer_off_data, data_ptr, tensor_size);
  data_ptr += tensor_size;

  // flow offsets
  tensor_size = *(reinterpret_cast<int64_t*>(data_ptr));
  int64_t num_hops = tensor_size / sizeof(int64_t);
  data_ptr += sizeof(int64_t);
  nf->flow_offsets = IdArray::Empty(
      {static_cast<int64_t>(num_hops)},
      DLDataType{kDLInt, 64, 1},
      DLContext{kDLCPU, 0});
  dgl_id_t* flow_off_data = static_cast<dgl_id_t*>(nf->flow_offsets->data);
  memcpy(flow_off_data, data_ptr, tensor_size);
  data_ptr += tensor_size;

  // edge_mapping
  tensor_size = *(reinterpret_cast<int64_t*>(data_ptr));
  data_ptr += sizeof(int64_t);
  if (tensor_size == 0) {
    nf->edge_mapping_available = false;
  } else {
    nf->edge_mapping_available = true;
    num_edges = tensor_size / sizeof(int64_t);
    nf->edge_mapping = IdArray::Empty(
        {static_cast<int64_t>(num_edges)},
        DLDataType{kDLInt, 64, 1},
        DLContext{kDLCPU, 0});
    dgl_id_t* edge_mapping_data = static_cast<dgl_id_t*>(nf->edge_mapping->data);
    memcpy(edge_mapping_data, data_ptr, tensor_size);
    data_ptr += tensor_size;
  }

  // node_data
  int64_t node_ndim = *(reinterpret_cast<int64_t*>(data_ptr));
  data_ptr += sizeof(int64_t);
  if (node_ndim == 0) {
    nf->node_data_name = "";
  } else {
    DLDataType *dtype_ptr = reinterpret_cast<DLDataType*>(data_ptr);
    data_ptr += sizeof(int64_t);

    nf->node_data_name = data_ptr;  // data_ptr points to null-terminated name string
    data_ptr += roundup(nf->node_data_name.length(), 8);

    int64_t *node_shape_ptr = reinterpret_cast<int64_t*>(data_ptr);
    int64_t node_data_size = dtype_ptr->bits / 8;
    data_ptr += node_ndim * sizeof(int64_t);
    std::vector<int64_t> node_data_shape(node_shape_ptr, node_shape_ptr + node_ndim);
    for (int i = 0; i < node_ndim; ++i)
      node_data_size *= node_shape_ptr[i];

    nf->node_data = NDArray::Empty(node_data_shape, *dtype_ptr, DLContext{kDLCPU, 0});
    memcpy(nf->node_data->data, data_ptr, node_data_size);
    data_ptr += node_data_size;
  }

  // edge_data
  int64_t edge_ndim = *(reinterpret_cast<int64_t*>(data_ptr));
  data_ptr += sizeof(int64_t);
  if (edge_ndim == 0) {
    nf->edge_data_name = "";
  } else {
    DLDataType *dtype_ptr = reinterpret_cast<DLDataType*>(data_ptr);
    data_ptr += sizeof(int64_t);

    nf->edge_data_name = data_ptr;  // data_ptr points to null-terminated name string
    data_ptr += roundup(nf->edge_data_name.length(), 8);

    int64_t *edge_shape_ptr = reinterpret_cast<int64_t*>(data_ptr);
    int64_t edge_data_size = dtype_ptr->bits / 8;
    data_ptr += edge_ndim * sizeof(int64_t);
    std::vector<int64_t> edge_data_shape(edge_shape_ptr, edge_shape_ptr + edge_ndim);
    for (int i = 0; i < edge_ndim; ++i)
      edge_data_size *= edge_shape_ptr[i];

    nf->edge_data = NDArray::Empty(edge_data_shape, *dtype_ptr, DLContext{kDLCPU, 0});
    memcpy(nf->edge_data->data, data_ptr, edge_data_size);
    data_ptr += edge_data_size;
  }

  // graph
  int64_t indices_size = *(reinterpret_cast<int64_t*>(data_ptr));
  num_edges = indices_size / sizeof(int64_t);
  data_ptr += sizeof(int64_t);
  dgl_id_t *indices = reinterpret_cast<dgl_id_t*>(data_ptr);
  data_ptr += indices_size;

  int64_t edge_ids_size = *(reinterpret_cast<int64_t*>(data_ptr));
  data_ptr += sizeof(int64_t);
  dgl_id_t *edge_ids = reinterpret_cast<dgl_id_t*>(data_ptr);
  data_ptr += edge_ids_size;

  int64_t indptr_size = *(reinterpret_cast<int64_t*>(data_ptr));
  data_ptr += sizeof(int64_t);
  int64_t *indptr = reinterpret_cast<int64_t*>(data_ptr);
  data_ptr += indptr_size;

  int64_t flags = *(reinterpret_cast<int64_t*>(data_ptr));
  data_ptr += sizeof(int64_t);
  bool multigraph = ((flags & 0x1) != 0);

  ImmutableGraph::CSR::Ptr csr = std::make_shared<ImmutableGraph::CSR>(
      num_vertices, num_edges);
  csr->indices.resize(num_edges);
  csr->edge_ids.resize(num_edges);
  memcpy(csr->indices.data(), indices, indices_size);
  memcpy(csr->edge_ids.data(), edge_ids, edge_ids_size);
  memcpy(csr->indptr.data(), indptr, indptr_size);

  nf->graph = GraphPtr(new ImmutableGraph(csr, nullptr, multigraph));
}

}  // namespace network
}  // namespace dgl
