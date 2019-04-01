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

namespace dgl {
namespace network {

const int kNumTensor = 7;  // We need to serialize 7 conponents (tensor) here

int64_t SerializeSampledSubgraph(char* data,
                                 const ImmutableGraph::CSR::Ptr csr,
                                 const IdArray& node_mapping,
                                 const IdArray& edge_mapping,
                                 const IdArray& layer_offsets,
                                 const IdArray& flow_offsets) {
  int64_t total_size = 0;
  // For each component, we first write its size at the
  // begining of the buffer and then write its binary data
  int64_t node_mapping_size = node_mapping->shape[0] * sizeof(dgl_id_t);
  int64_t edge_mapping_size = edge_mapping->shape[0] * sizeof(dgl_id_t);
  int64_t layer_offsets_size = layer_offsets->shape[0] * sizeof(dgl_id_t);
  int64_t flow_offsets_size = flow_offsets->shape[0] * sizeof(dgl_id_t);
  int64_t indptr_size = csr->indptr.size() * sizeof(int64_t);
  int64_t indices_size = csr->indices.size() * sizeof(dgl_id_t);
  int64_t edge_ids_size = csr->edge_ids.size() * sizeof(dgl_id_t);
  total_size += node_mapping_size;
  total_size += edge_mapping_size;
  total_size += layer_offsets_size;
  total_size += flow_offsets_size;
  total_size += indptr_size;
  total_size += indices_size;
  total_size += edge_ids_size;
  total_size += kNumTensor * sizeof(int64_t);
  if (total_size > kMaxBufferSize) {
    LOG(FATAL) << "Message size: (" << total_size
               << ") is larger than buffer size: ("
               << kMaxBufferSize << ")";
  }
  // Write binary data to buffer
  char* data_ptr = data;
  dgl_id_t* node_map_data = static_cast<dgl_id_t*>(node_mapping->data);
  dgl_id_t* edge_map_data = static_cast<dgl_id_t*>(edge_mapping->data);
  dgl_id_t* layer_off_data = static_cast<dgl_id_t*>(layer_offsets->data);
  dgl_id_t* flow_off_data = static_cast<dgl_id_t*>(flow_offsets->data);
  int64_t* indptr = static_cast<int64_t*>(csr->indptr.data());
  dgl_id_t* indices = static_cast<dgl_id_t*>(csr->indices.data());
  dgl_id_t* edge_ids = static_cast<dgl_id_t*>(csr->edge_ids.data());
  // node_mapping
  *(reinterpret_cast<int64_t*>(data_ptr)) = node_mapping_size;
  data_ptr += sizeof(int64_t);
  memcpy(data_ptr, node_map_data, node_mapping_size);
  data_ptr += node_mapping_size;
  // layer_offsets
  *(reinterpret_cast<int64_t*>(data_ptr)) = layer_offsets_size;
  data_ptr += sizeof(int64_t);
  memcpy(data_ptr, layer_off_data, layer_offsets_size);
  data_ptr += layer_offsets_size;
  // flow_offsets
  *(reinterpret_cast<int64_t*>(data_ptr)) = flow_offsets_size;
  data_ptr += sizeof(int64_t);
  memcpy(data_ptr, flow_off_data, flow_offsets_size);
  data_ptr += flow_offsets_size;
  // edge_mapping
  *(reinterpret_cast<int64_t*>(data_ptr)) = edge_mapping_size;
  data_ptr += sizeof(int64_t);
  memcpy(data_ptr, edge_map_data, edge_mapping_size);
  data_ptr += edge_mapping_size;
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
  return total_size;
}

void DeserializeSampledSubgraph(char* data,
                                ImmutableGraph::CSR::Ptr* csr,
                                IdArray* node_mapping,
                                IdArray* edge_mapping,
                                IdArray* layer_offsets,
                                IdArray* flow_offsets) {
  // For each component, we first read its size at the
  // begining of the buffer and then read its binary data
  char* data_ptr = data;
  // node_mapping
  int64_t tensor_size = *(reinterpret_cast<int64_t*>(data_ptr));
  int64_t num_vertices = tensor_size / sizeof(int64_t);
  data_ptr += sizeof(int64_t);
  *node_mapping = IdArray::Empty({static_cast<int64_t>(num_vertices)},
                                  DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  dgl_id_t* node_map_data = static_cast<dgl_id_t*>((*node_mapping)->data);
  memcpy(node_map_data, data_ptr, tensor_size);
  data_ptr += tensor_size;
  // layer offsets
  tensor_size = *(reinterpret_cast<int64_t*>(data_ptr));
  int64_t num_hops_add_one = tensor_size / sizeof(int64_t);
  data_ptr += sizeof(int64_t);
  *layer_offsets = IdArray::Empty({static_cast<int64_t>(num_hops_add_one)},
                                   DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  dgl_id_t* layer_off_data = static_cast<dgl_id_t*>((*layer_offsets)->data);
  memcpy(layer_off_data, data_ptr, tensor_size);
  data_ptr += tensor_size;
  // flow offsets
  tensor_size = *(reinterpret_cast<int64_t*>(data_ptr));
  int64_t num_hops = tensor_size / sizeof(int64_t);
  data_ptr += sizeof(int64_t);
  *flow_offsets = IdArray::Empty({static_cast<int64_t>(num_hops)},
                                  DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  dgl_id_t* flow_off_data = static_cast<dgl_id_t*>((*flow_offsets)->data);
  memcpy(flow_off_data, data_ptr, tensor_size);
  data_ptr += tensor_size;
  // edge_mapping
  tensor_size = *(reinterpret_cast<int64_t*>(data_ptr));
  int64_t num_edges = tensor_size / sizeof(int64_t);
  data_ptr += sizeof(int64_t);
  *edge_mapping = IdArray::Empty({static_cast<int64_t>(num_edges)},
                                  DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  dgl_id_t* edge_mapping_data = static_cast<dgl_id_t*>((*edge_mapping)->data);
  memcpy(edge_mapping_data, data_ptr, tensor_size);
  data_ptr += tensor_size;
  // Construct sub_csr_graph
  *csr = std::make_shared<ImmutableGraph::CSR>(num_vertices, num_edges);
  (*csr)->indices.resize(num_edges);
  (*csr)->edge_ids.resize(num_edges);
  // indices (CSR)
  tensor_size = *(reinterpret_cast<int64_t*>(data_ptr));
  data_ptr += sizeof(int64_t);
  dgl_id_t* col_list_out = (*csr)->indices.data();
  memcpy(col_list_out, data_ptr, tensor_size);
  data_ptr += tensor_size;
  // edge_ids (CSR)
  tensor_size = *(reinterpret_cast<int64_t*>(data_ptr));
  data_ptr += sizeof(int64_t);
  dgl_id_t* edge_ids = (*csr)->edge_ids.data();
  memcpy(edge_ids, data_ptr, tensor_size);
  data_ptr += tensor_size;
  // indptr (CSR)
  tensor_size = *(reinterpret_cast<int64_t*>(data_ptr));
  data_ptr += sizeof(int64_t);
  int64_t* indptr_out = (*csr)->indptr.data();
  memcpy(indptr_out, data_ptr, tensor_size);
  data_ptr += tensor_size;
}

}  // namespace network
}  // namespace dgl
