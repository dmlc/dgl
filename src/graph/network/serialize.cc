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

const int kNumTensor = 7;

int64_t SerializeSampledSubgraph(char* data,
                                 const CSRPtr csr,
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
  int64_t indptr_size = csr->indptr().GetSize();
  int64_t indices_size = csr->indices().GetSize();
  int64_t edge_ids_size = csr->edge_ids().GetSize();
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
  dgl_id_t* indptr = static_cast<dgl_id_t*>(csr->indptr()->data);
  dgl_id_t* indices = static_cast<dgl_id_t*>(csr->indices()->data);
  dgl_id_t* edge_ids = static_cast<dgl_id_t*>(csr->edge_ids()->data);
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
                                CSRPtr* csr,
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
  // TODO(minjie): multigraph flag
  *csr = CSRPtr(new CSR(num_vertices, num_edges, false));
  // indices (CSR)
  tensor_size = *(reinterpret_cast<int64_t*>(data_ptr));
  data_ptr += sizeof(int64_t);
  dgl_id_t* col_list_out = static_cast<dgl_id_t*>((*csr)->indices()->data);
  memcpy(col_list_out, data_ptr, tensor_size);
  data_ptr += tensor_size;
  // edge_ids (CSR)
  tensor_size = *(reinterpret_cast<int64_t*>(data_ptr));
  data_ptr += sizeof(int64_t);
  dgl_id_t* edge_ids = static_cast<dgl_id_t*>((*csr)->edge_ids()->data);
  memcpy(edge_ids, data_ptr, tensor_size);
  data_ptr += tensor_size;
  // indptr (CSR)
  tensor_size = *(reinterpret_cast<int64_t*>(data_ptr));
  data_ptr += sizeof(int64_t);
  dgl_id_t* indptr_out = static_cast<dgl_id_t*>((*csr)->indptr()->data);
  memcpy(indptr_out, data_ptr, tensor_size);
  data_ptr += tensor_size;
}

int64_t SerializeKVMsg(char* data,
                       const int msg_type,
                       const int rank,
                       const std::string& name,
                       const NDArray* ID,
                       const NDArray* tensor) {
  int64_t total_size = 0;
  int64_t id_size = ID->GetSize();
  int64_t data_size = tensor->GetSize();
  int64_t id_shape = (*ID)->shape[0];
  int64_t data_shape_0 = 0;
  int64_t data_shape_1 = 0;
  if (tensor != nullptr) {
    data_shape_0 = (*tensor)->shape[0];
    data_shape_1 = (*tensor)->shape[1];
  }
  total_size += sizeof(msg_type);
  total_size += sizeof(rank);
  total_size += sizeof(name.length());
  total_size += name.length();
  total_size += id_size;
  total_size += sizeof(id_size);
  total_size += sizeof(id_shape);
  if (tensor != nullptr) {
    total_size += data_size;
    total_size += sizeof(data_size);
    total_size += sizeof(data_shape_0);
    total_size += sizeof(data_shape_1);
  }
  if (total_size > kMaxBufferSize) {
    LOG(FATAL) << "Message size: (" << total_size
               << ") is larger than buffer size: ("
               << kMaxBufferSize << ")";
  }
  char* data_ptr = data;
  // Write message type
  *(reinterpret_cast<int*>(data_ptr)) = msg_type;
  data_ptr += sizeof(msg_type);
  // Write rank
  *(reinterpret_cast<int*>(data_ptr)) = rank;
  data_ptr += sizeof(rank);
  // write name
  *(reinterpret_cast<size_t*>(data_ptr)) = name.length();
  data_ptr += sizeof(name.length());
  memcpy(data_ptr, name.data(), name.length());
  data_ptr += name.length();
  // Write id
  *(reinterpret_cast<int64_t*>(data_ptr)) = id_shape;
  data_ptr += sizeof(id_shape);
  *(reinterpret_cast<int64_t*>(data_ptr)) = id_size;
  data_ptr += sizeof(id_size);
  char* id_ptr = static_cast<char*>((*ID)->data);
  memcpy(data_ptr, id_ptr, id_size);
  data_ptr += id_size;
  // Write tensor
  *(reinterpret_cast<int64_t*>(data_ptr)) = data_shape_0;
  data_ptr += sizeof(data_shape_0);
  *(reinterpret_cast<int64_t*>(data_ptr)) = data_shape_1;
  data_ptr += sizeof(data_shape_1);
  *(reinterpret_cast<int64_t*>(data_ptr)) = data_size;
  data_ptr += sizeof(data_size);
  char* tensor_ptr = static_cast<char*>((*tensor)->data);
  memcpy(data_ptr, tensor_ptr, data_size);
  return total_size;
}

void DeserializeKVMsg(char* data, 
                      int* msg_type, 
                      int* rank, 
                      std::string* name, 
                      NDArray* ID, 
                      NDArray* tensor) {
  char* data_ptr = data;
  // Read message type
  *msg_type = *(reinterpret_cast<int*>(data_ptr));
  data_ptr += sizeof(int);
  // Read rank
  *rank = *(reinterpret_cast<int*>(data_ptr));
  data_ptr += sizeof(int);
  // Read name
  size_t name_size = *(reinterpret_cast<size_t*>(data_ptr));
  data_ptr += sizeof(name_size);
  name->assign(data_ptr, name_size);
  data_ptr += name_size;
  // Read id
  int64_t id_shape = *(reinterpret_cast<int64_t*>(data_ptr));
  data_ptr += sizeof(id_shape);
  int64_t id_size = *(reinterpret_cast<int64_t*>(data_ptr));
  data_ptr += sizeof(id_size);
  *ID = NDArray::Empty({id_shape}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  char* id_ptr = static_cast<char*>((*ID)->data);
  memcpy(id_ptr, data_ptr, id_size);
  data_ptr += id_size;
  // Read tensor
  int64_t data_shape_0 = *(reinterpret_cast<int64_t*>(data_ptr));
  data_ptr += sizeof(data_shape_0);
  int64_t data_shape_1 = *(reinterpret_cast<int64_t*>(data_ptr));
  data_ptr += sizeof(data_shape_1);
  int64_t data_size = *(reinterpret_cast<int64_t*>(data_ptr));
  data_ptr += sizeof(data_size);
  *tensor = NDArray::Empty({data_shape_0, data_shape_1}, DLDataType{kDLFloat, 32, 1}, DLContext{kDLCPU, 0});
  char* tensor_ptr = static_cast<char*>((*tensor)->data);
  memcpy(tensor_ptr, data_ptr, data_size);
}

}  // namespace network
}  // namespace dgl
