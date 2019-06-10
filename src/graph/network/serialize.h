/*!
 *  Copyright (c) 2019 by Contributors
 * \file serialize.h
 * \brief Serialization for DGL distributed training.
 */
#ifndef DGL_GRAPH_NETWORK_SERIALIZE_H_
#define DGL_GRAPH_NETWORK_SERIALIZE_H_

#include <dgl/sampler.h>
#include <dgl/immutable_graph.h>

using dgl::runtime::NDArray;

namespace dgl {
namespace network {

/*!
 * \brief Serialize sampled subgraph to binary data
 * \param data pointer of data buffer
 * \param csr subgraph csr
 * \param node_mapping node mapping in NodeFlowIndex
 * \param edge_mapping edge mapping in NodeFlowIndex
 * \param layer_offsets layer offsets in NodeFlowIndex
 * \param flow_offsets flow offsets in NodeFlowIndex
 * \return the total size of the serialized binary data
 */
int64_t SerializeSampledSubgraph(char* data,
                                 const CSRPtr csr,
                                 const IdArray& node_mapping,
                                 const IdArray& edge_mapping,
                                 const IdArray& layer_offsets,
                                 const IdArray& flow_offsets);

/*!
 * \brief Deserialize sampled subgraph from binary data
 * \param data pointer of data buffer
 * \param csr subgraph csr
 * \param node_mapping node mapping in NodeFlowIndex
 * \param edge_mapping edge mapping in NodeFlowIndex
 * \param layer_offsets layer offsets in NodeFlowIndex
 * \param flow_offsets flow offsets in NodeFlowIndex
 */
void DeserializeSampledSubgraph(char* data,
                                CSRPtr* csr,
                                IdArray* node_mapping,
                                IdArray* edge_mapping,
                                IdArray* layer_offsets,
                                IdArray* flow_offsets);
/*!
 * \brief Serialize KVStoreMsg to binary data
 * \param data pointer of data buffer
 * \param msg_type message type
 * \param rank sender's ID
 * \param name data name
 * \param tensor data NDArray
 */
int64_t SerializeKVMsg(char* data,
                       const int msg_type,
                       const int rank,
                       const std::string& name,
                       const NDArray* ID,
                       const NDArray* tensor);
/*!
 * \brief Deserialize KVStoreMsg from binary data
 * \param data pointer of data buffer
 * \param msg_type message type
 * \param rank sender's ID
 * \param name data name
 * \param tensor data NDArray
 */
void DeserializeKVMsg(char* data, 
                      int* msg_type, 
                      int* rank, 
                      std::string* name, 
                      NDArray* ID, 
                      NDArray* tensor);

}  // namespace network
}  // namespace dgl

#endif  // DGL_GRAPH_NETWORK_SERIALIZE_H_
