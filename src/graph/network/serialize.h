/*!
 *  Copyright (c) 2019 by Contributors
 * \file serialize.h
 * \brief Serialization for DGL distributed training.
 */
#ifndef DGL_GRAPH_NETWORK_SERIALIZE_H_
#define DGL_GRAPH_NETWORK_SERIALIZE_H_

#include <dgl/sampler.h>
#include <dgl/immutable_graph.h>

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
                                 const ImmutableGraph::CSR::Ptr csr,
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
                                ImmutableGraph::CSR::Ptr* csr,
                                IdArray* node_mapping,
                                IdArray* edge_mapping,
                                IdArray* layer_offsets,
                                IdArray* flow_offsets);

/*!
 * \brief Serialize IP address to binary data
 * \param data pointer of data buffer
 * \param ip IP address
 * \return the total size of the serialized binary data
 */
int64_t SerializeIP(char* data, const std::string& ip);

/*!
 * \brief Serialize port to binary data
 * \param data pointer of data buffer
 * \param port port number
 * \return the total size of the serialized binary data
 */
int64_t SerializePort(char* data, int port);

/*!
 * \brief Deserialize IP address from binary data
 * \param data pointer of data buffer
 * \param ip IP address
 */
void DeserializeIP(char* data, std::string* ip);

/*!
 * \brief Deserialize port from binary data
 * \param data pointer of data buffer
 * \param port port number
 */
void DeserializePort(char* data, int* port);

// TODO(chao): we can add compression and decompression method here

}  // namespace network
}  // namespace dgl

#endif  // DGL_GRAPH_NETWORK_SERIALIZE_H_
