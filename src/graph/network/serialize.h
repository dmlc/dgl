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
 * \param data data buffer
 * \param csr subgraph csr
 * \param node_mapping node mapping in NodeFlowIndex
 * \param edge_mapping edge mapping in NodeFlowIndex
 * \param layer_offsets layer offsets in NodeFlowIndex
 * \param flow_offsets flow offsets in NodeFlowIndex
 * \return the total size of the serialized binary data
 */
int64_t SerializeSampledSubgraph(char* &data,
	                             const ImmutableGraph::CSR::Ptr csr,
	                             const IdArray& node_mapping,
	                             const IdArray& edge_mapping,
	                             const IdArray& layer_offsets,
	                             const IdArray& flow_offsets);

/*!
 * \brief Deserialize sampled subgraph from binary data
 * \param nf NodeFlow
 * \param data data buffer
 */
void DeserializeSampledSubgraph(NodeFlow* nf, char* data);

// TODO(chao): we can add compression and decompression method here

}  // namespace network
}  // namespace dgl

#endif  // DGL_GRAPH_NETWORK_SERIALIZE_H_