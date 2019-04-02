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
 * \param graph the nodeflow graph
 * \param nf the nodeflow structure itself, except for the graph
 * \return the total size of the serialized binary data
 */
int64_t SerializeNodeFlow(char *data,
                          const ImmutableGraph *graph,
                          const NodeFlow &nf);

/*!
 * \brief Deserialize sampled subgraph from binary data
 * \param data pointer of data buffer
 * \param[out] nf output NodeFlow object
 */
void DeserializeNodeFlow(char* data, NodeFlow *nf);

// TODO(chao): we can add compression and decompression method here

}  // namespace network
}  // namespace dgl

#endif  // DGL_GRAPH_NETWORK_SERIALIZE_H_
