/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/network.h
 * \brief DGL networking related APIs
 */
#ifndef DGL_GRAPH_NETWORK_H_
#define DGL_GRAPH_NETWORK_H_

namespace dgl {
namespace network {

#define IS_SENDER   true
#define IS_RECEIVER false

// TODO(chao): make these number configurable

// Message cannot larger than 300 MB
const long long kMaxBufferSize = 300 * 1024 * 2014;
// Size of message queue is 2 GB
const long long kQueueSize = 2147483648;

}  // namespace network
}  // namespace dgl

#endif  // DGL_GRAPH_NETWORK_H_