/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/network.h
 * \brief DGL networking related APIs
 */
#ifndef DGL_GRAPH_NETWORK_H_
#define DGL_GRAPH_NETWORK_H_

#include <dmlc/logging.h>
#include <dgl/runtime/ndarray.h>

namespace dgl {
namespace network {

using dgl::runtime::NDArray;

#define IS_SENDER    true
#define IS_RECEIVER  false

// TODO(chao): make these numbers configurable

// Each single message cannot larger than 300 MB
const int64_t kMaxBufferSize = 300 * 1024 * 2014;
// Size of message queue is 1 GB
const int64_t kQueueSize = 1024 * 1024 * 1024;
// Maximal try count of connection
const int kMaxTryCount = 500;

// Control number
const int CONTROL_NODEFLOW = 0;
const int CONTROL_END_SIGNAL = 1;

// KVStore message type
const int INIT_MSG = 1;
const int PUSH_MSG = 2;
const int PULL_MSG = 3;
const int PULL_BACK_MSG = 4;
const int FINAL_MSG = 5;

/*!
 * \brief C structure for holding DGL distributed kvstore message
 */
struct KVStoreMsg {
  /*!
   * \brief Message type
   */
  int msg_type;
  /*!
   * \brief Sender's ID
   */
  int rank;
   /*!
   * \brief data name
   */
  std::string name;
  /*!
   * \brief data ID
   */
  NDArray ID;
  /*!
   * \brief data matrix
   */
  NDArray data;
};

}  // namespace network
}  // namespace dgl

#endif  // DGL_GRAPH_NETWORK_H_
