/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/network.h
 * \brief DGL networking related APIs
 */
#ifndef DGL_GRAPH_NETWORK_H_
#define DGL_GRAPH_NETWORK_H_

#include <dmlc/logging.h>
#include <dgl/runtime/ndarray.h>

#include <string.h>
#include <vector>

#include "../c_api_common.h"
#include "./network/msg_queue.h"

using dgl::runtime::NDArray;

namespace dgl {
namespace network {

// Max size of message queue for communicator is 200 MB
// TODO(chao): Make this number configurable
const int64_t kQueueSize = 200 * 1024 * 1024;

/*!
 * \brief Free memory buffer of NodeFlow
 */
inline void NDArrayDeleter(Message* msg) {
  delete reinterpret_cast<NDArray*>(msg->aux_handler);
}

/*!
 * \brief Message type for DGL distributed training
 */
enum MessageType {
 /*!
  * \brief Message for send/recv NodeFlow
  */
  kNodeFlowMsg = 0,
 /*!
  * \brief Message for end-signal
  */
  kEndMsg = 1
};

/*!
 * \brief Meta data for communicator message
 */
class MsgMeta {
 public:
  /*!
   * \brief MsgMeta constructor.
   * \param msg_type type of message
   */
  explicit MsgMeta(int msg_type)
  : msg_type_(msg_type), ndarray_count_(0) {}

  /*!
   * \brief Construct MsgMeta from binary data buffer.
   * \param buffer data buffer
   * \param size data size
   */
  MsgMeta(char* buffer, int64_t size) {
    CHECK_NOTNULL(buffer);
    this->Deserialize(buffer, size);
  }

  /*!
   * \return message type
   */
  inline int msg_type() const {
    return msg_type_;
  }

  /*!
   * \return count of ndarray
   */
  inline int ndarray_count() const {
    return ndarray_count_;
  }

  /*!
   * \brief Add NDArray meta data to MsgMeta
   * \param array DGL NDArray
   */
  void AddArray(const NDArray& array);

  /*!
   * \brief Serialize MsgMeta to data buffer
   * \param size size of serialized message
   * \return pointer of data buffer
   */
  char* Serialize(int64_t* size);

  /*!
   * \brief Deserialize MsgMeta from data buffer
   * \param buffer data buffer
   * \param size size of data buffer
   */
  void Deserialize(char* buffer, int64_t size);

  /*!
   * \brief type of message
   */
  int msg_type_;

  /*!
   * \brief count of ndarray in MetaMsg
   */
  int ndarray_count_;

  /*!
   * \brief We first write the ndim to data_shape_ 
   * and then write the data shape. 
   */
  std::vector<int64_t> data_shape_;
};


}  // namespace network
}  // namespace dgl

#endif  // DGL_GRAPH_NETWORK_H_
