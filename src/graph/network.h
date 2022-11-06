/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/network.h
 * \brief DGL networking related APIs
 */
#ifndef DGL_GRAPH_NETWORK_H_
#define DGL_GRAPH_NETWORK_H_

#include <dgl/runtime/ndarray.h>
#include <dmlc/logging.h>
#include <string.h>

#include <string>
#include <vector>

#include "../c_api_common.h"
#include "../rpc/network/msg_queue.h"

using dgl::runtime::NDArray;

namespace dgl {
namespace network {

/*!
 * \brief Create NDArray from raw data
 */
NDArray CreateNDArrayFromRaw(
    std::vector<int64_t> shape, DGLDataType dtype, DGLContext ctx, void* raw);

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
  kFinalMsg = 1,
  /*!
   * \brief Initialize KVStore
   */
  kInitMsg = 2,
  /*!
   * \brief Push msg to KVStore
   */
  kPushMsg = 3,
  /*!
   * \brief Pull msg from KVStore
   */
  kPullMsg = 4,
  /*!
   * \brief PullBack msg from KVStore
   */
  kPullBackMsg = 5,
  /*!
   * \brief Barrier msg for KVStore
   */
  kBarrierMsg = 6,
  /*!
   * \brief IP and ID msg for KVStore
   */
  kIPIDMsg = 7,
  /*!
   * \brief Get data shape msg for KVStore
   */
  kGetShapeMsg = 8,
  /*!
   * \brief Get data shape back msg for KVStore
   */
  kGetShapeBackMsg = 9
};

/*!
 * \brief Meta data for NDArray message
 */
class ArrayMeta {
 public:
  /*!
   * \brief ArrayMeta constructor.
   * \param msg_type type of message
   */
  explicit ArrayMeta(int msg_type) : msg_type_(msg_type), ndarray_count_(0) {}

  /*!
   * \brief Construct ArrayMeta from binary data buffer.
   * \param buffer data buffer
   * \param size data size
   */
  ArrayMeta(char* buffer, int64_t size) {
    CHECK_NOTNULL(buffer);
    this->Deserialize(buffer, size);
  }

  /*!
   * \return message type
   */
  inline int msg_type() const { return msg_type_; }

  /*!
   * \return count of ndarray
   */
  inline int ndarray_count() const { return ndarray_count_; }

  /*!
   * \brief Add NDArray meta data to ArrayMeta
   * \param array DGL NDArray
   */
  void AddArray(const NDArray& array);

  /*!
   * \brief Serialize ArrayMeta to data buffer
   * \param size size of serialized message
   * \return pointer of data buffer
   */
  char* Serialize(int64_t* size);

  /*!
   * \brief Deserialize ArrayMeta from data buffer
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
   * \brief DataType for each NDArray
   */
  std::vector<DGLDataType> data_type_;

  /*!
   * \brief We first write the ndim to data_shape_
   * and then write the data shape.
   */
  std::vector<int64_t> data_shape_;
};

/*!
 * \brief C structure for holding DGL KVServer message
 */
class KVStoreMsg {
 public:
  /*!
   * \brief KVStoreMsg constructor.
   */
  KVStoreMsg() {}

  /*!
   * \brief Construct KVStoreMsg from binary data buffer.
   * \param buffer data buffer
   * \param size data size
   */
  KVStoreMsg(char* buffer, int64_t size) {
    CHECK_NOTNULL(buffer);
    this->Deserialize(buffer, size);
  }
  /*!
   * \brief Serialize KVStoreMsg to data buffer
   *  Note that we don't serialize ID and data here.
   * \param size size of serialized message
   * \return pointer of data buffer
   */
  char* Serialize(int64_t* size);

  /*!
   * \brief Deserialize KVStoreMsg from data buffer
   * \param buffer data buffer
   * \param size size of data buffer
   */
  void Deserialize(char* buffer, int64_t size);

  /*!
   * \brief Message type of kvstore
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
  NDArray id;
  /*!
   * \brief data matrix
   */
  NDArray data;
  /*!
   * \brief data shape
   */
  NDArray shape;
};

}  // namespace network
}  // namespace dgl

#endif  // DGL_GRAPH_NETWORK_H_
