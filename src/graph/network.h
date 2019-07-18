/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/network.h
 * \brief DGL networking related APIs
 */
#ifndef DGL_GRAPH_NETWORK_H_
#define DGL_GRAPH_NETWORK_H_

#include <dmlc/logging.h>

#include "../c_api_common.h"

using dgl::runtime::NDArray;

namespace dgl {
namespace network {

// Max size of message queue for communicator is 200 MB
// TODO(chao): Make this number configurable
const int64_t kQueueSize = 200 * 1024 * 1024;

// Message type
const int NF_MSG = 0;
const int END_MSG = 1;

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
   * \brief MsgMeta constructor. Construct MsgMeta from binary data buffer.
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
  int Type() const {
    return msg_type_;
  }

  /*!
   * \return count of ndarray
   */
  int NDArrayCount() const {
    return ndarray_count_;
  }

  /*!
   * \brief Add shape of ndarray to MsgMeta
   * \param array NDArray
   */
  void AddArray(const NDArray& array) {
    // We first write the ndim to the data_shape_
    data_shape_.push_back(static_cast<int64_t>(array->ndim));
    // Then we write the data shape
    for (int i = 0; i < array->ndim; ++i) {
      data_shape_.push_back(array->shape[i]);
    }
    ndarray_count_++;
  }

  /*!
   * \brief Serialize MsgMeta to data buffer
   * \param size size of serialized message
   * \return pointer of data buffer
   */
  char* Serialize(int64_t* size) {
  	char* buffer = nullptr;
    int64_t buffer_size = 0;
    buffer_size += sizeof(msg_type_);
    if (ndarray_count_ != 0) {
      buffer_size += sizeof(ndarray_count_);
      buffer_size += sizeof(data_shape_.size());
      buffer_size += sizeof(int64_t) * data_shape_.size();
    }
    buffer = new char[buffer_size];
    char* pointer = buffer;
    // Write msg_type_
    *(reinterpret_cast<int*>(pointer)) = msg_type_;
    pointer += sizeof(msg_type_);
    if (ndarray_count_ != 0) {
      // Write ndarray_count_
      *(reinterpret_cast<int*>(pointer)) = ndarray_count_;
      pointer += sizeof(ndarray_count_);
      // Write size of data_shape_
      *(reinterpret_cast<size_t*>(pointer)) = data_shape_.size();
      pointer += sizeof(data_shape_.size());
      // Write data of data_shape_
      memcpy(pointer, 
        reinterpret_cast<char*>(data_shape_.data()),
        sizeof(int64_t) * data_shape_.size());
    }
    *size = buffer_size;
    return buffer;
  }

  /*!
   * \brief Deserialize MsgMeta from data buffer
   * \param buffer data buffer
   * \param size size of data buffer
   */
  void Deserialize(char* buffer, int64_t size) {
    int64_t data_size = 0;
    // Read mesg_type_
    msg_type_ = *(reinterpret_cast<int*>(buffer));
    buffer += sizeof(int);
    data_size += sizeof(int);
    if (data_size < size) {
      // Read ndarray_count_
      ndarray_count_ = *(reinterpret_cast<int*>(buffer));
      buffer += sizeof(int);
      data_size += sizeof(int);
      // Read size of data_shape_
      size_t count = *(reinterpret_cast<size_t*>(buffer));
      buffer += sizeof(size_t);
      data_size += sizeof(size_t);
      data_shape_.resize(count);
      // Read data of data_shape_
      memcpy(data_shape_.data(), buffer, 
        count * sizeof(int64_t));
      data_size += count * sizeof(int64_t);
    }
    CHECK_EQ(data_size, size);
  }

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
