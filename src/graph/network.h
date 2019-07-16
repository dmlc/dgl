/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/network.h
 * \brief DGL networking related APIs
 */
#ifndef DGL_GRAPH_NETWORK_H_
#define DGL_GRAPH_NETWORK_H_

#include <dmlc/logging.h>

#include <vector>

namespace dgl {
namespace network {

// Max size of message queue for communicator is 200 MB
// TODO(chao): Make this number configurable
const int64_t kQueueSize = 200 * 1024 * 1024;

// Control number
const int NF_MSG = 0;
const int END_MSG = 1;

/*!
 * \brief Meta data for communicator message
 */
class Message {
 public:
  /*!
   * \brief Message constructor. Construct message from binary buffer.
   * \param buffer data buffer
   * \param size data size
   *
   * Note that current implementation simplfy the NDArray data structure, 
   * which can be extended in the future. 
   */
  Message(char* buffer, int64_t size) { 
    this->Deserialize(buffer, size);
  }

  /*!
   * \brief Message constructor
   * \param msg_type type of message, e.g., NF_MSG or END_MSG
   *
   * Note that current implementation simplfy the NDArray data structure, 
   * which can be extended in the future. 
   */
  Message(int msg_type) : msg_type_(msg_type), ndarray_count_(0) { }

  /*!
   * \brief Add shape of new ndarray to current message
   * \param shape shape of ndarray
   */
  void AddShape(std::vector<int64_t> shape) {
    CHECK_NE(shape.empty(), true);
    for (int i = 0; i < shape.size(); ++i) {
      data_shape_.push_back(shape[i]);
    }
    ndarray_count_++;
  }

  /*!
   * \brief Serialize message to data buffer
   * \param size size of serialized message
   * \return data buffer
   */
  char* Serialize(int64_t* size) {
    int64_t buffer_size = 0;
    buffer_size += sizeof(msg_type_);
    if (ndarray_count_ != 0) {
      buffer_size += sizeof(ndarray_count_);
      buffer_size += sizeof(data_shape_.size());
      buffer_size += sizeof(int64_t) * data_shape_.size();
    }
    char* buffer = nullptr;
    try {
      buffer = new char[buffer_size];
    } catch (const std::bad_alloc&) {
      LOG(FATAL) << "Not enough memory for buffer: " << buffer_size;
    }
    // Write msg_type_
    *(reinterpret_cast<int*>(buffer)) = msg_type_;
    LOG(INFO) << "msg_type_: " << *(reinterpret_cast<int*>(buffer));
    buffer += sizeof(msg_type_);
    if (ndarray_count_ != 0) {
      // Write ndarray_count_
      *(reinterpret_cast<int*>(buffer)) = ndarray_count_;
      LOG(INFO) << "ndarray_count_: " << *(reinterpret_cast<int*>(buffer));
      buffer += sizeof(ndarray_count_);
      // Write size of data_shape_
      *(reinterpret_cast<size_t*>(buffer)) = data_shape_.size();
      LOG(INFO) << "size(): " << *(reinterpret_cast<size_t*>(buffer));
      buffer += sizeof(data_shape_.size());
      // Write data of data_shape_
      memcpy(buffer, 
    	reinterpret_cast<char*>(data_shape_.data()),
    	sizeof(int64_t) * data_shape_.size());
    }
    LOG(INFO) << "------------------------";
    *size = buffer_size;
    return buffer;
  }

  /*!
   * \brief Deserialize message from data buffer
   * \param buffer binary data buffer
   * \param size size of data buffer
   */
  void Deserialize(char* buffer, int64_t size) {
  	int64_t data_size = 0;
  	// Read mesg_type_
    msg_type_ = *(reinterpret_cast<int*>(buffer));
    LOG(INFO) << "recv msg_type_: " << msg_type_;
    buffer += sizeof(int);
    data_size += sizeof(int);
    if (data_size < size) {
      // Read ndarray_count_
      ndarray_count_ = *(reinterpret_cast<int*>(buffer));
      buffer += sizeof(int);
      data_size += sizeof(int);
      LOG(INFO) << "recv ndarray_count_: " << ndarray_count_;
      // Read size of data_shape_
      size_t count = *(reinterpret_cast<size_t*>(buffer));
      LOG(INFO) << "recv size(): " << count;
      buffer += sizeof(size_t);
      data_size += sizeof(size_t);
      data_shape_.resize(count);
      // Read data of data_shape_
      memcpy(data_shape_.data(), 
    	buffer, count * sizeof(int64_t));
      data_size += count * sizeof(int64_t);
      LOG(INFO) << "data size: " << count * sizeof(int64_t);
    }
    CHECK_EQ(data_size, size);
  }

  /*!
   * \return message type
   */
  int Type() const {
    return msg_type_;
  }

  /*!
   * \brief type of message, e.g., NF_MSG or END_MSG
   */
  int msg_type_;

  /*!
   * \brief count of ndarray in the following messages
   */
  int ndarray_count_;

  /*!
   * \brief data_count_.size() == ndarray_count_
   */
  std::vector<int64_t> data_shape_;
};


}  // namespace network
}  // namespace dgl

#endif  // DGL_GRAPH_NETWORK_H_
