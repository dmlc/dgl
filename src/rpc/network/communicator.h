/**
 *  Copyright (c) 2019 by Contributors
 * @file communicator.h
 * @brief Communicator for DGL distributed training.
 */
#ifndef DGL_RPC_NETWORK_COMMUNICATOR_H_
#define DGL_RPC_NETWORK_COMMUNICATOR_H_

#include <dmlc/logging.h>

#include <string>

#include "msg_queue.h"

namespace dgl {
namespace network {

/**
 * @brief Network Sender for DGL distributed training.
 *
 * Sender is an abstract class that defines a set of APIs for sending binary
 * data message over network. It can be implemented by different underlying
 * networking libraries such TCP socket and MPI. One Sender can connect to
 * multiple receivers and it can send data to specified receiver via receiver's
 * ID.
 */
class Sender {
 public:
  /**
   * @brief Sender constructor
   * @param queue_size size (bytes) of message queue.
   * @param max_thread_count size of thread pool. 0 for no limit
   * Note that, the queue_size parameter is optional.
   */
  explicit Sender(int64_t queue_size = 0, int max_thread_count = 0) {
    CHECK_GE(queue_size, 0);
    CHECK_GE(max_thread_count, 0);
    queue_size_ = queue_size;
    max_thread_count_ = max_thread_count;
  }

  virtual ~Sender() {}

  /**
   * @brief Send data to specified Receiver.
   * @param msg data message
   * @param recv_id receiver's ID
   * @return Status code
   *
   * (1) The send is non-blocking. There is no guarantee that the message has
   * been physically sent out when the function returns. (2) The communicator
   * will assume the responsibility of the given message. (3) The API is
   * multi-thread safe. (4) Messages sent to the same receiver are guaranteed to
   * be received in the same order. There is no guarantee for messages sent to
   * different receivers.
   */
  virtual STATUS Send(Message msg, int recv_id) = 0;

 protected:
  /**
   * @brief Size of message queue
   */
  int64_t queue_size_;
  /**
   * @brief Size of thread pool. 0 for no limit
   */
  int max_thread_count_;
};

/**
 * @brief Network Receiver for DGL distributed training.
 *
 * Receiver is an abstract class that defines a set of APIs for receiving binary
 * data message over network. It can be implemented by different underlying
 * networking libraries such as TCP socket and MPI. One Receiver can connect
 * with multiple Senders and it can receive data from multiple Senders
 * concurrently.
 */
class Receiver {
 public:
  /**
   * @brief Receiver constructor
   * @param queue_size size of message queue.
   * @param max_thread_count size of thread pool. 0 for no limit
   * Note that, the queue_size parameter is optional.
   */
  explicit Receiver(int64_t queue_size = 0, int max_thread_count = 0) {
    if (queue_size < 0) {
      LOG(FATAL) << "queue_size cannot be a negative number.";
    }
    CHECK_GE(max_thread_count, 0);
    queue_size_ = queue_size;
    max_thread_count_ = max_thread_count;
  }

  virtual ~Receiver() {}

  /**
   * @brief Recv data from Sender
   * @param msg pointer of data message
   * @param send_id which sender current msg comes from
   * @param timeout The timeout value in milliseconds. If zero, wait
   * indefinitely.
   * @return Status code
   *
   * (1) The Recv() API is thread-safe.
   * (2) Memory allocated by communicator but will not own it after the function
   * returns.
   */
  virtual STATUS Recv(Message* msg, int* send_id, int timeout = 0) = 0;

  /**
   * @brief Recv data from a specified Sender
   * @param msg pointer of data message
   * @param send_id sender's ID
   * @param timeout The timeout value in milliseconds. If zero, wait
   * indefinitely.
   * @return Status code
   *
   * (1) The RecvFrom() API is thread-safe.
   * (2) Memory allocated by communicator but will not own it after the function
   * returns.
   */
  virtual STATUS RecvFrom(Message* msg, int send_id, int timeout = 0) = 0;

 protected:
  /**
   * @brief Size of message queue
   */
  int64_t queue_size_;
  /**
   * @brief Size of thread pool. 0 for no limit
   */
  int max_thread_count_;
};

}  // namespace network
}  // namespace dgl

#endif  // DGL_RPC_NETWORK_COMMUNICATOR_H_
