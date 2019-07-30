/*!
 *  Copyright (c) 2019 by Contributors
 * \file communicator.h
 * \brief Communicator for DGL distributed training.
 */
#ifndef DGL_GRAPH_NETWORK_COMMUNICATOR_H_
#define DGL_GRAPH_NETWORK_COMMUNICATOR_H_

#include <dmlc/logging.h>

#include <string>

#include "msg_queue.h"

namespace dgl {
namespace network {

/*!
 * \brief Network Sender for DGL distributed training.
 *
 * Sender is an abstract class that defines a set of APIs for sending binary 
 * data message over network. It can be implemented by different underlying 
 * networking libraries such TCP socket and MPI. One Sender can connect to 
 * multiple receivers and it can send data to specified receiver via receiver's ID.
 */
class Sender {
 public:
  /*!
   * \brief Sender constructor
   * \param queue_size size (bytes) of message queue. 
   * Note that, the queue_size parameter is optional.
   */
  explicit Sender(int64_t queue_size = 0) {
    CHECK_GE(queue_size, 0);
    queue_size_ = queue_size;
  }

  virtual ~Sender() {}

  /*!
   * \brief Add receiver's address and ID to the sender's namebook
   * \param addr Networking address, e.g., 'socket://127.0.0.1:50091', 'mpi://0'
   * \param id receiver's ID
   */
  virtual void AddReceiver(const char* addr, int id) = 0;

  /*!
   * \brief Connect with all the Receivers
   * \return True for success and False for fail
   */
  virtual bool Connect() = 0;

  /*!
   * \brief Send data to specified Receiver.
   * \param msg data message
   * \param recv_id receiver's ID
   * \return bytes of data
   *   > 0 : bytes we sent
   *   - 1 : error
   */
  virtual int64_t Send(Message msg, int recv_id) = 0;

  /*!
   * \brief Finalize Sender
   */
  virtual void Finalize() = 0;

  /*!
   * \brief Communicator type: 'socket', 'mpi', etc.
   */
  virtual std::string Type() const = 0;

 protected:
  /*!
   * \brief Size of message queue
   */
  int64_t queue_size_;
};

/*!
 * \brief Network Receiver for DGL distributed training.
 *
 * Receiver is an abstract class that defines a set of APIs for receiving binary data 
 * message over network. It can be implemented by different underlying networking 
 * libraries such as TCP socket and MPI. One Receiver can connect with multiple Senders 
 * and it can receive data from multiple Senders concurrently.
 */
class Receiver {
 public:
  /*!
   * \brief Receiver constructor
   * \param queue_size size of message queue.
   * Note that, the queue_size parameter is optional.
   */
  explicit Receiver(int64_t queue_size = 0) {
    if (queue_size < 0) {
      LOG(FATAL) << "queue_size cannot be a negative number.";
    }
    queue_size_ = queue_size;
  }

  virtual ~Receiver() {}

  /*!
   * \brief Wait for all the Senders to connect
   * \param addr Networking address, e.g., 'socket://127.0.0.1:50051', 'mpi://0'
   * \param num_sender total number of Senders
   * \return True for success and False for fail
   */
  virtual bool Wait(const char* addr, int num_sender) = 0;

  /*!
   * \brief Recv data from Sender
   * \param msg pointer of data message
   * \param send_id which sender current msg comes from
   * \return bytes of data
   *   > 0 : bytes we sent
   *   - 1 : error
   */
  virtual int64_t Recv(Message* msg, int* send_id) = 0;

  /*!
   * \brief Recv data from a specified Sender
   * \param msg pointer of data message
   * \param send_id sender's ID
   * \return bytes of data
   *   > 0 : bytes we sent
   *   - 1 : error
   */
  virtual int64_t RecvFrom(Message* msg, int send_id) = 0;

  /*!
   * \brief Finalize Receiver
   */
  virtual void Finalize() = 0;

  /*!
   * \brief Communicator type: 'socket', 'mpi', etc
   */
  virtual std::string Type() const = 0;

 protected:
  /*!
   * \brief Size of message queue
   */
  int64_t queue_size_;
};

}  // namespace network
}  // namespace dgl

#endif  // DGL_GRAPH_NETWORK_COMMUNICATOR_H_
