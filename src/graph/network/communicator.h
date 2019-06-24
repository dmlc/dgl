/*!
 *  Copyright (c) 2019 by Contributors
 * \file communicator.h
 * \brief Communicator for DGL distributed training.
 */
#ifndef DGL_GRAPH_NETWORK_COMMUNICATOR_H_
#define DGL_GRAPH_NETWORK_COMMUNICATOR_H_

#include <string>

namespace dgl {
namespace network {

/*!
 * \brief Network Sender for DGL distributed training.
 *
 * Sender is an abstract class that defines a set of APIs for sending 
 * binary data over network. It can be implemented by different underlying 
 * networking libraries such TCP socket and MPI. One Sender can connect to 
 * multiple receivers, and it can send data to specified receiver via receiver's ID.
 */
class Sender {
 public:
  virtual ~Sender() {}

  /*!
   * \brief Add receiver address and it's ID to the namebook
   * \param addr Networking address, e.g., 'socket://127.0.0.1:500591', 'mpi://0'
   * \param id receiver's ID
   */
  virtual void AddReceiver(const char* addr, int id) = 0;

  /*!
   * \brief Connect with all the Receivers
   * \return True for success of all connections and False for fail
   */
  virtual bool Connect() = 0;

  /*!
   * \brief Send data to specified Receiver
   * \param data data buffer for sending
   * \param size data size for sending
   * \param recv_id receiver's ID
   * \return bytes we sent
   *   > 0 : bytes we sent
   *   - 1 : error
   * Note that, the Send() API is a blocking API that 
   * returns until the target receiver get its message.
   */
  virtual int64_t Send(const char* data, int64_t size, int recv_id) = 0;

  /*!
   * \brief Finalize Sender
   */
  virtual void Finalize() = 0;
};

/*!
 * \brief Network Receiver for DGL distributed training.
 *
 * Receiver is an abstract class that defines a set of APIs for receiving binary 
 * data over network. It can be implemented by different underlying networking libraries 
 * such TCP socket and MPI. One Receiver can connect with multiple Senders, and it can receive 
 * data from multiple Senders concurrently via multi-threading and message queue.
 */
class Receiver {
 public:
  virtual ~Receiver() {}

  /*!
   * \brief Wait all of the Senders to connect
   * \param addr Networking address, e.g., 'socket://127.0.0.1:50051', 'mpi://0'
   * \param num_sender total number of Senders
   * \param queue_size size of message queue
   * \return True for sucess and False for fail
   */
  virtual bool Wait(const char* addr, int num_sender, int queue_size) = 0;

  /*!
   * \brief Recv data from Sender (copy data from message queue)
   * \param buffer data buffer
   * \param buff_size size of data buffer
   * \return real bytes received
   *   > 0 : size of message
   *   - 1 : error
   * Note that, the Recv() API is blocking API that returns until getting data.
   */
  virtual int64_t Recv(char* buffer, int64_t buff_size) = 0;

  /*!
   * \brief Finalize Receiver
   */
  virtual void Finalize() = 0;
};

}  // namespace network
}  // namespace dgl

#endif  // DGL_GRAPH_NETWORK_COMMUNICATOR_H_
