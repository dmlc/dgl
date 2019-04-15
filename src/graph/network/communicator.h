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
 * Sender is an abstract class that defines a set of APIs for sending data. 
 * It can be implemented by different underlying libraries such TCP socket and ZMQ. 
 * One Sender can connect to multiple receivers, and it and send binary 
 * data to specified receiver via receiver's ID.
 */
class Sender {
 public:
  virtual ~Sender() {}

  /*!
   * \brief Add receiver address to the list
   * \param ip receviver IP address
   * \param port receiver port
   * \param recv_id receiver's ID
   */
  virtual void AddReceiver(char* ip, int port, int recv_id) = 0;

  /*!
   * \brief Connect to Receiver
   * \return true for sucess and false for fail
   */
  virtual bool Connect() = 0;

  /*!
   * \brief Send data to receiver
   * \param data data buffer
   * \param size data size
   * \param recv_id receiver's ID
   * \return bytes send
   *   > 0 : bytes send
   *   - 1 : error
   */
  virtual int64_t Send(char* data, int64_t size, int recv_id) = 0;

  /*!
   * \brief Finalize Sender
   */
  virtual void Finalize() = 0;
};

/*!
 * \brief Network Receiver for DGL distributed training.
 *
 * Receiver is an abstract class that defines a set of APIs for receiving data.
 * It can be implemented by different underlying libraries such TCP socket and ZMQ.
 * One Receiver can connect with multiple senders, and it and receive binary data 
 * from all of the senders concurrently via multi-threading and message queue.
 */
class Receiver {
 public:
  virtual ~Receiver() {}

  /*!
   * \brief Wait all of the Sender connect
   * \param ip IP address of receiver
   * \param port port of receiver
   * \param num_sender total number of Sender
   * \param queue_size size of message queue
   * \return true for sucess and false for fail
   */
  virtual bool Wait(char* ip, 
                    int port, 
                    int num_sender, 
                    int queue_size) = 0;

  /*!
   * \brief Recv data from Sender
   * \param dest data buffer
   * \param max_size maximul size of data buffer
   * \return bytes received
   *   > 0 : bytes received
   *   - 1 : error
   */
  virtual int64_t Recv(char* dest, int64_t max_size) = 0;

  /*!
   * \brief Finalize Receiver
   */
  virtual void Finalize() = 0;
};

}  // namespace network
}  // namespace dgl

#endif  // DGL_GRAPH_NETWORK_COMMUNICATOR_H_
