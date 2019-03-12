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
 * \brief Communicator for DGL distributed training.
 *
 * Communicator is a set of interface for network communication, which
 * can be implemented by real network libraries, such as grpc, mpi, as well
 * as raw socket. There has two types of Communicator, one is Sender 
 * (is_sender = true), and another is Receiver. For Sender, it can send binary 
 * data to remote Receiver node. For Receiver, it can listen on a specified 
 * endpoint and receive the binary data sent from Sender node. Note that, a 
 * receiver node can recv messages from multiple senders concurrently.
 */
class Communicator {
 public:
  virtual ~Communicator() {}

  /*!
   * \brief Initialize Communicator
   * \param is_sender true for sender and false for receiver
   * \param ip ip address
   * \param port end port
   * (e.g. "168.123.2.43:50051"). For Receiver, this address identifies
   * the local listening endpoint (e.g. "0.0.0.0:50051").
   * \param num_sender number of senders, only used for receiver.
   * \param queue_size the size of message queue (500MB default), only for receiver.
   * \return true for success and false for error
   */
  virtual bool Initialize(bool is_sender,
                          const char* ip,
                          int port,
                          int num_sender = 0,
                          int64_t queue_size = 500*1024*1024) = 0;
  /*!
   * \brief Send message to receiver node
   * \param src data pointer
   * \param size data size
   * \return bytes send
   *   > 0 : bytes send
   *   - 1 : error
   */
  virtual int64_t Send(char* src, int64_t size) = 0;

  /*!
   * \brief Receive mesesage from sender node, we
   * actually reading data from local message queue.
   * \param dest destination data pointer
   * \param max_size maximal data size
   * \return bytes received
   *   > 0 : bytes received
   *   - 1 : error
   */
  virtual int64_t Receive(char* dest, int64_t max_size) = 0;

  /*!
   * \brief Finalize the Communicator class
   */
  virtual void Finalize() = 0;
};

}  // namespace network
}  // namespace dgl

#endif  // DGL_GRAPH_NETWORK_COMMUNICATOR_H_
