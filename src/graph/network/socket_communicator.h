/*!
 *  Copyright (c) 2019 by Contributors
 * \file communicator.h
 * \brief SocketCommunicator for DGL distributed training.
 */
#ifndef DGL_GRAPH_NETWORK_SOCKET_COMMUNICATOR_H_
#define DGL_GRAPH_NETWORK_SOCKET_COMMUNICATOR_H_

#include <thread>
#include <vector>
#include <string>
#include <unordered_map>

#include "communicator.h"
#include "msg_queue.h"
#include "tcp_socket.h"

namespace dgl {
namespace network {

using dgl::network::MessageQueue;
using dgl::network::TCPSocket;
using dgl::network::Sender;
using dgl::network::Receiver;

/*!
 * \breif Networking address
 */
struct Addr {
  std::string ip_;
  int port_;
};

/*!
 * \brief Network Sender for DGL distributed training.
 *
 * Sender is an abstract class that defines a set of APIs for sending 
 * binary data over network. It can be implemented by different underlying 
 * networking libraries such TCP socket and ZMQ. One Sender can connect to 
 * multiple receivers, and it can send data to specified receiver via receiver's ID.
 */
class SocketSender : public Sender {
 public:
  /*!
   * \brief Add receiver address and it's ID to the namebook
   * \param ip receviver's IP address
   * \param port receiver's port
   * \param id receiver's ID
   */
  void AddReceiver(const char* ip, int port, int recv_id);

  /*!
   * \brief Connect with all the Receivers
   * \return True for sucess and False for fail
   */
  bool Connect();

  /*!
   * \brief Send data to specified Receiver
   * \param data data buffer for sending
   * \param size data size for sending
   * \param recv_id receiver's ID
   * \return bytes we sent
   *   > 0 : bytes we sent
   *   - 1 : error
   */
  int64_t Send(const char* data, int64_t size, int recv_id);

  /*!
   * \brief Finalize Sender
   */
  void Finalize();

  /*!
   * \brief Get data buffer
   * \return buffer pointer
   */
  char* GetBuffer();

  /*!
   * \brief Set data buffer
   */
  void SetBuffer(char* buffer);

 private:
  /*!
   * \brief socket map
   */ 
  std::unordered_map<int, TCPSocket*> socket_map_;

  /*!
   * \brief receiver address map
   */ 
  std::unordered_map<int, Addr> receiver_addr_map_;

  /*!
   * \brief data buffer
   */ 
  char* buffer_;
};

/*!
 * \brief Network Receiver for DGL distributed training.
 *
 * Receiver is an abstract class that defines a set of APIs for receiving binary 
 * data over network. It can be implemented by different underlying networking libraries 
 * such TCP socket and ZMQ. One Receiver can connect with multiple Senders, and it can receive 
 * data from these Senders concurrently via multi-threading and message queue.
 */
class SocketReceiver : public Receiver {
 public:
  /*!
   * \brief Wait all of the Senders to connect
   * \param ip Receiver's IP address
   * \param port Receiver's port
   * \param num_sender total number of Senders
   * \param queue_size size of message queue
   * \return True for sucess and False for fail
   */
  bool Wait(const char* ip, int port, int num_sender, int queue_size);

  /*!
   * \brief Recv data from Sender (copy data from message queue)
   * \param dest data buffer of destination
   * \param max_size maximul size of data buffer
   * \return bytes we received
   *   > 0 : bytes we received
   *   - 1 : error
   */
  int64_t Recv(char* dest, int64_t max_size);

  /*!
   * \brief Finalize Receiver
   */
  void Finalize();

  /*!
   * \brief Get data buffer
   * \return buffer pointer
   */
  char* GetBuffer();

  /*!
   * \brief Set data buffer
   */
  void SetBuffer(char* buffer);

 private:
  /*!
   * \brief number of sender
   */
  int num_sender_;

  /*!
   * \brief maximal size of message queue
   */ 
  int64_t queue_size_;

  /*!
   * \brief socket list
   */ 
  std::vector<TCPSocket*> socket_;

  /*!
   * \brief Thread pool for socket connection
   */ 
  std::vector<std::thread*> thread_;

  /*!
   * \brief Message queue for communicator
   */ 
  MessageQueue* queue_;

  /*!
   * \brief data buffer
   */ 
  char* buffer_;

  /*!
   * \brief Process received message in independent threads
   * \param socket new accpeted socket
   * \param queue message queue
   * \param id producer_id
   */ 
  static void MsgHandler(TCPSocket* socket, MessageQueue* queue, int id);
};

}  // namespace network
}  // namespace dgl

#endif  // DGL_GRAPH_NETWORK_SOCKET_COMMUNICATOR_H_
