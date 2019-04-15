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
 * Sender is an abstract class that defines a set of APIs for sending data. 
 * It can be implemented by different underlying libraries such TCP socket and ZMQ. 
 * One Sender can connect to multiple receivers, and it and send binary 
 * data to specified receiver via receiver's ID.
 */
class SocketSender : public Sender {
 public:
  /*!
   * \brief Add receiver address to the list
   * \param ip receviver IP address
   * \param port receiver port
   * \param id receiver's ID
   */
  void AddReceiver(char* ip, int port, int recv_id);

  /*!
   * \brief Connect to Receiver
   * \return true for sucess and false for fail
   */
  bool Connect();

  /*!
   * \brief Send data to receiver
   * \param data data buffer
   * \param size data size
   * \param id receiver's ID
   * \return bytes send
   *   > 0 : bytes send
   *   - 1 : error
   */
  int64_t Send(char* data, int64_t size, int recv_id);

  /*!
   * \brief Finalize Sender
   */
  void Finalize();

 private:
  /*!
   * \brief socket map
   */ 
  std::unordered_map<int, TCPSocket*> socket_map_;

  /*!
   * \brief receiver address map
   */ 
  std::unordered_map<int, Addr> receiver_addr_map_;
};

/*!
 * \brief Network Receiver for DGL distributed training.
 *
 * Receiver is an abstract class that defines a set of APIs for receiving data.
 * It can be implemented by different underlying libraries such TCP socket and ZMQ.
 * One Receiver can connect with multiple senders, and it and receive binary data 
 * from all of the senders concurrently via multi-threading and message queue.
 */
class SocketReceiver : public Receiver {
 public:
  /*!
   * \brief Wait all of the Sender connect
   * \param ip IP address of receiver
   * \param port port of receiver
   * \param num_sender total number of Sender
   * \param queue_size size of message queue
   * \return true for sucess and false for fail
   */
  bool Wait(char* ip, int port, int num_sender, int queue_size);

  /*!
   * \brief Recv data from Sender
   * \param dest data buffer
   * \param max_size maximul size of data buffer
   * \return bytes received
   *   > 0 : bytes received
   *   - 1 : error
   */
  int64_t Recv(char* dest, int64_t max_size);

  /*!
   * \brief Finalize Receiver
   */
  void Finalize();

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
