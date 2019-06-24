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
 * \brief Socket Sender for DGL distributed training.
 *
 * SocketSender is the communicator implemented by tcp socket.
 */
class SocketSender : public Sender {
 public:
  /*!
   * \brief Add receiver address and it's ID to the namebook
   * \param addr Networking address, e.g., 'socket://127.0.0.1:50051'
   * \param id receiver's ID
   */
  void AddReceiver(const char* addr, int recv_id);

  /*!
   * \brief Connect with all the Receivers
   * \return True for success of all connections and False for fail
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
   * Note that, the Send() API is a blocking API that 
   * returns until the target receiver get its message.
   */
  int64_t Send(const char* data, int64_t size, int recv_id);

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
 * \brief Socket Receiver for DGL distributed training.
 *
 * SocketReceiver is the communicator implemented by tcp socket.
 */
class SocketReceiver : public Receiver {
 public:
  /*!
   * \brief Wait all of the Senders to connect
   * \param addr Networking address, e.g., 'socket://127.0.0.1:50051'
   * \param num_sender total number of Senders
   * \param queue_size size of message queue
   * \return True for sucess and False for fail
   */
  bool Wait(const char* ip, int num_sender, int queue_size);

  /*!
   * \brief Recv data from Sender (copy data from message queue)
   * \param buffer data buffer
   * \param buff_size size of data buffer
   * \return real bytes received
   *   > 0 : size of message
   *   - 1 : error
   * Note that, the Recv() API is blocking API that returns until getting data.
   */
  int64_t Recv(char* buffer, int64_t buff_size);

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
