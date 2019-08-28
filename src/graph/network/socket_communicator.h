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
#include "common.h"

namespace dgl {
namespace network {

static int kMaxTryCount = 1024;    // maximal connection: 1024
static int kTimeOut = 10;          // 10 minutes for socket timeout
static int kMaxConnection = 1024;  // maximal connection: 1024

/*!
 * \breif Networking address
 */
struct IPAddr {
  std::string ip;
  int port;
};

/*!
 * \brief SocketSender for DGL distributed training.
 *
 * SocketSender is the communicator implemented by tcp socket.
 */
class SocketSender : public Sender {
 public:
  /*!
   * \brief Sender constructor
   * \param queue_size size of message queue 
   */
  explicit SocketSender(int64_t queue_size) : Sender(queue_size) {}

  /*!
   * \brief Add receiver's address and ID to the sender's namebook
   * \param addr Networking address, e.g., 'socket://127.0.0.1:50091', 'mpi://0'
   * \param id receiver's ID
   *
   * AddReceiver() is not thread-safe and only one thread can invoke this API.
   */
  void AddReceiver(const char* addr, int recv_id);

  /*!
   * \brief Connect with all the Receivers
   * \return True for success and False for fail
   *
   * Connect() is not thread-safe and only one thread can invoke this API.
   */
  bool Connect();

  /*!
   * \brief Send data to specified Receiver. Actually pushing message to message queue.
   * \param msg data message
   * \param recv_id receiver's ID
   * \return Status code
   *
   * (1) The send is non-blocking. There is no guarantee that the message has been 
   *     physically sent out when the function returns.
   * (2) The communicator will assume the responsibility of the given message.
   * (3) The API is multi-thread safe.
   * (4) Messages sent to the same receiver are guaranteed to be received in the same order. 
   *     There is no guarantee for messages sent to different receivers.
   */
  STATUS Send(Message msg, int recv_id);

  /*!
   * \brief Finalize SocketSender
   *
   * Finalize() is not thread-safe and only one thread can invoke this API.
   */
  void Finalize();

  /*!
   * \brief Communicator type: 'socket'
   */
  inline std::string Type() const { return std::string("socket"); }

 private:
  /*!
   * \brief socket for each connection of receiver
   */ 
  std::unordered_map<int /* receiver ID */, std::shared_ptr<TCPSocket>> sockets_;

  /*!
   * \brief receivers' address
   */ 
  std::unordered_map<int /* receiver ID */, IPAddr> receiver_addrs_;

  /*!
   * \brief message queue for each socket connection
   */ 
  std::unordered_map<int /* receiver ID */, std::shared_ptr<MessageQueue>> msg_queue_;

  /*!
   * \brief Independent thread for each socket connection
   */ 
  std::unordered_map<int /* receiver ID */, std::shared_ptr<std::thread>> threads_;

  /*!
   * \brief Send-loop for each socket in per-thread
   * \param socket TCPSocket for current connection
   * \param queue message_queue for current connection
   * 
   * Note that, the SendLoop will finish its loop-job and exit thread
   * when the main thread invokes Signal() API on the message queue.
   */
  static void SendLoop(TCPSocket* socket, MessageQueue* queue);
};

/*!
 * \brief SocketReceiver for DGL distributed training.
 *
 * SocketReceiver is the communicator implemented by tcp socket.
 */
class SocketReceiver : public Receiver {
 public:
  /*!
   * \brief Receiver constructor
   * \param queue_size size of message queue.
   */
  explicit SocketReceiver(int64_t queue_size) : Receiver(queue_size) {}

  /*!
   * \brief Wait for all the Senders to connect
   * \param addr Networking address, e.g., 'socket://127.0.0.1:50051', 'mpi://0'
   * \param num_sender total number of Senders
   * \return True for success and False for fail
   *
   * Wait() is not thread-safe and only one thread can invoke this API.
   */
  bool Wait(const char* addr, int num_sender);

  /*!
   * \brief Recv data from Sender. Actually removing data from msg_queue.
   * \param msg pointer of data message
   * \param send_id which sender current msg comes from
   * \return Status code
   *
   * (1) The Recv() API is blocking, which will not 
   *     return until getting data from message queue.
   * (2) The Recv() API is thread-safe.
   * (3) Memory allocated by communicator but will not own it after the function returns.
   */
  STATUS Recv(Message* msg, int* send_id);

  /*!
   * \brief Recv data from a specified Sender. Actually removing data from msg_queue.
   * \param msg pointer of data message
   * \param send_id sender's ID
   * \return Status code
   *
   * (1) The RecvFrom() API is blocking, which will not 
   *     return until getting data from message queue.
   * (2) The RecvFrom() API is thread-safe.
   * (3) Memory allocated by communicator but will not own it after the function returns.
   */
  STATUS RecvFrom(Message* msg, int send_id);

  /*!
   * \brief Finalize SocketReceiver
   *
   * Finalize() is not thread-safe and only one thread can invoke this API.
   */
  void Finalize();

  /*!
   * \brief Communicator type: 'socket'
   */
  inline std::string Type() const { return std::string("socket"); }

 private:
  /*!
   * \brief number of sender
   */
  int num_sender_;

  /*!
   * \brief server socket for listening connections
   */ 
  TCPSocket* server_socket_;

  /*!
   * \brief socket for each client connections
   */ 
  std::unordered_map<int /* Sender (virutal) ID */, std::shared_ptr<TCPSocket>> sockets_;

  /*!
   * \brief Message queue for each socket connection
   */ 
  std::unordered_map<int /* Sender (virtual) ID */, std::shared_ptr<MessageQueue>> msg_queue_;

  /*!
   * \brief Independent thead for each socket connection
   */ 
  std::unordered_map<int /* Sender (virtual) ID */, std::shared_ptr<std::thread>> threads_;

  /*!
   * \brief Recv-loop for each socket in per-thread
   * \param socket client socket
   * \param queue message queue
   *
   * Note that, the RecvLoop will finish its loop-job and exit thread
   * when the main thread invokes Signal() API on the message queue.
   */ 
  static void RecvLoop(TCPSocket* socket, MessageQueue* queue);
};

}  // namespace network
}  // namespace dgl

#endif  // DGL_GRAPH_NETWORK_SOCKET_COMMUNICATOR_H_
