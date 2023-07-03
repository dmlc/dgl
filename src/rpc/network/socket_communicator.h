/**
 *  Copyright (c) 2019 by Contributors
 * @file communicator.h
 * @brief SocketCommunicator for DGL distributed training.
 */
#ifndef DGL_RPC_NETWORK_SOCKET_COMMUNICATOR_H_
#define DGL_RPC_NETWORK_SOCKET_COMMUNICATOR_H_

#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../../runtime/semaphore_wrapper.h"
#include "../rpc_msg.h"
#include "common.h"
#include "communicator.h"
#include "msg_queue.h"
#include "tcp_socket.h"

namespace dgl {
namespace network {

static constexpr int kTimeOut =
    10 * 60;  // 10 minutes (in seconds) for socket timeout
static constexpr int kMaxConnection = 1024;  // maximal connection: 1024

/**
 * @breif Networking address
 */
struct IPAddr {
  std::string ip;
  int port;
};

/**
 * @brief SocketSender for DGL distributed training.
 *
 * SocketSender is the communicator implemented by tcp socket.
 */
class SocketSender : public Sender {
 public:
  /**
   * @brief Sender constructor
   * @param queue_size size of message queue
   * @param max_thread_count size of thread pool. 0 for no limit
   */
  SocketSender(int64_t queue_size, int max_thread_count)
      : Sender(queue_size, max_thread_count) {}

  /**
   * @brief Connect to a receiver.
   *
   * When there are multiple receivers to be connected, application will call
   * `ConnectReceiver` for each and then call `ConnectReceiverFinalize` to make
   * sure that either all the connections are successfully established or some
   * of them fail.
   *
   * @param addr Networking address, e.g., 'tcp://127.0.0.1:50091'
   * @param recv_id receiver's ID
   * @return True for success and False for fail
   *
   * The function is *not* thread-safe; only one thread can invoke this API.
   */
  bool ConnectReceiver(const std::string& addr, int recv_id);

  /**
   * @brief Finalize the action to connect to receivers. Make sure that either
   *        all connections are successfully established or connection fails.
   * @return True for success and False for fail
   *
   * The function is *not* thread-safe; only one thread can invoke this API.
   */
  bool ConnectReceiverFinalize(const int max_try_times);

  /**
   * @brief Send RPCMessage to specified Receiver.
   * @param msg data message
   * @param recv_id receiver's ID
   */
  void Send(const rpc::RPCMessage& msg, int recv_id);

  /**
   * @brief Finalize TPSender
   */
  void Finalize();

  /**
   * @brief Send data to specified Receiver. Actually pushing message to message
   * queue.
   * @param msg data message.
   * @param recv_id receiver's ID.
   * @return Status code.
   *
   * (1) The send is non-blocking. There is no guarantee that the message has
   * been physically sent out when the function returns. (2) The communicator
   * will assume the responsibility of the given message. (3) The API is
   * multi-thread safe. (4) Messages sent to the same receiver are guaranteed to
   * be received in the same order. There is no guarantee for messages sent to
   * different receivers.
   */
  STATUS Send(Message msg, int recv_id) override;

 private:
  /**
   * @brief socket for each connection of receiver
   */
  std::vector<
      std::unordered_map<int /* receiver ID */, std::shared_ptr<TCPSocket>>>
      sockets_;

  /**
   * @brief receivers' address
   */
  std::unordered_map<int /* receiver ID */, IPAddr> receiver_addrs_;

  /**
   * @brief message queue for each thread
   */
  std::vector<std::shared_ptr<MessageQueue>> msg_queue_;

  /**
   * @brief Independent thread
   */
  std::vector<std::shared_ptr<std::thread>> threads_;

  /**
   * @brief Send-loop for each thread
   * @param sockets TCPSockets for current thread
   * @param queue message_queue for current thread
   *
   * Note that, the SendLoop will finish its loop-job and exit thread
   * when the main thread invokes Signal() API on the message queue.
   */
  static void SendLoop(
      std::unordered_map<
          int /* Receiver (virtual) ID */, std::shared_ptr<TCPSocket>>
          sockets,
      std::shared_ptr<MessageQueue> queue);
};

/**
 * @brief SocketReceiver for DGL distributed training.
 *
 * SocketReceiver is the communicator implemented by tcp socket.
 */
class SocketReceiver : public Receiver {
 public:
  /**
   * @brief Receiver constructor
   * @param queue_size size of message queue.
   * @param max_thread_count size of thread pool. 0 for no limit
   */
  SocketReceiver(int64_t queue_size, int max_thread_count)
      : Receiver(queue_size, max_thread_count) {}

  /**
   * @brief Wait for all the Senders to connect
   * @param addr Networking address, e.g., 'tcp://127.0.0.1:50051', 'mpi://0'
   * @param num_sender total number of Senders
   * @return True for success and False for fail
   *
   * Wait() is not thread-safe and only one thread can invoke this API.
   */
  bool Wait(const std::string& addr, int num_sender);

  /**
   * @brief Recv RPCMessage from Sender. Actually removing data from queue.
   * @param msg pointer of RPCmessage
   * @param timeout The timeout value in milliseconds. If zero, wait
   * indefinitely.
   * @return RPCStatus: kRPCSuccess or kRPCTimeOut.
   */
  rpc::RPCStatus Recv(rpc::RPCMessage* msg, int timeout);

  /**
   * @brief Recv data from Sender. Actually removing data from msg_queue.
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
  STATUS Recv(Message* msg, int* send_id, int timeout = 0) override;

  /**
   * @brief Recv data from a specified Sender. Actually removing data from
   * msg_queue.
   * @param msg pointer of data message.
   * @param send_id sender's ID
   * @param timeout The timeout value in milliseconds. If zero, wait
   * indefinitely.
   * @return Status code
   *
   * (1) The RecvFrom() API is thread-safe.
   * (2) Memory allocated by communicator but will not own it after the function
   * returns.
   */
  STATUS RecvFrom(Message* msg, int send_id, int timeout = 0) override;

  /**
   * @brief Finalize SocketReceiver
   *
   * Finalize() is not thread-safe and only one thread can invoke this API.
   */
  void Finalize();

 private:
  struct RecvContext {
    int64_t data_size = -1;
    int64_t received_bytes = 0;
    char* buffer = nullptr;
  };
  /**
   * @brief number of sender
   */
  int num_sender_;

  /**
   * @brief server socket for listening connections
   */
  TCPSocket* server_socket_;

  /**
   * @brief socket for each client connections
   */
  std::vector<std::unordered_map<
      int /* Sender (virutal) ID */, std::shared_ptr<TCPSocket>>>
      sockets_;

  /**
   * @brief Message queue for each socket connection
   */
  std::unordered_map<
      int /* Sender (virtual) ID */, std::shared_ptr<MessageQueue>>
      msg_queue_;
  std::unordered_map<int, std::shared_ptr<MessageQueue>>::iterator mq_iter_;

  /**
   * @brief Independent thead
   */
  std::vector<std::shared_ptr<std::thread>> threads_;

  /**
   * @brief queue_sem_ semphore to indicate number of messages in multiple
   * message queues to prevent busy wait of Recv
   */
  runtime::Semaphore queue_sem_;

  /**
   * @brief Recv-loop for each thread
   * @param sockets client sockets of current thread
   * @param queue message queues of current thread
   *
   * Note that, the RecvLoop will finish its loop-job and exit thread
   * when the main thread invokes Signal() API on the message queue.
   */
  static void RecvLoop(
      std::unordered_map<
          int /* Sender (virtual) ID */, std::shared_ptr<TCPSocket>>
          sockets,
      std::unordered_map<
          int /* Sender (virtual) ID */, std::shared_ptr<MessageQueue>>
          queues,
      runtime::Semaphore* queue_sem);
};

}  // namespace network
}  // namespace dgl

#endif  // DGL_RPC_NETWORK_SOCKET_COMMUNICATOR_H_
