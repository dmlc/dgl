/*!
 *  Copyright (c) 2019 by Contributors
 * \file tp_communicator.h
 * \brief Tensorpipe Communicator for DGL distributed training.
 */
#ifndef DGL_RPC_TENSORPIPE_TP_COMMUNICATOR_H_
#define DGL_RPC_TENSORPIPE_TP_COMMUNICATOR_H_

#include <dmlc/logging.h>
#include <tensorpipe/tensorpipe.h>

#include <deque>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "./queue.h"

// #include "communicator.h"
// #include "msg_queue.h"
// #include "tcp_socket.h"
// #include "common.h"

namespace dgl {
namespace rpc {

class RPCMessage;

typedef Queue<RPCMessage> RPCMessageQueue;

/*!
 * \brief TPSender for DGL distributed training.
 *
 * TPSender is the communicator implemented by tcp socket.
 */
class TPSender {
 public:
  /*!
   * \brief Sender constructor
   * \param queue_size size of message queue
   */
  explicit TPSender(std::shared_ptr<tensorpipe::Context> ctx) {
    CHECK(ctx) << "Context is not initialized";
    this->context = ctx;
  }

  /*!
   * \brief Add receiver's address and ID to the sender's namebook
   * \param addr Networking address, e.g., 'tcp://127.0.0.1:50091'
   * \param id receiver's ID
   *
   * AddReceiver() is not thread-safe and only one thread can invoke this API.
   */
  void AddReceiver(const std::string& addr, int recv_id);

  /*!
   * \brief Connect with all the Receivers
   * \return True for success and False for fail
   *
   * Connect() is not thread-safe and only one thread can invoke this API.
   */
  bool Connect();

  /*!
   * \brief Send data to specified Receiver. Actually pushing message to message
   * queue. \param msg data message \param recv_id receiver's ID \return Status
   * code
   *
   * (1) The send is non-blocking. There is no guarantee that the message has
   * been physically sent out when the function returns. (2) The communicator
   * will assume the responsibility of the given message. (3) The API is
   * multi-thread safe. (4) Messages sent to the same receiver are guaranteed to
   * be received in the same order. There is no guarantee for messages sent to
   * different receivers.
   */
  void Send(RPCMessage msg, int recv_id);

  /*!
   * \brief Finalize SocketSender
   *
   * Finalize() is not thread-safe and only one thread can invoke this API.
   */
  void Finalize();

  /*!
   * \brief Communicator type: 'tp'
   */
  inline std::string Type() const { return std::string("tp"); }

 private:
  std::shared_ptr<tensorpipe::Context> context;

  /*!
   * \brief socket for each connection of receiver
   */
  std::unordered_map<int /* receiver ID */, std::shared_ptr<tensorpipe::Pipe>>
    pipes_;

  /*!
   * \brief receivers' address
   */
  std::unordered_map<int /* receiver ID */, std::string> receiver_addrs_;

  /*!
   * \brief message queue for each socket connection
   */
  //   std::unordered_map<int /* receiver ID */, std::shared_ptr<MessageQueue>>
  //   msg_queue_;

  /*!
   * \brief Independent thread for each socket connection
   */
  //   std::unordered_map<int /* receiver ID */, std::shared_ptr<std::thread>>
  //   threads_;

  /*!
   * \brief Send-loop for each socket in per-thread
   * \param socket TCPSocket for current connection
   * \param queue message_queue for current connection
   *
   * Note that, the SendLoop will finish its loop-job and exit thread
   * when the main thread invokes Signal() API on the message queue.
   */
  //   static void SendLoop(TCPSocket* socket, MessageQueue* queue);
}

/*!
 * \brief SocketReceiver for DGL distributed training.
 *
 * SocketReceiver is the communicator implemented by tcp socket.
 */
class TPReceiver {
 public:
  /*!
   * \brief Receiver constructor
   * \param queue_size size of message queue.
   */
  explicit TPReceiver(std::shared_ptr<tensorpipe::Context> ctx) {
    CHECK(ctx) << "Context is not initialized";
    this->context = ctx;
    queue_ = std::make_shared<RPCMessageQueue>();
  }

  /*!
   * \brief Wait for all the Senders to connect
   * \param addr Networking address, e.g., 'tcp://127.0.0.1:50051'
   * \param num_sender total number of Senders
   * \return True for success and False for fail
   *
   * Wait() is not thread-safe and only one thread can invoke this API.
   */
  bool Wait(const std::string& addr, int num_sender);

  /*!
   * \brief Recv RPCMessage from Sender. Actually removing data from queue.
   * \param msg pointer of RPCmessage
   * \param send_id which sender current msg comes from
   * \return Status code
   *
   * (1) The Recv() API is blocking, which will not
   *     return until getting data from message queue.
   * (2) The Recv() API is thread-safe.
   * (3) Memory allocated by communicator but will not own it after the function
   * returns.
   */
  void Recv(RPCMessage* msg);

  /*!
   * \brief Finalize SocketReceiver
   *
   * Finalize() is not thread-safe and only one thread can invoke this API.
   */
  void Finalize();

  /*!
   * \brief Communicator type: 'tp' (tensorpipe)
   */
  inline std::string Type() const { return std::string("tp"); }

  /*!
   * \brief Issue a receive request on pipe, and push the result into queue
   */
  static void ReceiveFromPipe(std::shared_ptr<tensorpipe::Pipe> pipe,
                              std::shared_ptr<RPCMessageQueue> queue);

 private:
  /*!
   * \brief number of sender
   */
  int num_sender_;

  std::shared_ptr<tensorpipe::Listener> listener;

  /*!
   * \brief server socket for listening connections
   */
  // TCPSocket* server_socket_;

  std::shared_ptr<tensorpipe::Context> context;
  /*!
   * \brief socket for each client connections
   */
  std::unordered_map<int /* Sender (virutal) ID */,
                     std::shared_ptr<tensorpipe::Pipe>>
    pipes_;

  std::shared_ptr<RPCMessageQueue> queue_;
};

}  // namespace rpc
}  // namespace dgl

#endif  // DGL_RPC_TENSORPIPE_TP_COMMUNICATOR_H_
