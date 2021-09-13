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
   * \brief Send RPCMessage to specified Receiver.
   * \param msg data message \param recv_id receiver's ID
   */
  void Send(const RPCMessage& msg, int recv_id);

  /*!
   * \brief Finalize TPSender
   */
  void Finalize();

  /*!
   * \brief Communicator type: 'tp'
   */
  inline std::string Type() const { return std::string("tp"); }

 private:
  /*!
   * \brief global context of tensorpipe
   */
  std::shared_ptr<tensorpipe::Context> context;

  /*!
   * \brief pipe for each connection of receiver
   */
  std::unordered_map<int /* receiver ID */, std::shared_ptr<tensorpipe::Pipe>>
    pipes_;

  /*!
   * \brief receivers' listening address
   */
  std::unordered_map<int /* receiver ID */, std::string> receiver_addrs_;
};

/*!
 * \brief TPReceiver for DGL distributed training.
 *
 * Tensorpipe Receiver is the communicator implemented by tcp socket.
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

  /*!
   * \brief listener to build pipe
   */
  std::shared_ptr<tensorpipe::Listener> listener;

  /*!
   * \brief global context of tensorpipe
   */
  std::shared_ptr<tensorpipe::Context> context;

  /*!
   * \brief pipe for each client connections
   */
  std::unordered_map<int /* Sender (virutal) ID */,
                     std::shared_ptr<tensorpipe::Pipe>>
    pipes_;

  /*!
   * \brief RPCMessage queue
   */
  std::shared_ptr<RPCMessageQueue> queue_;
};

}  // namespace rpc
}  // namespace dgl

#endif  // DGL_RPC_TENSORPIPE_TP_COMMUNICATOR_H_
