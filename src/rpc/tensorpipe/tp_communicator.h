/*!
 *  Copyright (c) 2019 by Contributors
 * \file tp_communicator.h
 * \brief Tensorpipe Communicator for DGL distributed training.
 */
#ifndef DGL_RPC_TENSORPIPE_TP_COMMUNICATOR_H_
#define DGL_RPC_TENSORPIPE_TP_COMMUNICATOR_H_

#include <dmlc/logging.h>
#include <tensorpipe/tensorpipe.h>

#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../net_type.h"
#include "./queue.h"

namespace dgl {
namespace rpc {

typedef Queue<RPCMessage> RPCMessageQueue;

/*!
 * \brief TPSender for DGL distributed training.
 *
 * TPSender is the communicator implemented by tcp socket.
 */
class TPSender : public RPCSender {
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
   * \brief Sender destructor
   */
  ~TPSender() { Finalize(); }

  /*!
   * \brief Connect to a receiver.
   *
   * When there are multiple receivers to be connected, application will call
   * `ConnectReceiver` for each and then call `ConnectReceiverFinalize` to make
   * sure that either all the connections are successfully established or some
   * of them fail.
   *
   * \param addr Networking address, e.g., 'tcp://127.0.0.1:50091'
   * \param recv_id receiver's ID
   * \return True for success and False for fail
   *
   * The function is *not* thread-safe; only one thread can invoke this API.
   */
  bool ConnectReceiver(const std::string& addr, int recv_id) override;

  /*!
   * \brief Send RPCMessage to specified Receiver.
   * \param msg data message
   * \param recv_id receiver's ID
   */
  void Send(const RPCMessage& msg, int recv_id) override;

  /*!
   * \brief Finalize TPSender
   */
  void Finalize() override;

  /*!
   * \brief Communicator type: 'tp'
   */
  const std::string& NetType() const override {
    static const std::string net_type = "tensorpipe";
    return net_type;
  }

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
class TPReceiver : public RPCReceiver {
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
   * \brief Receiver destructor
   */
  ~TPReceiver() { Finalize(); }

  /*!
   * \brief Wait for all the Senders to connect
   * \param addr Networking address, e.g., 'tcp://127.0.0.1:50051'
   * \param num_sender total number of Senders
   * \param blocking whether to wait blockingly
   * \return True for success and False for fail
   *
   * Wait() is not thread-safe and only one thread can invoke this API.
   */
  bool Wait(
      const std::string& addr, int num_sender, bool blocking = true) override;

  /*!
   * \brief Recv RPCMessage from Sender. Actually removing data from queue.
   * \param msg pointer of RPCmessage
   * \param timeout The timeout value in milliseconds. If zero, wait
   * indefinitely.
   * \return RPCStatus: kRPCSuccess or kRPCTimeOut.
   */
  RPCStatus Recv(RPCMessage* msg, int timeout) override;

  /*!
   * \brief Finalize SocketReceiver
   *
   * Finalize() is not thread-safe and only one thread can invoke this API.
   */
  void Finalize() override;

  /*!
   * \brief Communicator type: 'tp' (tensorpipe)
   */
  const std::string& NetType() const override {
    static const std::string net_type = "tensorpipe";
    return net_type;
  }

  /*!
   * \brief Issue a receive request on pipe, and push the result into queue
   */
  static void ReceiveFromPipe(
      std::shared_ptr<tensorpipe::Pipe> pipe,
      std::shared_ptr<RPCMessageQueue> queue);

 private:
  /*!
   * \brief Callback for new connection is accepted.
   */
  void OnAccepted(const tensorpipe::Error&, std::shared_ptr<tensorpipe::Pipe>);

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
  std::unordered_map<
      int /* Sender (virutal) ID */, std::shared_ptr<tensorpipe::Pipe>>
      pipes_;

  /*!
   * \brief RPCMessage queue
   */
  std::shared_ptr<RPCMessageQueue> queue_;

  /*!
   * \brief number of accepted connections
   */
  std::atomic<int32_t> num_connected_{0};

  /*!
   * \brief listner
   */
  std::shared_ptr<tensorpipe::Listener> listener_{nullptr};
};

}  // namespace rpc
}  // namespace dgl

#endif  // DGL_RPC_TENSORPIPE_TP_COMMUNICATOR_H_
