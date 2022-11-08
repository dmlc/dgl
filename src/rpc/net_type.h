/**
 *  Copyright (c) 2022 by Contributors
 * @file net_type.h
 * @brief Base communicator for DGL distributed training.
 */
#ifndef DGL_RPC_NET_TYPE_H_
#define DGL_RPC_NET_TYPE_H_

#include <string>

#include "rpc_msg.h"

namespace dgl {
namespace rpc {

struct RPCBase {
  /**
   * @brief Finalize Receiver
   *
   * Finalize() is not thread-safe and only one thread can invoke this API.
   */
  virtual void Finalize() = 0;

  /**
   * @brief Communicator type: 'socket', 'tensorpipe', etc
   */
  virtual const std::string &NetType() const = 0;
};

struct RPCSender : RPCBase {
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
  virtual bool ConnectReceiver(const std::string &addr, int recv_id) = 0;

  /**
   * @brief Finalize the action to connect to receivers. Make sure that either
   *        all connections are successfully established or connection fails.
   * @return True for success and False for fail
   *
   * The function is *not* thread-safe; only one thread can invoke this API.
   */
  virtual bool ConnectReceiverFinalize(const int max_try_times) { return true; }

  /**
   * @brief Send RPCMessage to specified Receiver.
   * @param msg data message
   * @param recv_id receiver's ID
   */
  virtual void Send(const RPCMessage &msg, int recv_id) = 0;
};

struct RPCReceiver : RPCBase {
  /**
   * @brief Wait for all the Senders to connect
   * @param addr Networking address, e.g., 'tcp://127.0.0.1:50051', 'mpi://0'
   * @param num_sender total number of Senders
   * @param blocking whether wait blockingly
   * @return True for success and False for fail
   *
   * Wait() is not thread-safe and only one thread can invoke this API.
   */
  virtual bool Wait(
      const std::string &addr, int num_sender, bool blocking = true) = 0;

  /**
   * @brief Recv RPCMessage from Sender. Actually removing data from queue.
   * @param msg pointer of RPCmessage
   * @param timeout The timeout value in milliseconds. If zero, wait
   * indefinitely.
   * @return RPCStatus: kRPCSuccess or kRPCTimeOut.
   */
  virtual RPCStatus Recv(RPCMessage *msg, int timeout) = 0;
};

}  // namespace rpc
}  // namespace dgl

#endif  // DGL_RPC_NET_TYPE_H_
