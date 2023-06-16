/**
 *  Copyright (c) 2020 by Contributors
 * @file rpc/rpc.h
 * @brief Common headers for remote process call (RPC).
 */
#ifndef DGL_RPC_RPC_H_
#define DGL_RPC_RPC_H_

#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/object.h>
#include <dgl/zerocopy_serializer.h>
#include <dmlc/thread_local.h>

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "./network/common.h"
#include "./rpc_msg.h"
#include "./server_state.h"
#include "network/socket_communicator.h"

namespace dgl {
namespace rpc {

struct RPCContext;

// Communicator handler type
typedef void* CommunicatorHandle;

/** @brief Context information for RPC communication */
struct RPCContext {
  /**
   * @brief Rank of this process.
   *
   * If the process is a client, this is equal to client ID. Otherwise, the
   * process is a server and this is equal to server ID.
   */
  int32_t rank = -1;

  /**
   * @brief Cuurent machine ID
   */
  int32_t machine_id = -1;

  /**
   * @brief Total number of machines.
   */
  int32_t num_machines = 0;

  /**
   * @brief Message sequence number.
   */
  std::atomic<int64_t> msg_seq{0};

  /**
   * @brief Total number of server.
   */
  int32_t num_servers = 0;

  /**
   * @brief Total number of client.
   */
  int32_t num_clients = 0;

  /**
   * @brief Current barrier count
   */
  std::unordered_map<int32_t, int32_t> barrier_count;

  /**
   * @brief Total number of server per machine.
   */
  int32_t num_servers_per_machine = 0;

  /**
   * @brief Sender communicator.
   */
  std::shared_ptr<network::SocketSender> sender;

  /**
   * @brief Receiver communicator.
   */
  std::shared_ptr<network::SocketReceiver> receiver;

  /**
   * @brief Server state data.
   *
   * If the process is a server, this stores necessary
   * server-side data. Otherwise, the process is a client and it stores a cache
   * of the server co-located with the client (if available). When the client
   * invokes a RPC to the co-located server, it can thus perform computation
   * locally without an actual remote call.
   */
  std::shared_ptr<ServerState> server_state;

  /**
   * @brief Cuurent group ID
   */
  int32_t group_id = -1;
  int32_t curr_client_id = -1;
  std::unordered_map<int32_t, std::unordered_map<int32_t, int32_t>> clients_;

  /** @brief Get the RPC context singleton */
  static RPCContext* getInstance() {
    static RPCContext ctx;
    return &ctx;
  }

  /** @brief Reset the RPC context */
  static void Reset() {
    auto* t = getInstance();
    t->rank = -1;
    t->machine_id = -1;
    t->num_machines = 0;
    t->msg_seq = 0;
    t->num_servers = 0;
    t->num_clients = 0;
    t->barrier_count.clear();
    t->num_servers_per_machine = 0;
    t->sender.reset();
    t->receiver.reset();
    t->server_state.reset();
    t->group_id = -1;
    t->curr_client_id = -1;
    t->clients_.clear();
  }

  int32_t RegisterClient(int32_t client_id, int32_t group_id) {
    auto&& m = clients_[group_id];
    if (m.find(client_id) != m.end()) {
      return -1;
    }
    m[client_id] = ++curr_client_id;
    return curr_client_id;
  }

  int32_t GetClient(int32_t client_id, int32_t group_id) const {
    if (clients_.find(group_id) == clients_.end()) {
      return -1;
    }
    const auto& m = clients_.at(group_id);
    if (m.find(client_id) == m.end()) {
      return -1;
    }
    return m.at(client_id);
  }
};

/**
 * @brief Send out one RPC message.
 *
 * The operation is non-blocking -- it does not guarantee the payloads have
 * reached the target or even have left the sender process. However,
 * all the payloads (i.e., data and arrays) can be safely freed after this
 * function returns.
 *
 * The data buffer in the requst will be copied to internal buffer for actual
 * transmission, while no memory copy for tensor payloads (a.k.a. zero-copy).
 * The underlying sending threads will hold references to the tensors until
 * the contents have been transmitted.
 *
 * @param msg RPC message to send
 * @return status flag
 */
RPCStatus SendRPCMessage(const RPCMessage& msg);

/**
 * @brief Receive one RPC message.
 *
 * The operation is blocking -- it returns when it receives any message
 *
 * @param msg The received message
 * @param timeout The timeout value in milliseconds. If zero, wait indefinitely.
 * @return status flag
 */
RPCStatus RecvRPCMessage(RPCMessage* msg, int32_t timeout = 0);

}  // namespace rpc
}  // namespace dgl

#endif  // DGL_RPC_RPC_H_
