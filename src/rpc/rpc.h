/*!
 *  Copyright (c) 2020 by Contributors
 * \file rpc/rpc.h
 * \brief Common headers for remote process call (RPC).
 */
#ifndef DGL_RPC_RPC_H_
#define DGL_RPC_RPC_H_

#include <dgl/runtime/object.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/zerocopy_serializer.h>
#include <dmlc/thread_local.h>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include "./network/communicator.h"
#include "./network/socket_communicator.h"
#include "./network/msg_queue.h"
#include "./network/common.h"
#include "./server_state.h"

namespace dgl {
namespace rpc {

// Communicator handler type
typedef void* CommunicatorHandle;

/*! \brief Context information for RPC communication */
struct RPCContext {
  /*!
   * \brief Rank of this process.
   *
   * If the process is a client, this is equal to client ID. Otherwise, the process
   * is a server and this is equal to server ID.
   */
  int32_t rank = -1;

  /*!
   * \brief Cuurent machine ID
   */
  int32_t machine_id = -1;

  /*!
   * \brief Total number of machines.
   */
  int32_t num_machines = 0;

  /*!
   * \brief Message sequence number.
   */
  int64_t msg_seq = 0;

  /*!
   * \brief Total number of server.
   */
  int32_t num_servers = 0;

  /*!
   * \brief Total number of client.
   */
  int32_t num_clients = 0;

  /*!
   * \brief Current barrier count
   */
  int32_t barrier_count = 0;

  /*!
   * \brief Total number of server per machine.
   */
  int32_t num_servers_per_machine = 0;

  /*!
   * \brief Sender communicator.
   */
  std::shared_ptr<network::Sender> sender;

  /*!
   * \brief Receiver communicator.
   */
  std::shared_ptr<network::Receiver> receiver;

  /*!
   * \brief Server state data.
   *
   * If the process is a server, this stores necessary
   * server-side data. Otherwise, the process is a client and it stores a cache
   * of the server co-located with the client (if available). When the client
   * invokes a RPC to the co-located server, it can thus perform computation
   * locally without an actual remote call.
   */
  std::shared_ptr<ServerState> server_state;

  /*! \brief Get the thread-local RPC context structure */
  static RPCContext *ThreadLocal() {
    return dmlc::ThreadLocalStore<RPCContext>::Get();
  }

  /*! \brief Reset the RPC context */
  static void Reset() {
    auto* t = ThreadLocal();
    t->rank = -1;
    t->machine_id = -1;
    t->num_machines = 0;
    t->num_clients = 0;
    t->barrier_count = 0;
    t->num_servers_per_machine = 0;
    t->sender = std::shared_ptr<network::Sender>();
    t->receiver = std::shared_ptr<network::Receiver>();
  }
};

/*! \brief RPC message data structure
 *
 * This structure is exposed to Python and can be used as argument or return value
 * in C API.
 */
struct RPCMessage : public runtime::Object {
  /*! \brief Service ID */
  int32_t service_id;

  /*! \brief Sequence number of this message. */
  int64_t msg_seq;

  /*! \brief Client ID. */
  int32_t client_id;

  /*! \brief Server ID. */
  int32_t server_id;

  /*! \brief Payload buffer carried by this request.*/
  std::string data;

  /*! \brief Extra payloads in the form of tensors.*/
  std::vector<runtime::NDArray> tensors;

  bool Load(dmlc::Stream* stream) {
    stream->Read(&service_id);
    stream->Read(&msg_seq);
    stream->Read(&client_id);
    stream->Read(&server_id);
    stream->Read(&data);
    stream->Read(&tensors);
    return true;
  }

  void Save(dmlc::Stream* stream) const {
    stream->Write(service_id);
    stream->Write(msg_seq);
    stream->Write(client_id);
    stream->Write(server_id);
    stream->Write(data);
    stream->Write(tensors);
  }

  static constexpr const char* _type_key = "rpc.RPCMessage";
  DGL_DECLARE_OBJECT_TYPE_INFO(RPCMessage, runtime::Object);
};

DGL_DEFINE_OBJECT_REF(RPCMessageRef, RPCMessage);

/*! \brief RPC status flag */
enum RPCStatus {
  kRPCSuccess = 0,
  kRPCTimeOut,
};

/*!
 * \brief Send out one RPC message.
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
 * \param msg RPC message to send
 * \return status flag
 */
RPCStatus SendRPCMessage(const RPCMessage& msg);

/*!
 * \brief Receive one RPC message.
 *
 * The operation is blocking -- it returns when it receives any message
 *
 * \param msg The received message
 * \param timeout The timeout value in milliseconds. If zero, wait indefinitely.
 * \return status flag
 */
RPCStatus RecvRPCMessage(RPCMessage* msg, int32_t timeout = 0);

}  // namespace rpc
}  // namespace dgl

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, dgl::rpc::RPCMessage, true);
}  // namespace dmlc

#endif  // DGL_RPC_RPC_H_
