/*!
 *  Copyright (c) 2019 by Contributors
 * \file communicator.h
 * \brief SocketCommunicator for DGL distributed training.
 */
#ifndef DGL_RPC_NETWORK_FABRIC_COMMUNICATOR_H_
#define DGL_RPC_NETWORK_FABRIC_COMMUNICATOR_H_

#include <dgl/network/common.h>
#include <dgl/network/communicator.h>
#include <dgl/network/fabric/fabric_endpoint.h>
#include <dgl/network/msg_queue.h>
#include <dgl/network/socket_communicator.h>
#include <dgl/network/tcp_socket.h>

#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "dmlc/thread_local.h"

namespace dgl {
namespace network {

enum FabricMsgTag : uint64_t {
  kSizeMsg = 0x0000000000010000,
  kDataMsg = 0x0000000000020000,
  kCtrlAddrMsg = 0x0000000000030000,
  kFiAddrMsg = 0x0000000000040000,
  kIgnoreMsg = 0x0000000000050000,
};

// static const std::string handshake_msg = "ready";
// static char handshake_buffer[16] = "ready";

static const uint64_t MsgTagMask = 0x00000000FFFF0000;
static const uint64_t IdMask = 0x000000000000FFFF;
// static uint64_t MsgTagMask =std::numeric_limits<uint64_t>::max();

struct FabricAddrInfo {
  int sender_id;
  int receiver_id;
  struct FabricAddr addr;
};

/*!
 * \brief SocketSender for DGL distributed training.
 *
 * SocketSender is the communicator implemented by tcp socket.
 */
class FabricSender : public Sender {
 public:
  /*!
   * \brief Sender constructor
   * \param queue_size size of message queue
   */
  explicit FabricSender(int64_t queue_size, std::string net_type)
      : Sender(queue_size), fep(FabricEndpoint::GetEndpoint()) {
    fep->Init(net_type);
    ctrl_ep = std::unique_ptr<FabricEndpoint>(
      FabricEndpoint::CreateCtrlEndpoint(nullptr));
  }

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
  inline std::string Type() const {
    return std::string("fabric:") + fep->fabric_provider->prov_name;
  }
  // ~FabricSender() override { Finalize(); }

 private:
  int64_t sender_id;

  std::shared_ptr<std::thread> poll_thread;

  std::unique_ptr<FabricEndpoint> ctrl_ep;

  // fep uses global singleton from ThreadLocalStore
  std::shared_ptr<FabricEndpoint> fep;

  /*!
   * \brief receivers' address
   */
  std::unordered_map<int /* receiver ID */, IPAddr> ctrl_peer_addr;

  std::unordered_map<int /* Sender (virutal) ID */, fi_addr_t> peer_fi_addr,
    ctrl_peer_fi_addr;

  /*!
   * \brief message queue for each socket connection
   */
  std::unordered_map<int /* receiver ID */, std::shared_ptr<MessageQueue>>
    msg_queue_;
  
  struct fi_cq_tagged_entry cq_entries[kMaxConcurrentWorkRequest];

  /*!
   * \brief Send-loop for each socket in per-thread
   * \param socket TCPSocket for current connection
   * \param queue message_queue for current connection
   *
   * Note that, the SendLoop will finish its loop-job and exit thread
   * when the main thread invokes Signal() API on the message queue.
   */
  void SendLoop();

  ssize_t PollCompletionQueue(struct fi_cq_tagged_entry* cq_entries);

  void HandleCompletionEvent(const struct fi_cq_tagged_entry& cq_entry);
};

/*!
 * \brief SocketReceiver for DGL distributed training.
 *
 * SocketReceiver is the communicator implemented by tcp socket.
 */
class FabricReceiver : public Receiver {
 public:
  /*!
   * \brief Receiver constructor
   * \param queue_size size of message queue.
   */
  explicit FabricReceiver(int64_t queue_size, std::string net_type)
      : Receiver(queue_size), fep(FabricEndpoint::GetEndpoint()) {
    fep->Init(net_type);
  }

  /*!
   * \brief Wait for all the Senders to connect
   * \param addr Networking address, e.g., 'socket://127.0.0.1:50051', 'mpi://0'
   * \param num_sender total number of Senders
   * \return True for success and False for fail
   *
   * Wait() is not thread-safe and only one thread can invoke this API.
   */
  bool Wait(const char* addr, int num_sender) override;

  /*!
   * \brief Recv data from Sender. Actually removing data from msg_queue.
   * \param msg pointer of data message
   * \param send_id which sender current msg comes from
   * \return Status code
   *
   * (1) The Recv() API is blocking, which will not
   *     return until getting data from message queue.
   * (2) The Recv() API is thread-safe.
   * (3) Memory allocated by communicator but will not own it after the function
   * returns.
   */
  STATUS Recv(Message* msg, int* send_id) override;

  /*!
   * \brief Recv data from a specified Sender. Actually removing data from
   * msg_queue. \param msg pointer of data message \param send_id sender's ID
   * \return Status code
   *
   * (1) The RecvFrom() API is blocking, which will not
   *     return until getting data from message queue.
   * (2) The RecvFrom() API is thread-safe.
   * (3) Memory allocated by communicator but will not own it after the function
   * returns.
   */
  STATUS RecvFrom(Message* msg, int send_id) override;

  /*!
   * \brief Finalize SocketReceiver
   *
   * Finalize() is not thread-safe and only one thread can invoke this API.
   */
  void Finalize() override;

  /*!
   * \brief Communicator type: 'fabric:efa'
   */
  inline std::string Type() const override {
    return std::string("fabric:") + fep->fabric_provider->prov_name;
  }

 private:
  bool PollCompletionQueue(struct fi_cq_tagged_entry* cq_entries);

  bool HandleCompletionEvent(const struct fi_cq_tagged_entry& cq_entry);

  /*!
   * \brief Recv-loop for each socket in per-thread
   * \param socket client socket
   * \param queue message queue
   *
   * Note that, the RecvLoop will finish its loop-job and exit thread
   * when the main thread invokes Signal() API on the message queue.
   */
  void RecvLoop();

  std::atomic<bool> should_stop_polling_;

  std::unique_ptr<FabricEndpoint> ctrl_ep;

  std::shared_ptr<FabricEndpoint> fep;
  struct fi_cq_tagged_entry cq_entries[kMaxConcurrentWorkRequest];

  std::unordered_map<int /* Sender (virtual) ID */,
                     std::shared_ptr<std::queue<Message>>>
    msg_queue_;

  int receiver_id_;
  /*!
   * \brief number of sender
   */
  int num_sender_;

  /*!
   * \brief socket for each client connections
   */
  std::unordered_map<int /* Sender (virutal) ID */, fi_addr_t> peer_fi_addr,
    ctrl_peer_fi_addr;

  // std::unordered_map<int /* Sender (virutal) ID */, fi_addr_t> fi_to_id;

  /*!
   * \brief Message queue for each socket connection
   */
  // std::unordered_map<int /* Sender (virtual) ID */,
  //                    std::shared_ptr<MessageQueue>>
  //   msg_queue_;

  std::shared_ptr<std::thread> poll_thread;
};

}  // namespace network
}  // namespace dgl

#endif  // DGL_RPC_NETWORK_FABRIC_COMMUNICATOR_H_
