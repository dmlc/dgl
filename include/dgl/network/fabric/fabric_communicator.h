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

namespace dgl {
namespace network {

enum FabricMsgTag { kSizeMsg, kDataMsg, kAddrMsg };

struct FabricAddrInfo {
  int64_t id;
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
  explicit FabricSender(int64_t queue_size, FabricEndpoint* fep)
      : Sender(queue_size),
        socket_sender(new SocketSender(queue_size)),
        fep(fep) {}

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
  inline std::string Type() const { return std::string("socket"); }

 private:
  std::unique_ptr<SocketSender> socket_sender;

  FabricEndpoint* fep;
  /*!
   * \brief socket for each connection of receiver
   */
  std::unordered_map<int /* receiver ID */, std::shared_ptr<TCPSocket>>
    sockets_;

  /*!
   * \brief receivers' address
   */
  std::unordered_map<int /* receiver ID */, IPAddr> receiver_addrs_;

  /*!
   * \brief message queue for each socket connection
   */
  std::unordered_map<int /* receiver ID */, std::shared_ptr<MessageQueue>>
    msg_queue_;

  /*!
   * \brief Independent thread for each socket connection
   */
  std::unordered_map<int /* receiver ID */, std::shared_ptr<std::thread>>
    threads_;

  /*!
   * \brief Send-loop for each socket in per-thread
   * \param socket TCPSocket for current connection
   * \param queue message_queue for current connection
   *
   * Note that, the SendLoop will finish its loop-job and exit thread
   * when the main thread invokes Signal() API on the message queue.
   */
  void SendLoop(fi_addr_t peer_addr, MessageQueue* queue);

  std::unordered_map<int /* Sender (virutal) ID */, fi_addr_t> addr_map;
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
  explicit FabricReceiver(int64_t queue_size, FabricEndpoint* fep)
      : Receiver(queue_size),
        socket_receiver(new SocketReceiver(queue_size)),
        fep(fep) {}

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
   * (3) Memory allocated by communicator but will not own it after the function
   * returns.
   */
  STATUS Recv(Message* msg, int* send_id);

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
  STATUS RecvFrom(Message* msg, int send_id);

  /*!
   * \brief Finalize SocketReceiver
   *
   * Finalize() is not thread-safe and only one thread can invoke this API.
   */
  void Finalize();

  void PollCompletionQueue(struct fi_cq_tagged_entry* cq_entries,
                           fi_addr_t* source_addrs);

  void HandleCompletionEvent(const struct fi_cq_tagged_entry& cq_entry,
                             const fi_addr_t& source_addrs);
  /*!
   * \brief Communicator type: 'socket'
   */
  inline std::string Type() const {
    return std::string("fabric:") +
           fep->fabric_provider->info->fabric_attr->prov_name;
  }
  /*!
   * \brief Recv-loop for each socket in per-thread
   * \param socket client socket
   * \param queue message queue
   *
   * Note that, the RecvLoop will finish its loop-job and exit thread
   * when the main thread invokes Signal() API on the message queue.
   */
  void RecvLoop();

 private:
  std::atomic<bool> should_stop_polling_;

  std::unique_ptr<SocketReceiver> socket_receiver;

  FabricEndpoint* fep;

  int receiver_id_;
  /*!
   * \brief number of sender
   */
  int num_sender_;

  /*!
   * \brief socket for each client connections
   */
  std::unordered_map<int /* Sender (virutal) ID */, fi_addr_t> addr_map;

  /*!
   * \brief Message queue for each socket connection
   */
  std::unordered_map<fi_addr_t /* Sender (virtual) ID */,
                     std::shared_ptr<MessageQueue>>
    msg_queue_;

  /*!
   * \brief Independent thead for each socket connection
   */
  std::unordered_map<int /* Sender (virtual) ID */,
                     std::shared_ptr<std::thread>>
    threads_;
};

}  // namespace network
}  // namespace dgl

#endif  // DGL_RPC_NETWORK_FABRIC_COMMUNICATOR_H_
