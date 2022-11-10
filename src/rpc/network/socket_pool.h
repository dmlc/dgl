/**
 *  Copyright (c) 2021 by Contributors
 * @file socket_pool.h
 * @brief Socket pool of nonblocking sockets for DGL distributed training.
 */
#ifndef DGL_RPC_NETWORK_SOCKET_POOL_H_
#define DGL_RPC_NETWORK_SOCKET_POOL_H_

#include <memory>
#include <queue>
#include <unordered_map>

namespace dgl {
namespace network {

class TCPSocket;

/**
 * @brief SocketPool maintains a group of nonblocking sockets, and can provide
 * active sockets.
 * Currently SocketPool is based on epoll, a scalable I/O event notification
 * mechanism in Linux operating system.
 */
class SocketPool {
 public:
  /**
   * @brief socket mode read/receive
   */
  static const int READ = 1;
  /**
   * @brief socket mode write/send
   */
  static const int WRITE = 2;
  /**
   * @brief SocketPool constructor
   */
  SocketPool();

  /**
   * @brief Add a socket to SocketPool
   * @param socket tcp socket to add
   * @param socket_id receiver/sender id of the socket
   * @param events READ, WRITE or READ + WRITE
   */
  void AddSocket(
      std::shared_ptr<TCPSocket> socket, int socket_id, int events = READ);

  /**
   * @brief Remove socket from SocketPool
   * @param socket tcp socket to remove
   * @return number of remaing sockets in the pool
   */
  size_t RemoveSocket(std::shared_ptr<TCPSocket> socket);

  /**
   * @brief SocketPool destructor
   */
  ~SocketPool();

  /**
   * @brief Get current active socket. This is a blocking method
   * @param socket_id output parameter of the socket_id of active socket
   * @return active TCPSocket
   */
  std::shared_ptr<TCPSocket> GetActiveSocket(int* socket_id);

 private:
  /**
   * @brief Wait for event notification
   */
  void Wait();

  /**
   * @brief map from fd to TCPSocket
   */
  std::unordered_map<int, std::shared_ptr<TCPSocket>> tcp_sockets_;

  /**
   * @brief map from fd to socket_id
   */
  std::unordered_map<int, int> socket_ids_;

  /**
   * @brief fd for epoll base
   */
  int epfd_;

  /**
   * @brief queue for current active fds
   */
  std::queue<int> pending_fds_;
};

}  // namespace network
}  // namespace dgl

#endif  // DGL_RPC_NETWORK_SOCKET_POOL_H_
