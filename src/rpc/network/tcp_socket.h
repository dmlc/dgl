/**
 *  Copyright (c) 2019 by Contributors
 * @file tcp_socket.h
 * @brief TCP socket for DGL distributed training.
 */
#ifndef DGL_RPC_NETWORK_TCP_SOCKET_H_
#define DGL_RPC_NETWORK_TCP_SOCKET_H_

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib, "Ws2_32.lib")
#else  // !_WIN32
#include <sys/socket.h>
#endif  // _WIN32
#include <string>

namespace dgl {
namespace network {

/**
 * @brief TCPSocket is a simple wrapper around a socket.
 * It supports only TCP connections.
 */
class TCPSocket {
 public:
  /**
   * @brief TCPSocket constructor
   */
  TCPSocket();

  /**
   * @brief TCPSocket deconstructor
   */
  ~TCPSocket();

  /**
   * @brief Connect to a given server address
   * @param ip ip address
   * @param port end port
   * @return true for success and false for failure
   */
  bool Connect(const char* ip, int port);

  /**
   * @brief Bind on the given IP and PORT
   * @param ip ip address
   * @param port end port
   * @return true for success and false for failure
   */
  bool Bind(const char* ip, int port);

  /**
   * @brief listen for remote connection
   * @param max_connection maximal connection
   * @return true for success and false for failure
   */
  bool Listen(int max_connection);

  /**
   * @brief wait doe a new connection
   * @param socket new SOCKET will be stored to socket
   * @param ip_client new IP will be stored to ip_client
   * @param port_client new PORT will be stored to port_client
   * @return true for success and false for failure
   */
  bool Accept(TCPSocket* socket, std::string* ip_client, int* port_client);

  /**
   * @brief SetNonBlocking() is needed refering to this example of epoll:
   * http://www.kernel.org/doc/man-pages/online/pages/man4/epoll.4.html
   * @param flag true for nonblocking, false for blocking
   * @return true for success and false for failure
   */
  bool SetNonBlocking(bool flag);

  /**
   * @brief Set timeout for socket
   * @param timeout seconds timeout
   */
  void SetTimeout(int timeout);

  /**
   * @brief Shut down one or both halves of the connection.
   * @param ways ways for shutdown
   * If ways is SHUT_RD, further receives are disallowed.
   * If ways is SHUT_WR, further sends are disallowed.
   * If ways is SHUT_RDWR, further sends and receives are disallowed.
   * @return true for success and false for failure
   */
  bool ShutDown(int ways);

  /**
   * @brief close socket.
   */
  void Close();

  /**
   * @brief Send data.
   * @param data data for sending
   * @param len_data length of data
   * @return return number of bytes sent if OK, -1 on error
   */
  int64_t Send(const char* data, int64_t len_data);

  /**
   * @brief Receive data.
   * @param buffer buffer for receving
   * @param size_buffer size of buffer
   * @return return number of bytes received if OK, -1 on error
   */
  int64_t Receive(char* buffer, int64_t size_buffer);

  /**
   * @brief Get socket's file descriptor
   * @return socket's file descriptor
   */
  int Socket() const;

 private:
  /**
   * @brief socket's file descriptor
   */
  int socket_;
};

}  // namespace network
}  // namespace dgl

#endif  // DGL_RPC_NETWORK_TCP_SOCKET_H_
