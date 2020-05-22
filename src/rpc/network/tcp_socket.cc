/*!
 *  Copyright (c) 2019 by Contributors
 * \file tcp_socket.cc
 * \brief TCP socket for DGL distributed training.
 */
#include "tcp_socket.h"

#include <dmlc/logging.h>

#ifndef _WIN32
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif  // !_WIN32

namespace dgl {
namespace network {

typedef struct sockaddr_in SAI;
typedef struct sockaddr SA;

TCPSocket::TCPSocket() {
  // init socket
  socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (socket_ < 0) {
    LOG(FATAL) << "Can't create new socket.";
  }
}

TCPSocket::~TCPSocket() {
  Close();
}

bool TCPSocket::Connect(const char * ip, int port) {
  SAI sa_server;
  sa_server.sin_family      = AF_INET;
  sa_server.sin_port        = htons(port);

  if (0 < inet_pton(AF_INET, ip, &sa_server.sin_addr) &&
      0 <= connect(socket_, reinterpret_cast<SA*>(&sa_server),
                   sizeof(sa_server))) {
    return true;
  }

  LOG(ERROR) << "Failed connect to " << ip << ":" << port;
  return false;
}

bool TCPSocket::Bind(const char * ip, int port) {
  SAI sa_server;
  sa_server.sin_family      = AF_INET;
  sa_server.sin_port        = htons(port);

  if (0 < inet_pton(AF_INET, ip, &sa_server.sin_addr) &&
      0 <= bind(socket_, reinterpret_cast<SA*>(&sa_server),
                sizeof(sa_server))) {
    return true;
  }

  LOG(ERROR) << "Failed bind on " << ip << ":" << port;
  return false;
}

bool TCPSocket::Listen(int max_connection) {
  if (0 <= listen(socket_, max_connection)) {
    return true;
  }

  LOG(ERROR) << "Failed listen on socket fd: " << socket_;
  return false;
}

bool TCPSocket::Accept(TCPSocket * socket, std::string * ip, int * port) {
  int sock_client;
  SAI sa_client;
  socklen_t len = sizeof(sa_client);

  sock_client = accept(socket_, reinterpret_cast<SA*>(&sa_client), &len);
  if (sock_client < 0) {
    LOG(ERROR) << "Failed accept connection on " << *ip << ":" << *port;
    return false;
  }

  char tmp[INET_ADDRSTRLEN];
  const char * ip_client = inet_ntop(AF_INET,
                                     &sa_client.sin_addr,
                                     tmp,
                                     sizeof(tmp));
  CHECK(ip_client != nullptr);
  ip->assign(ip_client);
  *port = ntohs(sa_client.sin_port);
  socket->socket_ = sock_client;

  return true;
}

#ifdef _WIN32
bool TCPSocket::SetBlocking(bool flag) {
  int result;
  u_long argp = flag ? 1 : 0;

  // XXX Non-blocking Windows Sockets apparently has tons of issues:
  // http://www.sockets.com/winsock.htm#Overview_BlockingNonBlocking
  // Since SetBlocking() is not used at all, I'm leaving a default
  // implementation here.  But be warned that this is not fully tested.
  if ((result = ioctlsocket(socket_, FIONBIO, &argp)) != NO_ERROR) {
    LOG(ERROR) << "Failed to set socket status.";
    return false;
  }
  return true;
}
#else   // !_WIN32
bool TCPSocket::SetBlocking(bool flag) {
  int opts;

  if ((opts = fcntl(socket_, F_GETFL)) < 0) {
    LOG(ERROR) << "Failed to get socket status.";
    return false;
  }

  if (flag) {
    opts |= O_NONBLOCK;
  } else {
    opts &= ~O_NONBLOCK;
  }

  if (fcntl(socket_, F_SETFL, opts) < 0) {
    LOG(ERROR) << "Failed to set socket status.";
    return false;
  }

  return true;
}
#endif  // _WIN32

void TCPSocket::SetTimeout(int timeout) {
  setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO,
    reinterpret_cast<char*>(&timeout), sizeof(timeout));
}

bool TCPSocket::ShutDown(int ways) {
  return 0 == shutdown(socket_, ways);
}

void TCPSocket::Close() {
  if (socket_ >= 0) {
#ifdef _WIN32
    CHECK_EQ(0, closesocket(socket_));
#else   // !_WIN32
    CHECK_EQ(0, close(socket_));
#endif  // _WIN32
    socket_ = -1;
  }
}

int64_t TCPSocket::Send(const char * data, int64_t len_data) {
  return send(socket_, data, len_data, 0);
}

int64_t TCPSocket::Receive(char * buffer, int64_t size_buffer) {
  return recv(socket_, buffer, size_buffer, 0);
}

int TCPSocket::Socket() const {
  return socket_;
}

}  // namespace network
}  // namespace dgl
