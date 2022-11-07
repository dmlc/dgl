/**
 *  Copyright (c) 2019 by Contributors
 * @file tcp_socket.cc
 * @brief TCP socket for DGL distributed training.
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
#include <errno.h>
#include <string.h>

namespace dgl {
namespace network {

typedef struct sockaddr_in SAI;
typedef struct sockaddr SA;

TCPSocket::TCPSocket() {
  // init socket
  socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (socket_ < 0) {
    LOG(FATAL) << "Can't create new socket. Error: " << strerror(errno);
  }
#ifndef _WIN32
  // This is to make sure the same port can be reused right after the socket is
  // closed.
  int enable = 1;
  if (setsockopt(socket_, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0) {
    LOG(WARNING) << "cannot make the socket reusable. Error: "
                 << strerror(errno);
  }
#endif  // _WIN32
}

TCPSocket::~TCPSocket() { Close(); }

bool TCPSocket::Connect(const char *ip, int port) {
  SAI sa_server;
  sa_server.sin_family = AF_INET;
  sa_server.sin_port = htons(port);

  int retval = 0;
  do {  // retry if EINTR failure appears
    if (0 < inet_pton(AF_INET, ip, &sa_server.sin_addr) &&
        0 <= (retval = connect(
                  socket_, reinterpret_cast<SA *>(&sa_server),
                  sizeof(sa_server)))) {
      return true;
    }
  } while (retval == -1 && errno == EINTR);

  return false;
}

bool TCPSocket::Bind(const char *ip, int port) {
  SAI sa_server;
  sa_server.sin_family = AF_INET;
  sa_server.sin_port = htons(port);
  int ret = 0;
  ret = inet_pton(AF_INET, ip, &sa_server.sin_addr);
  if (ret == 0) {
    LOG(ERROR) << "Invalid IP: " << ip;
    return false;
  } else if (ret < 0) {
    LOG(ERROR) << "Failed to convert [" << ip
               << "] to binary form, error: " << strerror(errno);
    return false;
  }
  do {  // retry if EINTR failure appears
    if (0 <=
        (ret = bind(
             socket_, reinterpret_cast<SA *>(&sa_server), sizeof(sa_server)))) {
      return true;
    }
  } while (ret == -1 && errno == EINTR);

  LOG(ERROR) << "Failed bind on " << ip << ":" << port
             << " , error: " << strerror(errno);
  return false;
}

bool TCPSocket::Listen(int max_connection) {
  int retval;
  do {  // retry if EINTR failure appears
    if (0 <= (retval = listen(socket_, max_connection))) {
      return true;
    }
  } while (retval == -1 && errno == EINTR);

  LOG(ERROR) << "Failed listen on socket fd: " << socket_
             << " , error: " << strerror(errno);
  return false;
}

bool TCPSocket::Accept(TCPSocket *socket, std::string *ip, int *port) {
  int sock_client;
  SAI sa_client;
  socklen_t len = sizeof(sa_client);

  do {  // retry if EINTR failure appears
    sock_client = accept(socket_, reinterpret_cast<SA *>(&sa_client), &len);
  } while (sock_client == -1 && errno == EINTR);

  if (sock_client < 0) {
    LOG(ERROR) << "Failed accept connection on " << *ip << ":" << *port
               << ", error: " << strerror(errno)
               << (errno == EAGAIN ? " SO_RCVTIMEO timeout reached" : "");
    return false;
  }

  char tmp[INET_ADDRSTRLEN];
  const char *ip_client =
      inet_ntop(AF_INET, &sa_client.sin_addr, tmp, sizeof(tmp));
  CHECK(ip_client != nullptr);
  ip->assign(ip_client);
  *port = ntohs(sa_client.sin_port);
  socket->socket_ = sock_client;

  return true;
}

#ifdef _WIN32
bool TCPSocket::SetNonBlocking(bool flag) {
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
bool TCPSocket::SetNonBlocking(bool flag) {
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
#ifdef _WIN32
  timeout = timeout * 1000;  // WIN API accepts millsec
  setsockopt(
      socket_, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<char *>(&timeout),
      sizeof(timeout));
#else   // !_WIN32
  struct timeval tv;
  tv.tv_sec = timeout;
  tv.tv_usec = 0;
  setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif  // _WIN32
}

bool TCPSocket::ShutDown(int ways) { return 0 == shutdown(socket_, ways); }

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

int64_t TCPSocket::Send(const char *data, int64_t len_data) {
  int64_t number_send;

  do {  // retry if EINTR failure appears
    number_send = send(socket_, data, len_data, 0);
  } while (number_send == -1 && errno == EINTR);
  if (number_send == -1) {
    LOG(ERROR) << "send error: " << strerror(errno);
  }

  return number_send;
}

int64_t TCPSocket::Receive(char *buffer, int64_t size_buffer) {
  int64_t number_recv;

  do {  // retry if EINTR failure appears
    number_recv = recv(socket_, buffer, size_buffer, 0);
  } while (number_recv == -1 && errno == EINTR);
  if (number_recv == -1 && errno != EAGAIN && errno != EWOULDBLOCK) {
    LOG(ERROR) << "recv error: " << strerror(errno);
  }

  return number_recv;
}

int TCPSocket::Socket() const { return socket_; }

}  // namespace network
}  // namespace dgl
