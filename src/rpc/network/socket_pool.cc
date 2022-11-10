/**
 *  Copyright (c) 2021 by Contributors
 * @file socket_pool.cc
 * @brief Socket pool of nonblocking sockets for DGL distributed training.
 */
#include "socket_pool.h"

#include <dmlc/logging.h>

#include "tcp_socket.h"

#ifdef USE_EPOLL
#include <sys/epoll.h>
#endif

namespace dgl {
namespace network {

SocketPool::SocketPool() {
#ifdef USE_EPOLL
  epfd_ = epoll_create1(0);
  if (epfd_ < 0) {
    LOG(FATAL) << "SocketPool cannot create epfd";
  }
#endif
}

void SocketPool::AddSocket(
    std::shared_ptr<TCPSocket> socket, int socket_id, int events) {
  int fd = socket->Socket();
  tcp_sockets_[fd] = socket;
  socket_ids_[fd] = socket_id;

#ifdef USE_EPOLL
  epoll_event e;
  e.data.fd = fd;
  if (events == READ) {
    e.events = EPOLLIN;
  } else if (events == WRITE) {
    e.events = EPOLLOUT;
  } else if (events == READ + WRITE) {
    e.events = EPOLLIN | EPOLLOUT;
  }
  if (epoll_ctl(epfd_, EPOLL_CTL_ADD, fd, &e) < 0) {
    LOG(FATAL) << "SocketPool cannot add socket";
  }
  socket->SetNonBlocking(true);
#else
  if (tcp_sockets_.size() > 1) {
    LOG(FATAL) << "SocketPool supports only one socket if not use epoll."
                  "Please turn on USE_EPOLL on building";
  }
#endif
}

size_t SocketPool::RemoveSocket(std::shared_ptr<TCPSocket> socket) {
  int fd = socket->Socket();
  socket_ids_.erase(fd);
  tcp_sockets_.erase(fd);
#ifdef USE_EPOLL
  epoll_ctl(epfd_, EPOLL_CTL_DEL, fd, NULL);
#endif
  return socket_ids_.size();
}

SocketPool::~SocketPool() {
#ifdef USE_EPOLL
  for (auto& id : socket_ids_) {
    int fd = id.first;
    epoll_ctl(epfd_, EPOLL_CTL_DEL, fd, NULL);
  }
#endif
}

std::shared_ptr<TCPSocket> SocketPool::GetActiveSocket(int* socket_id) {
  if (socket_ids_.empty()) {
    return nullptr;
  }

  for (;;) {
    while (pending_fds_.empty()) {
      Wait();
    }
    int fd = pending_fds_.front();
    pending_fds_.pop();

    // Check if this socket is not removed
    if (socket_ids_.find(fd) != socket_ids_.end()) {
      *socket_id = socket_ids_[fd];
      return tcp_sockets_[fd];
    }
  }

  return nullptr;
}

void SocketPool::Wait() {
#ifdef USE_EPOLL
  static const int MAX_EVENTS = 10;
  epoll_event events[MAX_EVENTS];
  int nfd = epoll_wait(epfd_, events, MAX_EVENTS, -1 /*Timeout*/);
  for (int i = 0; i < nfd; ++i) {
    pending_fds_.push(events[i].data.fd);
  }
#else
  pending_fds_.push(tcp_sockets_.begin()->second->Socket());
#endif
}

}  // namespace network
}  // namespace dgl
