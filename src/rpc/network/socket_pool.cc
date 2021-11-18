/*!
 *  Copyright (c) 2021 by Contributors
 * \file socket_pool.cc
 * \brief Socket pool of nonblocking sockets for DGL distributed training.
 */
#include "socket_pool.h"
#include <dmlc/logging.h>
#if defined(__linux__)
#include <sys/epoll.h>
#endif
#include "tcp_socket.h"

namespace dgl {
namespace network {

SocketPool::SocketPool() {
#if defined(__linux__)
  epfd_ = epoll_create1(0);
  if (epfd_ < 0) {
    LOG(FATAL) << "SocketPool cannot create epfd";
  }
#endif
}

SocketPool::SocketPool(SocketPool &&) {
#if defined(__linux__)
  epfd_ = epoll_create1(0);
  if (epfd_ < 0) {
    LOG(FATAL) << "SocketPool cannot create epfd";
  }
#endif
}

void SocketPool::AddSocket(std::shared_ptr<TCPSocket> socket, int socket_id,
                           int events) {
  int fd = socket->Socket();
  std::unique_lock<std::mutex> lk(mtx_);
  tcp_sockets_[fd] = socket;
  socket_ids_[fd] = socket_id;
  lk.unlock();

#if defined(__linux__)
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
    LOG(FATAL) << "SocketPool supports only one socket for non-linux platforms";
  }
#endif
}

size_t SocketPool::RemoveSocket(std::shared_ptr<TCPSocket> socket) {
  int fd = socket->Socket();
  std::unique_lock<std::mutex> lk(mtx_);
  socket_ids_.erase(fd);
  tcp_sockets_.erase(fd);
  size_t ret = socket_ids_.size();
  lk.unlock();
#if defined(__linux__)
  epoll_ctl(epfd_, EPOLL_CTL_DEL, fd, NULL);
#endif
  return ret;
}

SocketPool::~SocketPool() {
#if defined(__linux__)
  std::lock_guard<std::mutex> lk(mtx_);
  for (auto &id : socket_ids_) {
    int fd = id.first;
    epoll_ctl(epfd_, EPOLL_CTL_DEL, fd, NULL);
  }
#endif
}

std::shared_ptr<TCPSocket> SocketPool::GetActiveSocket(int *socket_id) {
  if (socket_ids_.empty()) {
    return nullptr;
  }

  Wait();
  while (!pending_fds_.empty()) {
    int fd = pending_fds_.front();
    pending_fds_.pop();
    std::lock_guard<std::mutex> lk(mtx_);
    // Check if this socket is not removed
    if (socket_ids_.find(fd) != socket_ids_.end()) {
      *socket_id = socket_ids_[fd];
      return tcp_sockets_[fd];
    }
  }

  return nullptr;
}

void SocketPool::Wait() {
#if defined(__linux__)
  static const int MAX_EVENTS = 10;
  epoll_event events[MAX_EVENTS];
  int nfd = epoll_wait(epfd_, events, MAX_EVENTS, 0 /*Timeout*/);
  for (int i = 0; i < nfd; ++i) {
    pending_fds_.push(events[i].data.fd);
  }
#else
  pending_fds_.push(tcp_sockets_.begin()->second->Socket());
#endif
}

} // namespace network
} // namespace dgl
