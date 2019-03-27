/*!
 *  Copyright (c) 2019 by Contributors
 * \file communicator.cc
 * \brief SocketCommunicator for DGL distributed training.
 */
#include <dmlc/logging.h>

#include "socket_communicator.h"
#include "../../c_api_common.h"
#include "../network.h"

#ifdef _WIN32
#include <<windows.h>
#else   // !_WIN32
#include <unistd.h>
#endif  // _WIN32

namespace dgl {
namespace network {

const int kTimeOut = 10;  // 10 minutes for socket timeout
const int kMaxConnection = 1024;  // 1024 maximal socket connection

bool SocketCommunicator::Initialize(bool is_sender,
                                    const char* ip,
                                    int port,
                                    int num_sender,
                                    int64_t queue_size) {
  if (is_sender) {
    is_sender_ = true;
    return InitSender(ip, port);
  } else {
    is_sender_ = false;
    return InitReceiver(ip, port, num_sender, queue_size);
  }
}

bool SocketCommunicator::InitSender(const char* ip, int port) {
  // Sender only has a client socket
  socket_.resize(1);
  socket_[0] = new TCPSocket();
  TCPSocket* client = socket_[0];
  bool bo = false;
  int try_count = 0;
  // Connect to server
  while (bo == false && try_count < kMaxTryCount) {
    if (client->Connect(ip, port)) {
      LOG(INFO) << "Connected to " << ip << ":" << port;
      return true;
    } else {
      LOG(ERROR) << "Cannot connect to " << ip << ":" << port
                 << ", try again ...";
      bo = false;
      try_count++;
      sleep(1);  // sleep 1 second and try again
    }
  }
  return false;
}

bool SocketCommunicator::InitReceiver(const char* ip,
                                      int port,
                                      int num_sender,
                                      int64_t queue_size) {
  CHECK_GE(num_sender, 1);
  CHECK_GT(queue_size, 0);
  // Init message queue
  num_sender_ = num_sender;
  queue_size_ = queue_size;
  queue_ = new MessageQueue(queue_size_, num_sender_);
  // Init socket, and socket_[0] is the server socket
  socket_.resize(num_sender+1);
  thread_.resize(num_sender);
  socket_[0] = new TCPSocket();
  TCPSocket* server = socket_[0];
  server->SetTimeout(kTimeOut * 60 * 1000);  // millsec
  // Bind socket
  if (server->Bind(ip, port) == false) {
    LOG(ERROR) << "Cannot bind to " << ip << ":" << port;
    return false;
  }
  LOG(INFO) << "Bind to " << ip << ":" << port;
  // Listen
  if (server->Listen(kMaxConnection) == false) {
    LOG(ERROR) << "Cannot listen on " << ip << ":" << port;
    return false;
  }
  LOG(INFO) << "Listen on " << ip << ":" << port << ", wait sender connect ...";
  // Accept all sender sockets
  std::string accept_ip;
  int accept_port;
  for (int i = 1; i <= num_sender_; ++i) {
    socket_[i] = new TCPSocket();
    if (server->Accept(socket_[i], &accept_ip, &accept_port) == false) {
      LOG(ERROR) << "Error on accept socket.";
      return false;
    }
    // new thread for the socket
    thread_[i-1] = new std::thread(MsgHandler, socket_[i], queue_);
    LOG(INFO) << "Accept new sender: " << accept_ip << ":" << accept_port;
  }

  return true;
}

void SocketCommunicator::MsgHandler(TCPSocket* socket, MessageQueue* queue) {
  char* buffer = new char[kMaxBufferSize];
  for (;;) {
    // First recv the size
    int64_t received_bytes = 0;
    int64_t data_size = 0;
    while (received_bytes < sizeof(int64_t)) {
      int64_t max_len = sizeof(int64_t) - received_bytes;
      int64_t tmp = socket->Receive(
        reinterpret_cast<char*>(&data_size)+received_bytes,
        max_len);
      received_bytes += tmp;
    }
    if (data_size <= 0) {
      LOG(INFO) << "Socket finish job";
      break;
    }
    // Then recv the data
    received_bytes = 0;
    while (received_bytes < data_size) {
      int64_t max_len = data_size - received_bytes;
      int64_t tmp = socket->Receive(buffer+received_bytes, max_len);
      received_bytes += tmp;
    }
    queue->Add(buffer, data_size);
  }
  delete [] buffer;
}

void SocketCommunicator::Finalize() {
  if (is_sender_) {
    FinalizeSender();
  } else {
    FinalizeReceiver();
  }
}

void SocketCommunicator::FinalizeSender() {
  // We send a size = -1 signal to notify
  // receiver to finish its job
  if (socket_[0] != nullptr) {
    int64_t size = -1;
    int64_t sent_bytes = 0;
    while (sent_bytes < sizeof(int64_t)) {
      int64_t max_len = sizeof(int64_t) - sent_bytes;
      int64_t tmp = socket_[0]->Send(
        reinterpret_cast<char*>(&size)+sent_bytes,
        max_len);
      sent_bytes += tmp;
    }
    socket_[0]->Close();
    LOG(INFO) << "Close sender socket.";
    delete socket_[0];
    socket_[0] = nullptr;
  }
}

void SocketCommunicator::FinalizeReceiver() {
  for (int i = 0; i <= num_sender_; ++i) {
    if (socket_[i] != nullptr) {
      socket_[i]->Close();
      delete socket_[i];
      socket_[i] = nullptr;
    }
  }
}

int64_t SocketCommunicator::Send(char* src, int64_t size) {
  if (!is_sender_) {
    LOG(ERROR) << "Receiver cannot invoke send() API.";
    return -1;
  }
  TCPSocket* client = socket_[0];
  // First sent the size of data
  int64_t sent_bytes = 0;
  while (sent_bytes < sizeof(int64_t)) {
    int64_t max_len = sizeof(int64_t) - sent_bytes;
    int64_t tmp = client->Send(
      reinterpret_cast<char*>(&size)+sent_bytes,
      max_len);
    sent_bytes += tmp;
  }
  // Then send the data
  sent_bytes = 0;
  while (sent_bytes < size) {
    int64_t max_len = size - sent_bytes;
    int64_t tmp = client->Send(src+sent_bytes, max_len);
    sent_bytes += tmp;
  }

  return size + sizeof(int64_t);
}

int64_t SocketCommunicator::Receive(char* dest, int64_t max_size) {
  if (is_sender_) {
    LOG(ERROR) << "Sender cannot invoke Receive() API.";
    return -1;
  }
  // Get message from the message queue
  return queue_->Remove(dest, max_size);
}

}  // namespace network
}  // namespace dgl
