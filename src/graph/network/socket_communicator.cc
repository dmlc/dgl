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
#include <windows.h>
#else   // !_WIN32
#include <unistd.h>
#endif  // _WIN32

namespace dgl {
namespace network {

const int kTimeOut = 10;  // 10 minutes for socket timeout
const int kMaxConnection = 1024;  // 1024 maximal socket connection

void SocketSender::AddReceiver(const char* ip, int port, int recv_id) {
  dgl::network::Addr addr;
  addr.ip_.assign(const_cast<char*>(ip));
  addr.port_ = port;
  receiver_addr_map_[recv_id] = addr;
}

bool SocketSender::Connect() {
  // Create N sockets for Receiver
  for (const auto& r : receiver_addr_map_) {
    int ID = r.first;
    socket_map_[ID] = new TCPSocket();
    TCPSocket* client = socket_map_[ID];
    bool bo = false;
    int try_count = 0;
    const char* ip = r.second.ip_.c_str();
    int port = r.second.port_;
    while (bo == false && try_count < kMaxTryCount) {
      if (client->Connect(ip, port)) {
        LOG(INFO) << "Connected to Receiver: " << ip << ":" << port;
        bo = true;
      } else {
        LOG(ERROR) << "Cannot connect to Receiver: " << ip << ":" << port
                   << ", try again ...";
        bo = false;
        try_count++;
#ifdef _WIN32
        Sleep(1);
#else   // !_WIN32
        sleep(1);
#endif  // _WIN32
      }
    }
    if (bo == false) {
      return bo;
    }
  }
  return true;
}

int64_t SocketSender::Send(const char* data, int64_t size, int recv_id) {
  TCPSocket* client = socket_map_[recv_id];
  // First sent the size of data
  int64_t sent_bytes = 0;
  while (static_cast<size_t>(sent_bytes) < sizeof(int64_t)) {
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
    int64_t tmp = client->Send(data+sent_bytes, max_len);
    sent_bytes += tmp;
  }

  return size + sizeof(int64_t);
}

void SocketSender::Finalize() {
  // Close all sockets
  for (const auto& socket : socket_map_) {
    TCPSocket* client = socket.second;
    if (client != nullptr) {
      client->Close();
      delete client;
      client = nullptr;
    }
  }
  delete buffer_;
}

char* SocketSender::GetBuffer() {
  return buffer_;
}

void SocketSender::SetBuffer(char* buffer) {
  buffer_ = buffer;
}

bool SocketReceiver::Wait(const char* ip,
                          int port,
                          int num_sender,
                          int queue_size) {
  CHECK_GE(num_sender, 1);
  CHECK_GT(queue_size, 0);
  // Initialize message queue
  num_sender_ = num_sender;
  queue_size_ = queue_size;
  queue_ = new MessageQueue(queue_size_, num_sender_);
  // Initialize socket, and socket_[0] is server socket
  socket_.resize(num_sender_+1);
  thread_.resize(num_sender_);
  socket_[0] = new TCPSocket();
  TCPSocket* server = socket_[0];
  server->SetTimeout(kTimeOut * 60 * 1000);  // millsec
  // Bind socket
  if (server->Bind(ip, port) == false) {
    LOG(FATAL) << "Cannot bind to " << ip << ":" << port;
    return false;
  }
  LOG(INFO) << "Bind to " << ip << ":" << port;
  // Listen
  if (server->Listen(kMaxConnection) == false) {
    LOG(FATAL) << "Cannot listen on " << ip << ":" << port;
    return false;
  }
  LOG(INFO) << "Listen on " << ip << ":" << port << ", wait sender connect ...";
  // Accept all sender sockets
  std::string accept_ip;
  int accept_port;
  for (int i = 1; i <= num_sender_; ++i) {
    socket_[i] = new TCPSocket();
    if (server->Accept(socket_[i], &accept_ip, &accept_port) == false) {
      LOG(FATAL) << "Error on accept socket.";
      return false;
    }
    // create new thread for each socket
    thread_[i-1] = new std::thread(MsgHandler, socket_[i], queue_, i-1);
    LOG(INFO) << "Accept new sender: " << accept_ip << ":" << accept_port;
  }

  return true;
}

void SocketReceiver::MsgHandler(TCPSocket* socket, MessageQueue* queue, int id) {
  char* buffer = new char[kMaxBufferSize];
  for (;;) {
    // First recv the size
    int64_t received_bytes = 0;
    int64_t data_size = 0;
    while (static_cast<size_t>(received_bytes) < sizeof(int64_t)) {
      int64_t max_len = sizeof(int64_t) - received_bytes;
      int64_t tmp = socket->Receive(
        reinterpret_cast<char*>(&data_size)+received_bytes,
        max_len);
      received_bytes += tmp;
    }
    // Data_size ==-99 is a special signal to tell
    // the MsgHandler to exit the loop
    if (data_size <= 0) {
      queue->Signal(id);
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

int64_t SocketReceiver::Recv(char* dest, int64_t max_size) {
  // Get message from message queue
  return queue_->Remove(dest, max_size);
}

void SocketReceiver::Finalize() {
  for (int i = 0; i <= num_sender_; ++i) {
    if (i != 0) {  // write -99 signal to exit loop
      int64_t data_size = -99;
      queue_->Add(
        reinterpret_cast<char*>(&data_size),
        sizeof(int64_t));
    }
    if (socket_[i] != nullptr) {
      socket_[i]->Close();
      delete socket_[i];
      socket_[i] = nullptr;
    }
  }
  delete buffer_;
}

char* SocketReceiver::GetBuffer() {
  return buffer_;
}

void SocketReceiver::SetBuffer(char* buffer) {
  buffer_ = buffer;
}

}  // namespace network
}  // namespace dgl
