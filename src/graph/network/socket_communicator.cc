/*!
 *  Copyright (c) 2019 by Contributors
 * \file communicator.cc
 * \brief SocketCommunicator for DGL distributed training.
 */
#include <dmlc/logging.h>

#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "socket_communicator.h"
#include "../../c_api_common.h"

#ifdef _WIN32
#include <windows.h>
#else   // !_WIN32
#include <unistd.h>
#endif  // _WIN32

namespace dgl {
namespace network {

using dgl::network::SplitStringUsing;
using dgl::network::Addr;


/////////////////////////////////////// SocketSender ///////////////////////////////////////////


void SocketSender::AddReceiver(const char* addr, int recv_id) {
  CHECK_NOTNULL(addr);
  if (recv_id < 0) {
    LOG(FATAL) << "recv_id cannot be a negative number.";
  }
  std::vector<std::string> substring;
  std::vector<std::string> ip_and_port;
  SplitStringUsing(addr, "//", &substring);
  // Check address format
  if (substring[0] != "socket:" || substring.size() != 2) {
    LOG(FATAL) << "Incorrect address format:" << addr
               << " Please provide right address format, "
               << "e.g, 'socket://127.0.0.1:50051'. ";
  }
  // Get IP and port
  SplitStringUsing(substring[1], ":", &ip_and_port);
  if (ip_and_port.size() != 2) {
    LOG(FATAL) << "Incorrect address format:" << addr
               << " Please provide right address format, "
               << "e.g, 'socket://127.0.0.1:50051'. ";
  }
  Addr address;
  address.ip_ = ip_and_port[0];
  address.port_ = std::stoi(ip_and_port[1]);
  receiver_addrs_[recv_id] = address;
  msg_queue_[recv_id] = new MessageQueue(queue_size_);
}

bool SocketSender::Connect() {
  static int kMaxTryCount = 1024;  // maximal connection: 1024
  // Create N sockets for Receiver
  for (const auto& r : receiver_addrs_) {
    int ID = r.first;
    sockets_[ID] = new TCPSocket();
    TCPSocket* client_socket = sockets_[ID];
    bool bo = false;
    int try_count = 0;
    const char* ip = r.second.ip_.c_str();
    int port = r.second.port_;
    while (bo == false && try_count < kMaxTryCount) {
      if (client_socket->Connect(ip, port)) {
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
    // Create a new thread for this socket connection
    threads_[ID] = new std::thread(SendLoop, client_socket, msg_queue_[ID]);
  }
  return true;
}

int send_count = 0;
int loop_count = 0;

int64_t SocketSender::Send(const char* data, int64_t size, int recv_id) {
  CHECK_NOTNULL(data);
  if (recv_id < 0) {
    LOG(FATAL) << "recv_id cannot be a negative number.";
  }
  // Add data to message queue
  int64_t send_size = msg_queue_[recv_id]->Add(data, size);
  if (send_size < 0) {
    LOG(FATAL) << "Error on pushing data to message queue.";
  }
  send_count++;
  return send_size;
}

void SocketSender::Finalize() {
  // Send a signal to tell the msg_queue to finish its job
  for (auto& mq : msg_queue_) {
    // wait until queue is empty
    while (mq.second->Empty() == false) {
      usleep(1000);
    }
    int ID = mq.first;
    mq.second->Signal(ID);
  }
  // Block main thread until all socket-threads finish their jobs
  for (auto& thread : threads_) {
    thread.second->join();
  }
  // Clear all sockets
  for (auto& socket : sockets_) {
    socket.second->ShutDown(SHUT_RDWR);
    socket.second->Close();
  }
}

void SocketSender::SendLoop(TCPSocket* socket, MessageQueue* queue) {
  CHECK_NOTNULL(socket);
  CHECK_NOTNULL(queue);
  bool exit = false;
  while (exit == false) {
    // If main thread had finished its job
    if (queue->EmptyAndNoMoreAdd()) {
      exit = true;
    }
    int64_t data_size = 0;
    char* data = queue->Remove(&data_size);
    if (data_size < 0) {
      LOG(FATAL) << "Remove data from msg_queue error: " << data_size;
    }
    // First send the data size
    // If exit == true, we will send zero size to reciever
    int64_t sent_bytes = 0;
    while (static_cast<size_t>(sent_bytes) < sizeof(int64_t)) {
      int64_t max_len = sizeof(int64_t) - sent_bytes;
      int64_t tmp = socket->Send(
      reinterpret_cast<char*>(&data_size)+sent_bytes, max_len);
      if (tmp == -1) {
        LOG(FATAL) << "Socket send error.";
      }
      sent_bytes += tmp;
    }
    // Then send the data
    sent_bytes = 0;
    while (sent_bytes < data_size) {
      int64_t max_len = data_size - sent_bytes;
      int64_t tmp = socket->Send(data+sent_bytes, max_len);
      if (tmp == -1) {
        LOG(FATAL) << "Socket send error."; 
      }
      sent_bytes += tmp;
    }
    loop_count++;
  }
}

/////////////////////////////////////// SocketReceiver ///////////////////////////////////////////

bool SocketReceiver::Wait(const char* addr, int num_sender) {
  static int kTimeOut = 10;          // 10 minutes for socket timeout
  static int kMaxConnection = 1024;  // maximal connection: 1024
  CHECK_NOTNULL(addr);
  if (num_sender < 0) {
    LOG(FATAL) << "num_sender cannot be a negative number.";
  }
  std::vector<std::string> substring;
  std::vector<std::string> ip_and_port;
  SplitStringUsing(addr, "//", &substring);
  // Check address format
  if (substring[0] != "socket:" || substring.size() != 2) {
    LOG(FATAL) << "Incorrect address format:" << addr
               << " Please provide right address format, "
               << "e.g, 'socket://127.0.0.1:50051'. ";
  }
  // Get IP and port
  SplitStringUsing(substring[1], ":", &ip_and_port);
  if (ip_and_port.size() != 2) {
    LOG(FATAL) << "Incorrect address format:" << addr
               << " Please provide right address format, "
               << "e.g, 'socket://127.0.0.1:50051'. ";
  }
  std::string ip = ip_and_port[0];
  int port = stoi(ip_and_port[1]);
  // Initialize message queue for each connection
  num_sender_ = num_sender;
  for (int i = 0; i < num_sender_; ++i) {
    msg_queue_[i] = new MessageQueue(queue_size_);
  }
  // Initialize socket and socket-thread
  server_socket_ = new TCPSocket();
  server_socket_->SetTimeout(kTimeOut * 60 * 1000);  // millsec
  // Bind socket
  if (server_socket_->Bind(ip.c_str(), port) == false) {
    LOG(FATAL) << "Cannot bind to " << ip << ":" << port;
    return false;
  }
  LOG(INFO) << "Bind to " << ip << ":" << port;
  // Listen
  if (server_socket_->Listen(kMaxConnection) == false) {
    LOG(FATAL) << "Cannot listen on " << ip << ":" << port;
    return false;
  }
  LOG(INFO) << "Listen on " << ip << ":" << port << ", wait sender connect ...";
  // Accept all sender sockets
  std::string accept_ip;
  int accept_port;
  for (int i = 0; i < num_sender_; ++i) {
    sockets_[i] = new TCPSocket();
    if (server_socket_->Accept(sockets_[i], &accept_ip, &accept_port) == false) {
      LOG(FATAL) << "Error on accept socket.";
      return false;
    }
    // create new thread for each socket
    threads_[i] = new std::thread(RecvLoop, sockets_[i], msg_queue_[i]);
    LOG(INFO) << "Accept new sender: " << accept_ip << ":" << accept_port;
  }

  return true;
}

char* SocketReceiver::Recv(int64_t* size, int* send_id) {
  // Get the top message
  for (;;) { // loop until get a message
    for (auto& mq : msg_queue_) {
      if (mq.second->Empty() == false) {
        *send_id = mq.first;
        return msg_queue_[*send_id]->Remove(size);
      }
    }
  }
}

char* SocketReceiver::RecvFrom(int64_t* size, int send_id) {
  // Get message from specified message queueC++
  return msg_queue_[send_id]->Remove(size);
}

void SocketReceiver::Finalize() {
  // Send a signal to tell the message queue to finish its job
  for (auto& mq : msg_queue_) {
    // wait until queue is empty
    while (mq.second->Empty() == false) {
      usleep(1000);
    }
    int ID = mq.first;
    mq.second->Signal(ID);
  }
  // Block main thread until all socket-threads finish their jobs
  for (auto& thread : threads_) {
    thread.second->join();
  }
  // Clear all sockets
  for (auto& socket : sockets_) {
    socket.second->ShutDown(SHUT_RDWR);
    socket.second->Close();
  }
}

void SocketReceiver::RecvLoop(TCPSocket* socket, MessageQueue* queue) {
  CHECK_NOTNULL(socket);
  CHECK_NOTNULL(queue);
  for (;;) {
    // If main thread had finished its job
    if (queue->EmptyAndNoMoreAdd()) {
      return; // exit loop thread
    }
    // First recv the size
    int64_t received_bytes = 0;
    int64_t data_size = 0;
    while (static_cast<size_t>(received_bytes) < sizeof(int64_t)) {
      int64_t max_len = sizeof(int64_t) - received_bytes;
      int64_t tmp = socket->Receive(
        reinterpret_cast<char*>(&data_size)+received_bytes,
        max_len);
      if (tmp == -1) {
        LOG(FATAL) << "Socket recv error.";
      }
      received_bytes += tmp;
    }
    if (data_size < 0) {
      LOG(FATAL) << "Recv data error (data_size: " << data_size << ")";
    } else if (data_size == 0) { // This is a end-signal sent by client
      return;
    } else {
      char* buffer = new char[data_size];
      received_bytes = 0;
      while (received_bytes < data_size) {
        int64_t max_len = data_size - received_bytes;
        int64_t tmp = socket->Receive(buffer+received_bytes, max_len);
        if (tmp == -1) {
          LOG(FATAL) << "Socket recv error.";
        }
        received_bytes += tmp;
      }
      if (queue->Add(buffer, data_size) < 0) {
        LOG(FATAL) << "Push data into msg_queue error.";
      }
      delete [] buffer;
    }
  }
}

}  // namespace network
}  // namespace dgl
