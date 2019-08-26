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
  IPAddr address;
  address.ip = ip_and_port[0];
  address.port = std::stoi(ip_and_port[1]);
  receiver_addrs_[recv_id] = address;
  msg_queue_[recv_id] =  std::make_shared<MessageQueue>(queue_size_);
}

bool SocketSender::Connect() {
  // Create N sockets for Receiver
  for (const auto& r : receiver_addrs_) {
    int ID = r.first;
    sockets_[ID] = std::make_shared<TCPSocket>();
    TCPSocket* client_socket = sockets_[ID].get();
    bool bo = false;
    int try_count = 0;
    const char* ip = r.second.ip.c_str();
    int port = r.second.port;
    while (bo == false && try_count < kMaxTryCount) {
      if (client_socket->Connect(ip, port)) {
        LOG(INFO) << "Connected to Receiver: " << ip << ":" << port;
        bo = true;
      } else {
        LOG(ERROR) << "Cannot connect to Receiver: " << ip << ":" << port
                   << ", try again ...";
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
    threads_[ID] = std::make_shared<std::thread>(
      SendLoop,
      client_socket,
      msg_queue_[ID].get());
  }
  return true;
}

STATUS SocketSender::Send(Message msg, int recv_id) {
  CHECK_NOTNULL(msg.data);
  CHECK_GT(msg.size, 0);
  CHECK_GE(recv_id, 0);
  // Add data message to message queue
  STATUS code = msg_queue_[recv_id]->Add(msg);
  return code;
}

void SocketSender::Finalize() {
  // Send a signal to tell the msg_queue to finish its job
  for (auto& mq : msg_queue_) {
    // wait until queue is empty
    while (mq.second->Empty() == false) {
#ifdef _WIN32
        // just loop
#else   // !_WIN32
        usleep(1000);
#endif  // _WIN32
    }
    int ID = mq.first;
    mq.second->SignalFinished(ID);
  }
  // Block main thread until all socket-threads finish their jobs
  for (auto& thread : threads_) {
    thread.second->join();
  }
  // Clear all sockets
  for (auto& socket : sockets_) {
    socket.second->Close();
  }
}

void SocketSender::SendLoop(TCPSocket* socket, MessageQueue* queue) {
  CHECK_NOTNULL(socket);
  CHECK_NOTNULL(queue);
  bool exit = false;
  while (!exit) {
    Message msg;
    STATUS code = queue->Remove(&msg);
    if (code == QUEUE_CLOSE) {
      msg.size = 0;  // send an end-signal to receiver
      exit = true;
    }
    // First send the size
    // If exit == true, we will send zero size to reciever
    int64_t sent_bytes = 0;
    while (static_cast<size_t>(sent_bytes) < sizeof(int64_t)) {
      int64_t max_len = sizeof(int64_t) - sent_bytes;
      int64_t tmp = socket->Send(
        reinterpret_cast<char*>(&msg.size)+sent_bytes,
        max_len);
      CHECK_NE(tmp, -1);
      sent_bytes += tmp;
    }
    // Then send the data
    sent_bytes = 0;
    while (sent_bytes < msg.size) {
      int64_t max_len = msg.size - sent_bytes;
      int64_t tmp = socket->Send(msg.data+sent_bytes, max_len);
      CHECK_NE(tmp, -1);
      sent_bytes += tmp;
    }
    // delete msg
    if (msg.deallocator != nullptr) {
      msg.deallocator(&msg);
    }
  }
}

/////////////////////////////////////// SocketReceiver ///////////////////////////////////////////

bool SocketReceiver::Wait(const char* addr, int num_sender) {
  CHECK_NOTNULL(addr);
  CHECK_GT(num_sender, 0);
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
    msg_queue_[i] = std::make_shared<MessageQueue>(queue_size_);
  }
  // Initialize socket and socket-thread
  server_socket_ = new TCPSocket();
  server_socket_->SetTimeout(kTimeOut * 60 * 1000);  // millsec
  // Bind socket
  if (server_socket_->Bind(ip.c_str(), port) == false) {
    LOG(FATAL) << "Cannot bind to " << ip << ":" << port;
  }
  LOG(INFO) << "Bind to " << ip << ":" << port;
  // Listen
  if (server_socket_->Listen(kMaxConnection) == false) {
    LOG(FATAL) << "Cannot listen on " << ip << ":" << port;
  }
  LOG(INFO) << "Listen on " << ip << ":" << port << ", wait sender connect ...";
  // Accept all sender sockets
  std::string accept_ip;
  int accept_port;
  for (int i = 0; i < num_sender_; ++i) {
    sockets_[i] = std::make_shared<TCPSocket>();
    if (server_socket_->Accept(sockets_[i].get(), &accept_ip, &accept_port) == false) {
      LOG(WARNING) << "Error on accept socket.";
      return false;
    }
    // create new thread for each socket
    threads_[i] = std::make_shared<std::thread>(
      RecvLoop,
      sockets_[i].get(),
      msg_queue_[i].get());
    LOG(INFO) << "Accept new sender: " << accept_ip << ":" << accept_port;
  }

  return true;
}

STATUS SocketReceiver::Recv(Message* msg, int* send_id) {
  // loop until get a message
  for (;;) {
    for (auto& mq : msg_queue_) {
      *send_id = mq.first;
      // We use non-block remove here
      STATUS code = msg_queue_[*send_id]->Remove(msg, false);
      if (code == QUEUE_EMPTY) {
        continue;  // jump to the next queue
      } else {
        return code;
      }
    }
  }
}

STATUS SocketReceiver::RecvFrom(Message* msg, int send_id) {
  // Get message from specified message queue
  STATUS code = msg_queue_[send_id]->Remove(msg);
  return code;
}

void SocketReceiver::Finalize() {
  // Send a signal to tell the message queue to finish its job
  for (auto& mq : msg_queue_) {
    // wait until queue is empty
    while (mq.second->Empty() == false) {
#ifdef _WIN32
        // just loop
#else   // !_WIN32
        usleep(1000);
#endif  // _WIN32
    }
    int ID = mq.first;
    mq.second->SignalFinished(ID);
  }
  // Block main thread until all socket-threads finish their jobs
  for (auto& thread : threads_) {
    thread.second->join();
  }
  // Clear all sockets
  for (auto& socket : sockets_) {
    socket.second->Close();
  }
}

void SocketReceiver::RecvLoop(TCPSocket* socket, MessageQueue* queue) {
  CHECK_NOTNULL(socket);
  CHECK_NOTNULL(queue);
  for (;;) {
    // If main thread had finished its job
    if (queue->EmptyAndNoMoreAdd()) {
      return;  // exit loop thread
    }
    // First recv the size
    int64_t received_bytes = 0;
    int64_t data_size = 0;
    while (static_cast<size_t>(received_bytes) < sizeof(int64_t)) {
      int64_t max_len = sizeof(int64_t) - received_bytes;
      int64_t tmp = socket->Receive(
        reinterpret_cast<char*>(&data_size)+received_bytes,
        max_len);
      CHECK_NE(tmp, -1);
      received_bytes += tmp;
    }
    if (data_size < 0) {
      LOG(FATAL) << "Recv data error (data_size: " << data_size << ")";
    } else if (data_size == 0) {
      // This is an end-signal sent by client
      return;
    } else {
      char* buffer = nullptr;
      try {
        buffer = new char[data_size];
      } catch(const std::bad_alloc&) {
        LOG(FATAL) << "Cannot allocate enough memory for message, "
                   << "(message size: " << data_size << ")";
      }
      received_bytes = 0;
      while (received_bytes < data_size) {
        int64_t max_len = data_size - received_bytes;
        int64_t tmp = socket->Receive(buffer+received_bytes, max_len);
        CHECK_NE(tmp, -1);
        received_bytes += tmp;
      }
      Message msg;
      msg.data = buffer;
      msg.size = data_size;
      msg.deallocator = DefaultMessageDeleter;
      queue->Add(msg);
    }
  }
}

}  // namespace network
}  // namespace dgl
