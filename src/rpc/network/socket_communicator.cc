/**
 *  Copyright (c) 2019 by Contributors
 * @file communicator.cc
 * @brief SocketCommunicator for DGL distributed training.
 */
#include "socket_communicator.h"

#include <dmlc/logging.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <memory>

#include "../../c_api_common.h"
#include "socket_pool.h"

#ifdef _WIN32
#include <windows.h>
#else  // !_WIN32
#include <unistd.h>
#endif  // _WIN32

namespace dgl {
namespace network {

/////////////////////////////////////// SocketSender
//////////////////////////////////////////////

bool SocketSender::ConnectReceiver(const std::string& addr, int recv_id) {
  if (recv_id < 0) {
    LOG(FATAL) << "recv_id cannot be a negative number.";
  }
  std::vector<std::string> substring;
  std::vector<std::string> ip_and_port;
  SplitStringUsing(addr, "//", &substring);
  // Check address format
  if (substring[0] != "tcp:" || substring.size() != 2) {
    LOG(FATAL) << "Incorrect address format:" << addr
               << " Please provide right address format, "
               << "e.g, 'tcp://127.0.0.1:50051'. ";
  }
  // Get IP and port
  SplitStringUsing(substring[1], ":", &ip_and_port);
  if (ip_and_port.size() != 2) {
    LOG(FATAL) << "Incorrect address format:" << addr
               << " Please provide right address format, "
               << "e.g, 'tcp://127.0.0.1:50051'. ";
  }
  IPAddr address;
  address.ip = ip_and_port[0];
  address.port = std::stoi(ip_and_port[1]);
  receiver_addrs_[recv_id] = address;

  return true;
}

bool SocketSender::ConnectReceiverFinalize(const int max_try_times) {
  // Create N sockets for Receiver
  int receiver_count = static_cast<int>(receiver_addrs_.size());
  if (max_thread_count_ == 0 || max_thread_count_ > receiver_count) {
    max_thread_count_ = receiver_count;
  }
  sockets_.resize(max_thread_count_);
  for (const auto& r : receiver_addrs_) {
    int receiver_id = r.first;
    int thread_id = receiver_id % max_thread_count_;
    sockets_[thread_id][receiver_id] = std::make_shared<TCPSocket>();
    TCPSocket* client_socket = sockets_[thread_id][receiver_id].get();
    bool bo = false;
    int try_count = 0;
    const char* ip = r.second.ip.c_str();
    int port = r.second.port;
    while (bo == false && try_count < max_try_times) {
      if (client_socket->Connect(ip, port)) {
        bo = true;
      } else {
        if (try_count % 200 == 0 && try_count != 0) {
          // every 600 seconds show this message
          LOG(INFO) << "Trying to connect receiver: " << ip << ":" << port;
        }
        try_count++;
        std::this_thread::sleep_for(std::chrono::seconds(3));
      }
    }
    if (bo == false) {
      return bo;
    }
  }

  for (int thread_id = 0; thread_id < max_thread_count_; ++thread_id) {
    msg_queue_.push_back(std::make_shared<MessageQueue>(queue_size_));
    // Create a new thread for this socket connection
    threads_.push_back(std::make_shared<std::thread>(
        SendLoop, sockets_[thread_id], msg_queue_[thread_id]));
  }

  return true;
}

void SocketSender::Send(const rpc::RPCMessage& msg, int recv_id) {
  std::shared_ptr<std::string> zerocopy_blob(new std::string());
  StreamWithBuffer zc_write_strm(zerocopy_blob.get(), true);
  zc_write_strm.Write(msg);
  int32_t nonempty_ndarray_count = zc_write_strm.buffer_list().size();
  zerocopy_blob->append(
      reinterpret_cast<char*>(&nonempty_ndarray_count), sizeof(int32_t));
  Message rpc_meta_msg;
  rpc_meta_msg.data = const_cast<char*>(zerocopy_blob->data());
  rpc_meta_msg.size = zerocopy_blob->size();
  rpc_meta_msg.deallocator = [zerocopy_blob](Message*) {};
  CHECK_EQ(Send(rpc_meta_msg, recv_id), ADD_SUCCESS);
  // send real ndarray data
  for (auto ptr : zc_write_strm.buffer_list()) {
    Message ndarray_data_msg;
    ndarray_data_msg.data = reinterpret_cast<char*>(ptr.data);
    if (ptr.size == 0) {
      LOG(FATAL) << "Cannot send a empty NDArray.";
    }
    ndarray_data_msg.size = ptr.size;
    NDArray tensor = ptr.tensor;
    ndarray_data_msg.deallocator = [tensor](Message*) {};
    CHECK_EQ(Send(ndarray_data_msg, recv_id), ADD_SUCCESS);
  }
}

STATUS SocketSender::Send(Message msg, int recv_id) {
  CHECK_NOTNULL(msg.data);
  CHECK_GT(msg.size, 0);
  CHECK_GE(recv_id, 0);
  msg.receiver_id = recv_id;
  // Add data message to message queue
  STATUS code = msg_queue_[recv_id % max_thread_count_]->Add(msg);
  return code;
}

void SocketSender::Finalize() {
  // Send a signal to tell the msg_queue to finish its job
  for (int i = 0; i < max_thread_count_; ++i) {
    // wait until queue is empty
    auto& mq = msg_queue_[i];
    while (mq->Empty() == false) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    // All queues have only one producer, which is main thread, so
    // the producerID argument here should be zero.
    mq->SignalFinished(0);
  }
  // Block main thread until all socket-threads finish their jobs
  for (auto& thread : threads_) {
    thread->join();
  }
  // Clear all sockets
  for (auto& group_sockets_ : sockets_) {
    for (auto& socket : group_sockets_) {
      socket.second->Close();
    }
  }
}

void SendCore(Message msg, TCPSocket* socket) {
  // First send the size
  // If exit == true, we will send zero size to reciever
  int64_t sent_bytes = 0;
  while (static_cast<size_t>(sent_bytes) < sizeof(int64_t)) {
    int64_t max_len = sizeof(int64_t) - sent_bytes;
    int64_t tmp =
        socket->Send(reinterpret_cast<char*>(&msg.size) + sent_bytes, max_len);
    CHECK_NE(tmp, -1);
    sent_bytes += tmp;
  }
  // Then send the data
  sent_bytes = 0;
  while (sent_bytes < msg.size) {
    int64_t max_len = msg.size - sent_bytes;
    int64_t tmp = socket->Send(msg.data + sent_bytes, max_len);
    CHECK_NE(tmp, -1);
    sent_bytes += tmp;
  }
  // delete msg
  if (msg.deallocator != nullptr) {
    msg.deallocator(&msg);
  }
}

void SocketSender::SendLoop(
    std::unordered_map<int, std::shared_ptr<TCPSocket>> sockets,
    std::shared_ptr<MessageQueue> queue) {
  for (;;) {
    Message msg;
    STATUS code = queue->Remove(&msg);
    if (code == QUEUE_CLOSE) {
      msg.size = 0;  // send an end-signal to receiver
      for (auto& socket : sockets) {
        SendCore(msg, socket.second.get());
      }
      break;
    }
    SendCore(msg, sockets[msg.receiver_id].get());
  }
}

/////////////////////////////////////// SocketReceiver
//////////////////////////////////////////////
bool SocketReceiver::Wait(const std::string& addr, int num_sender) {
  CHECK_GT(num_sender, 0);
  std::vector<std::string> substring;
  std::vector<std::string> ip_and_port;
  SplitStringUsing(addr, "//", &substring);
  // Check address format
  if (substring[0] != "tcp:" || substring.size() != 2) {
    LOG(FATAL) << "Incorrect address format:" << addr
               << " Please provide right address format, "
               << "e.g, 'tcp://127.0.0.1:50051'. ";
  }
  // Get IP and port
  SplitStringUsing(substring[1], ":", &ip_and_port);
  if (ip_and_port.size() != 2) {
    LOG(FATAL) << "Incorrect address format:" << addr
               << " Please provide right address format, "
               << "e.g, 'tcp://127.0.0.1:50051'. ";
  }
  std::string ip = ip_and_port[0];
  int port = stoi(ip_and_port[1]);
  // Initialize message queue for each connection
  num_sender_ = num_sender;
#ifdef USE_EPOLL
  if (max_thread_count_ == 0 || max_thread_count_ > num_sender_) {
    max_thread_count_ = num_sender_;
  }
#else
  max_thread_count_ = num_sender_;
#endif
  // Initialize socket and socket-thread
  server_socket_ = new TCPSocket();
  // Bind socket
  if (server_socket_->Bind(ip.c_str(), port) == false) {
    LOG(FATAL) << "Cannot bind to " << ip << ":" << port;
  }

  // Listen
  if (server_socket_->Listen(kMaxConnection) == false) {
    LOG(FATAL) << "Cannot listen on " << ip << ":" << port;
  }
  // Accept all sender sockets
  std::string accept_ip;
  int accept_port;
  sockets_.resize(max_thread_count_);
  for (int i = 0; i < num_sender_; ++i) {
    int thread_id = i % max_thread_count_;
    auto socket = std::make_shared<TCPSocket>();
    sockets_[thread_id][i] = socket;
    msg_queue_[i] = std::make_shared<MessageQueue>(queue_size_);
    if (server_socket_->Accept(socket.get(), &accept_ip, &accept_port) ==
        false) {
      LOG(WARNING) << "Error on accept socket.";
      return false;
    }
  }
  mq_iter_ = msg_queue_.begin();

  for (int thread_id = 0; thread_id < max_thread_count_; ++thread_id) {
    // create new thread for each socket
    threads_.push_back(std::make_shared<std::thread>(
        RecvLoop, sockets_[thread_id], msg_queue_, &queue_sem_));
  }

  return true;
}

rpc::RPCStatus SocketReceiver::Recv(rpc::RPCMessage* msg, int timeout) {
  Message rpc_meta_msg;
  int send_id;
  auto status = Recv(&rpc_meta_msg, &send_id, timeout);
  if (status == QUEUE_EMPTY) {
    DLOG(WARNING) << "Timed out when trying to receive rpc meta data after "
                  << timeout << " milliseconds.";
    return rpc::kRPCTimeOut;
  }
  CHECK_EQ(status, REMOVE_SUCCESS);
  char* count_ptr = rpc_meta_msg.data + rpc_meta_msg.size - sizeof(int32_t);
  int32_t nonempty_ndarray_count = *(reinterpret_cast<int32_t*>(count_ptr));
  // Recv real ndarray data
  std::vector<void*> buffer_list(nonempty_ndarray_count);
  for (int i = 0; i < nonempty_ndarray_count; ++i) {
    Message ndarray_data_msg;
    // As meta message has been received, data message is always expected unless
    // connection is closed.
    STATUS status;
    do {
      status = RecvFrom(&ndarray_data_msg, send_id, timeout);
      if (status == QUEUE_EMPTY) {
        DLOG(WARNING)
            << "Timed out when trying to receive rpc ndarray data after "
            << timeout << " milliseconds.";
      }
    } while (status == QUEUE_EMPTY);
    CHECK_EQ(status, REMOVE_SUCCESS);
    buffer_list[i] = ndarray_data_msg.data;
  }
  StreamWithBuffer zc_read_strm(
      rpc_meta_msg.data, rpc_meta_msg.size - sizeof(int32_t), buffer_list);
  zc_read_strm.Read(msg);
  rpc_meta_msg.deallocator(&rpc_meta_msg);
  return rpc::kRPCSuccess;
}

STATUS SocketReceiver::Recv(Message* msg, int* send_id, int timeout) {
  // queue_sem_ is a semaphore indicating how many elements in multiple
  // message queues.
  // When calling queue_sem_.Wait(), this Recv will be suspended until
  // queue_sem_ > 0 or specified timeout expires, decrease queue_sem_ by 1,
  // then start to fetch a message.
  if (!queue_sem_.TimedWait(timeout)) {
    return QUEUE_EMPTY;
  }
  for (;;) {
    for (; mq_iter_ != msg_queue_.end(); ++mq_iter_) {
      STATUS code = mq_iter_->second->Remove(msg, false);
      if (code == QUEUE_EMPTY) {
        continue;  // jump to the next queue
      } else {
        *send_id = mq_iter_->first;
        ++mq_iter_;
        return code;
      }
    }
    mq_iter_ = msg_queue_.begin();
  }
  LOG(ERROR)
      << "Failed to remove message from queue due to unexpected queue status.";
  return QUEUE_CLOSE;
}

STATUS SocketReceiver::RecvFrom(Message* msg, int send_id, int timeout) {
  // Get message from specified message queue
  if (!queue_sem_.TimedWait(timeout)) {
    return QUEUE_EMPTY;
  }
  STATUS code = msg_queue_[send_id]->Remove(msg);
  return code;
}

void SocketReceiver::Finalize() {
  // Send a signal to tell the message queue to finish its job
  for (auto& mq : msg_queue_) {
    // wait until queue is empty
    while (mq.second->Empty() == false) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    mq.second->SignalFinished(mq.first);
  }
  // Block main thread until all socket-threads finish their jobs
  for (auto& thread : threads_) {
    thread->join();
  }
  // Clear all sockets
  for (auto& group_sockets : sockets_) {
    for (auto& socket : group_sockets) {
      socket.second->Close();
    }
  }
  server_socket_->Close();
  delete server_socket_;
}

int64_t RecvDataSize(TCPSocket* socket) {
  int64_t received_bytes = 0;
  int64_t data_size = 0;
  while (static_cast<size_t>(received_bytes) < sizeof(int64_t)) {
    int64_t max_len = sizeof(int64_t) - received_bytes;
    int64_t tmp = socket->Receive(
        reinterpret_cast<char*>(&data_size) + received_bytes, max_len);
    if (tmp == -1) {
      if (received_bytes > 0) {
        // We want to finish reading full data_size
        continue;
      }
      return -1;
    }
    received_bytes += tmp;
  }
  return data_size;
}

void RecvData(
    TCPSocket* socket, char* buffer, const int64_t& data_size,
    int64_t* received_bytes) {
  while (*received_bytes < data_size) {
    int64_t max_len = data_size - *received_bytes;
    int64_t tmp = socket->Receive(buffer + *received_bytes, max_len);
    if (tmp == -1) {
      // Socket not ready, no more data to read
      return;
    }
    *received_bytes += tmp;
  }
}

void SocketReceiver::RecvLoop(
    std::unordered_map<
        int /* Sender (virtual) ID */, std::shared_ptr<TCPSocket>>
        sockets,
    std::unordered_map<
        int /* Sender (virtual) ID */, std::shared_ptr<MessageQueue>>
        queues,
    runtime::Semaphore* queue_sem) {
  std::unordered_map<int, std::unique_ptr<RecvContext>> recv_contexts;
  SocketPool socket_pool;
  for (auto& socket : sockets) {
    auto& sender_id = socket.first;
    socket_pool.AddSocket(socket.second, sender_id);
    recv_contexts[sender_id] = std::unique_ptr<RecvContext>(new RecvContext());
  }

  // Main loop to receive messages
  for (;;) {
    int sender_id;
    // Get active socket using epoll
    std::shared_ptr<TCPSocket> socket = socket_pool.GetActiveSocket(&sender_id);
    if (queues[sender_id]->EmptyAndNoMoreAdd()) {
      // This sender has already stopped
      if (socket_pool.RemoveSocket(socket) == 0) {
        return;
      }
      continue;
    }

    // Nonblocking socket might be interrupted at any point. So we need to
    // store the partially received data
    std::unique_ptr<RecvContext>& ctx = recv_contexts[sender_id];
    int64_t& data_size = ctx->data_size;
    int64_t& received_bytes = ctx->received_bytes;
    char*& buffer = ctx->buffer;

    if (data_size == -1) {
      // This is a new message, so receive the data size first
      data_size = RecvDataSize(socket.get());
      if (data_size > 0) {
        try {
          buffer = new char[data_size];
        } catch (const std::bad_alloc&) {
          LOG(FATAL) << "Cannot allocate enough memory for message, "
                     << "(message size: " << data_size << ")";
        }
        received_bytes = 0;
      } else if (data_size == 0) {
        // Received stop signal
        if (socket_pool.RemoveSocket(socket) == 0) {
          return;
        }
      }
    }

    RecvData(socket.get(), buffer, data_size, &received_bytes);
    if (received_bytes >= data_size) {
      // Full data received, create Message and push to queue
      Message msg;
      msg.data = buffer;
      msg.size = data_size;
      msg.deallocator = DefaultMessageDeleter;
      queues[sender_id]->Add(msg);

      // Reset recv context
      data_size = -1;

      // Signal queue semaphore
      queue_sem->Post();
    }
  }
}

}  // namespace network
}  // namespace dgl
