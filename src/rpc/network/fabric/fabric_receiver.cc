#include <dgl/network/fabric/fabric_communicator.h>
#include <dgl/network/fabric/fabric_utils.h>
#include <unistd.h>

namespace dgl {
namespace network {

bool FabricReceiver::Wait(const char* addr, int num_sender) {
  num_sender_ = num_sender;
  ctrl_ep =
    std::unique_ptr<FabricEndpoint>(FabricEndpoint::CreateTcpEndpoint(addr));
  for (size_t i = 0; i < num_sender; i++) {
    FabricAddrInfo addr;
    ctrl_ep->Recv(&addr, sizeof(FabricAddrInfo), kAddrMsg, FI_ADDR_UNSPEC,
                  true);
    ctrl_peer_fi_addr[i] = ctrl_ep->AddPeerAddr(&addr.addr);
    ctrl_ep->Recv(&addr, sizeof(FabricAddrInfo), kAddrMsg, FI_ADDR_UNSPEC,
                  true);
    peer_fi_addr[i] = fep->AddPeerAddr(&addr.addr);
    receiver_id_ = addr.id;
  }
  // socket_receiver.reset();  // Finalize socket communicator
  for (auto kv : ctrl_peer_fi_addr) {
    FabricAddrInfo info = {.id = receiver_id_, .addr = fep->fabric_ctx->addr};
    ctrl_ep->Send(&info, sizeof(FabricAddrInfo), kAddrMsg, kv.second,
                  true);  // Send back server address
  }

  for (int i = 0; i < num_sender_; ++i) {
    msg_queue_[i] =
      std::make_shared<MessageQueue>(queue_size_, peer_fi_addr.size());
  }
  should_stop_polling_ = false;
  for (auto kv : peer_fi_addr) {
    threads_[kv.first] =
      std::make_shared<std::thread>(&FabricReceiver::RecvLoop, this);
  }
  ctrl_ep.reset();
  return true;
};

STATUS FabricReceiver::Recv(Message* msg, int* send_id) {
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

void HandleCQError(struct fid_cq* cq) {
  struct fi_cq_err_entry err_entry;
  int ret = fi_cq_readerr(cq, &err_entry, 1);
  if (ret == FI_EADDRNOTAVAIL) {
    LOG(WARNING) << "fi_cq_readerr: FI_EADDRNOTAVAIL";
  } else if (ret < 0) {
    LOG(FATAL) << "fi_cq_readerr failed. Return Code: " << ret << ". ERROR: "
               << fi_cq_strerror(cq, err_entry.prov_errno, err_entry.err_data,
                                 nullptr, err_entry.err_data_size);
  } else {
    check_err(-err_entry.err, "fi_cq_read failed. retrieved error: ");
  }
}

void FabricReceiver::PollCompletionQueue(struct fi_cq_tagged_entry* cq_entries,
                                         fi_addr_t* source_addrs) {
  int ret = fi_cq_readfrom(fep->fabric_ctx->rxcq.get(), cq_entries,
                           kMaxConcurrentWorkRequest, source_addrs);
  if (ret == -FI_EAGAIN) {
    return;
  } else if (ret == -FI_EAVAIL) {
    HandleCQError(fep->fabric_ctx->rxcq.get());
  } else if (ret < 0) {
    check_err(ret, "fi_cq_read failed");
  } else {
    CHECK_NE(ret, 0) << "at least one completion event is expected";
    for (int i = 0; i < ret; ++i) {
      const auto& cq_entry = cq_entries[i];
      const auto& source_addr = source_addrs[i];
      HandleCompletionEvent(cq_entry, source_addr);
    }
  }
}

// bool FabricReceiver::Wait(const char* addr, int num_sender) {}

STATUS FabricReceiver::RecvFrom(Message* msg, int send_id) {
  // Get message from specified message queue
  STATUS code = msg_queue_[send_id]->Remove(msg);
  return code;
}

void FabricReceiver::HandleCompletionEvent(
  const struct fi_cq_tagged_entry& cq_entry, const fi_addr_t& source_addr) {
  uint64_t tag = cq_entry.tag;
  // { LOG(INFO) << "Completion event"; }
  if (tag == kSizeMsg) {
    // LOG(INFO) << "recv size";
    CHECK(cq_entry.len == sizeof(int64_t)) << "Invalid size message";
    int64_t data_size = *(int64_t*)cq_entry.buf;
    free(cq_entry.buf);
    char* buffer = nullptr;
    try {
      buffer = new char[data_size];
    } catch (const std::bad_alloc&) {
      LOG(FATAL) << "Cannot allocate enough memory for message, "
                 << "(message size: " << data_size << ")";
    }
    // LOG(INFO) << "Issue recv data event";
    fep->Recv(buffer, data_size, kDataMsg, source_addr);
    // fep->Recv()
  } else if (tag == kDataMsg) {
    Message msg;
    // LOG(INFO) << "recv data";
    msg.data = reinterpret_cast<char*>(cq_entry.buf);
    msg.size = cq_entry.len;
    msg.deallocator = DefaultMessageDeleter;
    msg_queue_[source_addr]->Add(msg);
    int64_t* size_buffer = new int64_t;  // can be optimized with buffer pool
    fep->Recv(size_buffer, sizeof(int64_t), kSizeMsg,
              source_addr);  // can use FI_ADDR_UNSPEC flag
  }
}

void FabricReceiver::RecvLoop() {
  // CHECK_NOTNULL(socket);
  // CHECK_NOTNULL(queue);
  { LOG(INFO) << "Launch Recv Loop"; }
  for (auto kv : peer_fi_addr) {
    int64_t* size_buffer = new int64_t;  // can be optimized with buffer pool
    fep->Recv(size_buffer, sizeof(int64_t), kSizeMsg, FI_ADDR_UNSPEC);
  }

  struct fi_cq_tagged_entry cq_entries[kMaxConcurrentWorkRequest];
  fi_addr_t source_addrs[kMaxConcurrentWorkRequest];
  // while (!should_stop_polling_.load()) {
  while (true) {
    // If main thread had finished its job
    // if (queue->EmptyAndNoMoreAdd()) {
    //   return;  // exit loop thread
    // }
    PollCompletionQueue(cq_entries, source_addrs);
  }
}

void FabricReceiver::Finalize() {
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
    should_stop_polling_ = true;
    // mq.second->SignalFinished(ID);
  }
  // Block main thread until all socket-threads finish their jobs
  for (auto& thread : threads_) {
    thread.second->join();
  }
  // // Clear all sockets
  // for (auto& socket : sockets_) {
  //   socket.second->Close();
  // }
  // server_socket_->Close();
  // delete server_socket_;
}
}  // namespace network
}  // namespace dgl
