#include <dgl/network/fabric/fabric_communicator.h>
#include <dgl/network/fabric/fabric_utils.h>
#include <unistd.h>

#include "rdma/fi_domain.h"

namespace dgl {
namespace network {

bool FabricReceiver::Wait(const char* addr, int num_sender) {
  num_sender_ = num_sender;
  ctrl_ep =
    std::unique_ptr<FabricEndpoint>(FabricEndpoint::CreateCtrlEndpoint(addr));
  for (size_t i = 0; i < num_sender; i++) {
    FabricAddrInfo addr;
    ctrl_ep->Recv(&addr, sizeof(FabricAddrInfo), kCtrlAddrMsg, FI_ADDR_UNSPEC,
                  true);
    ctrl_peer_fi_addr[i] = ctrl_ep->AddPeerAddr(&addr.addr);
    receiver_id_ = addr.receiver_id;
    ctrl_ep->Recv(&addr, sizeof(FabricAddrInfo), kFiAddrMsg,
                  ctrl_peer_fi_addr[i], true);
    peer_fi_addr[i] = fep->AddPeerAddr(&addr.addr);
    // fi_to_id[peer_fi_addr[i]] = i;
    FabricAddrInfo info = {.sender_id = static_cast<int>(i),
                           .receiver_id = receiver_id_,
                           .addr = fep->fabric_ctx->addr};
    ctrl_ep->Send(&info, sizeof(FabricAddrInfo), kFiAddrMsg,
                  ctrl_peer_fi_addr[i],
                  true);  // Send back server address
    msg_queue_[i] = std::make_shared<MessageQueue>(queue_size_);
    // LOG(INFO) << "New peer fi addr" << peer_fi_addr[i] << "  id: " << i;
    fep->Send(handshake_msg.c_str(), handshake_msg.length(), kIgnoreMsg,
              peer_fi_addr[i]);
    fep->Recv(handshake_buffer, handshake_msg.length(), kIgnoreMsg,
              FI_ADDR_UNSPEC);
  }
  // LOG(INFO) << "Finish Initialization";
  // socket_receiver.reset();  // Finalize socket communicator
  // for (auto& kv : ctrl_peer_fi_addr) {
  // }

  // for (int i = 0; i < num_sender_; ++i) {
  //   LOG(INFO) << "Init Queue: " << peer_fi_addr[i];
  //  }
  //  / for (auto& kv : peer_fi_addr) {
  // //   fep->Send(handshake_msg.c_str(), handshake_msg.size(), kAddrMsg,
  //      kv.second);
  //  / }
  //   ll_thread = d::make_s rl_ep.reset();
  poll_thread = std::make_shared<std::thread>(&FabricReceiver::RecvLoop, this);

  // FabricAddrInfo fep_info;
  // for (auto& kv : ctrl_peer_fi_addr) {
  //   ctrl_ep->Send(&fep_info, sizeof(FabricAddrInfo), kAddrMsg, kv.second,
  //   true);
  // }
  return true;
}

STATUS FabricReceiver::Recv(Message* msg, int* send_id) {
  // loop until get a message
  for (;;) {
    for (auto& mq : msg_queue_) {
      *send_id = mq.first;
      // We use non-block remove here
      // {LOG(INFO) <<"RECV queue: "<< peer_fi_addr.at(*send_id);}
      // CHECK(peer_fi_addr.find(*send_id) != peer_fi_addr.end()) << *send_id;
      // CHECK(msg_queue_.find(peer_fi_addr.at(*send_id)) != msg_queue_.end())
      // << peer_fi_addr.at(*send_id);
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

bool FabricReceiver::PollCompletionQueue(struct fi_cq_tagged_entry* cq_entries,
                                         fi_addr_t* source_addrs) {
  // *source_addrs = 1;
  // fi_addr_t source_addrs2 = 999;
  int ret = fi_cq_read(fep->fabric_ctx->rxcq.get(), cq_entries, 1);
  // int ret = fi_cq_read(fep->fabric_ctx->rxcq.get(), cq_entries, 1);

  // if (ret > 0) {
  //   LOG(INFO) << "Recv fromAA: " << source_addrs2;
  // }
  bool keep_polling = true;
  if (ret == -FI_EAGAIN) {
    return true;
  } else if (ret == -FI_EAVAIL) {
    HandleCQError(fep->fabric_ctx->rxcq.get());
  } else if (ret < 0) {
    check_err(ret, "fi_cq_read failed");
  } else {
    CHECK_NE(ret, 0) << "at least one completion event is expected";
    // LOG(INFO) << ret << " Events received from: " << source_addrs2;
    for (int i = 0; i < ret; ++i) {
      // const auto& cq_entry = cq_entries[i];
      // const auto& source_addr = source_addrs[i];
      // CHECK_NE(source_addr, FI_ADDR_NOTAVAIL);

      // FabricAddr readable_addr, addr;
      // fi_av_lookup(fep->fabric_ctx->av.get(), source_addrs2, addr.name,
      //         false,     &addr.len);
      // fi_av_straddr(fep->fabric_ctx->av.get(), addr.name, readable_addr.name,
      //               &readable_addr.len);
      // std::string readable_peer_addr =
      //   std::string(readable_addr.name, readable_addr.len);
      // { LOG(INFO) << "Recv From readable: " << readable_peer_addr; }
      if (!HandleCompletionEvent(cq_entries[i], 0)) {
        keep_polling = false;
      };
    }
  }
  return keep_polling;
}

// bool FabricReceiver::Wait(const char* addr, int num_sender) {}

STATUS FabricReceiver::RecvFrom(Message* msg, int send_id) {
  // Get message from specified message queue
  STATUS code = msg_queue_.at(send_id)->Remove(msg);
  return code;
}

bool FabricReceiver::HandleCompletionEvent(
  const struct fi_cq_tagged_entry& cq_entry, const fi_addr_t& source_addr) {
  uint64_t tag = cq_entry.tag & MsgTagMask;
  // { LOG(INFO) << "Completion event"; }
  if (tag == kSizeMsg) {
    // LOG(INFO) << "recv size";
    CHECK(cq_entry.len == sizeof(int64_t)) << "Invalid size message";
    int64_t data_size = *(int64_t*)cq_entry.buf;

    // LOG(INFO) << "Receive Size: " << data_size << ". From "
    //           << (cq_entry.tag & IdMask);
    // free(cq_entry.buf);
    char* buffer = nullptr;
    if (data_size == 0) {
      return false;
    }
    try {
      buffer = new char[data_size];
    } catch (const std::bad_alloc&) {
      LOG(FATAL) << "Cannot allocate enough memory for message, "
                 << "(message size: " << data_size << ")";
    }
    // fep->Recv(buffer, data_size, kDataMsg, source_addr);
    fep->Recv(buffer, data_size, kDataMsg | (cq_entry.tag & IdMask),
              FI_ADDR_UNSPEC, false);
  } else if (tag == kDataMsg) {
    Message msg;
    msg.data = reinterpret_cast<char*>(cq_entry.buf);
    msg.size = cq_entry.len;
    msg.deallocator = DefaultMessageDeleter;
    msg_queue_.at((cq_entry.tag & IdMask))->Add(msg);
    int64_t* size_buffer = new int64_t;  // can be optimized with buffer pool
    fep->Recv(size_buffer, sizeof(int64_t), kSizeMsg, FI_ADDR_UNSPEC, false,
              0xFFFF);  // can use FI_ADDR_UNSPEC flag
  } else {
    if (tag != kIgnoreMsg) {
      LOG(INFO) << "Invalid tag";
    }
  }
  return true;
}

void FabricReceiver::RecvLoop() {
  // LOG(INFO) << "Start recv loop";
  for (size_t i = 0; i < peer_fi_addr.size(); i++) {
    int64_t* size_buffer = new int64_t;  // can be optimized with buffer pool
    fep->Recv(size_buffer, sizeof(int64_t), kSizeMsg, FI_ADDR_UNSPEC, false,
              0xFFFF);
  }

  struct fi_cq_tagged_entry cq_entries[kMaxConcurrentWorkRequest];
  fi_addr_t source_addrs[kMaxConcurrentWorkRequest];
  int finished_count = 0;
  // while (!should_stop_polling_.load()) {
  while (finished_count != num_sender_) {
    // If main thread had finished its job
    for (auto& kv : msg_queue_) {
      if (kv.second->EmptyAndNoMoreAdd()) {
        return;  // exit loop thread
      }
    }
    if (!PollCompletionQueue(cq_entries, source_addrs)) {
      finished_count++;
    };
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
    mq.second->SignalFinished(ID);
  }
  // Block main thread until all socket-threads finish their jobs
  poll_thread->join();
  // for (auto& thread : threads_) {
  //   thread.second->join();
  // }
  // // Clear all sockets
  // for (auto& socket : sockets_) {
  //   socket.second->Close();
  // }
  // server_socket_->Close();
  // delete server_socket_;
}
}  // namespace network
}  // namespace dgl
