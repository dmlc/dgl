#include <dgl/network/fabric/fabric_communicator.h>
#include <dgl/network/fabric/fabric_utils.h>
#include <netinet/in.h>
#include <unistd.h>

namespace dgl {
namespace network {

void FabricSender::AddReceiver(const char* addr, int recv_id) {
  struct sockaddr_in sin = {};
  FabricAddr fabric_addr;
  sin.sin_family = AF_INET;
  std::vector<std::string> substring, hostport;
  SplitStringUsing(addr, "//", &substring);
  SplitStringUsing(substring[1], ":", &hostport);
  // sin.sin_addr = hostport[0];
  CHECK(inet_pton(AF_INET, hostport[0].c_str(), &sin.sin_addr)>0) << "Invalid ip";
  sin.sin_port = htons(stoi(hostport[1]));
  
  fabric_addr.CopyFrom(&sin, sizeof(sin));
  peer_fi_addr[recv_id]=ctrl_ep->AddPeerAddr(&fabric_addr);
  // ctrl_ep->ctrl_peer_addr[recv_id] = address;
  msg_queue_[recv_id] = std::make_shared<MessageQueue>(queue_size_);
}

void FabricSender::Finalize() {
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
}

bool FabricSender::Connect() {
  Message msg;
  size_t num_peer = peer_fi_addr.size();
  for (int64_t i = 0; i < num_peer; i++) {
    FabricAddrInfo ctrl_info = {.id = i, .addr = ctrl_ep->fabric_ctx->addr};
    ctrl_ep->Send(&ctrl_info, sizeof(FabricAddrInfo), kAddrMsg, ctrl_peer_fi_addr[i],
                  true);

    FabricAddrInfo fep_info = {.id = i, .addr = fep->fabric_ctx->addr};
    ctrl_ep->Send(&fep_info, sizeof(FabricAddrInfo), kAddrMsg, ctrl_peer_fi_addr[i],
                  true);
  }
  struct FabricAddrInfo addrs[num_peer];
  for (size_t i = 0; i < num_peer; i++) {
    ctrl_ep->Recv(&addrs[i], sizeof(FabricAddrInfo), kAddrMsg, FI_ADDR_UNSPEC,
                  true);
    peer_fi_addr[addrs[i].id] = fep->AddPeerAddr(&addrs[i].addr);
  };

  for (auto kv : peer_fi_addr) {
    // LOG(INFO) << msg_queue_[kv.first].get();
    // LOG(INFO) << kv ) << kv.first;
            
    threads_[kv.first] = std::make_shared<std::thread>(
      &FabricSender::SendLoop, this, kv.second, msg_queue_[kv.first].get());
  }

  ctrl_ep.reset();
  return true;
}

STATUS FabricSender::Send(Message msg, int recv_id) {
  CHECK_NOTNULL(msg.data);
  CHECK_GT(msg.size, 0);
  CHECK_GE(recv_id, 0);
  // Add data message to message queue
  STATUS code = msg_queue_[recv_id]->Add(msg);
  return code;
}

void FabricSender::SendLoop(fi_addr_t peer_addr, MessageQueue* queue) {
  CHECK_NOTNULL(queue);
  bool exit = false;
  {LOG(INFO) << "Start sendloop";}
  while (!exit) {
    Message* msg_ptr = new Message();
    STATUS code = queue->Remove(msg_ptr);
    // TODO(AZ): How to finalize queue
    if (code == QUEUE_CLOSE) {
      // msg->size = 0;  // send an end-signal to receiver
      exit = true;
    }
    fep->Send(&msg_ptr->size, sizeof(msg_ptr->size), kSizeMsg, peer_addr);
    // LOG(INFO) << "send data";
    fep->Send(msg_ptr->data, msg_ptr->size, kDataMsg, peer_addr, false,
              msg_ptr);
    // LOG(INFO) << "send data done";
  }
}

void FabricSender::PollCompletionQueue(struct fi_cq_tagged_entry* cq_entries) {
  int ret = fi_cq_read(fep->fabric_ctx->txcq.get(), cq_entries,
                       kMaxConcurrentWorkRequest);
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
      HandleCompletionEvent(cq_entry);
    }
  }
};

void FabricSender::HandleCompletionEvent(
  const struct fi_cq_tagged_entry& cq_entry) {
  if ((cq_entry.op_context) and (cq_entry.tag == kDataMsg)) {
    Message* msg_ptr = reinterpret_cast<Message*>(cq_entry.op_context);
    if (msg_ptr->deallocator != nullptr) {
      msg_ptr->deallocator(msg_ptr);
    }
  }
};

}  // namespace network
}  // namespace dgl