#include <dgl/network/fabric/fabric_communicator.h>
#include <dgl/network/fabric/fabric_utils.h>
#include <netinet/in.h>
#include <unistd.h>

namespace dgl {
namespace network {

void FabricSender::AddReceiver(const char* addr, int recv_id) {
  LOG(INFO) << "Add Receiver: " << addr;
  struct sockaddr_in sin = {};
  FabricAddr ctrl_fi_addr;
  sin.sin_family = AF_INET;
  std::vector<std::string> substring, hostport;
  SplitStringUsing(addr, "//", &substring);
  SplitStringUsing(substring[1], ":", &hostport);
  CHECK(inet_pton(AF_INET, hostport[0].c_str(), &sin.sin_addr) > 0)
    << "Invalid ip";
  sin.sin_port = htons(stoi(hostport[1]));

  ctrl_fi_addr.CopyFrom(&sin, sizeof(sin));
  ctrl_peer_fi_addr[recv_id] = ctrl_ep->AddPeerAddr(&ctrl_fi_addr);
  // ctrl_ep->ctrl_peer_addr[recv_id] = address;
}

void FabricSender::Finalize() {
  // Send a signal to tell the msg_queue to finish its job
  for (auto& mq : msg_queue_) {
    // wait until queue is empty
    while (mq.second->Empty() == false) {
#ifdef _WIN32
      // just loop
      //
#else  // !_WIN32
      usleep(1000);
#endif  // _WIN32
    }
    int ID = mq.first;
    mq.second->SignalFinished(ID);
  }
  poll_thread->join();
  // Block main thread until all socket-threads finish their jobs
  // for (auto& thread : threads_) {
  //   thread.second->join();
  // }
}

bool FabricSender::Connect() {
  size_t num_peer = ctrl_peer_fi_addr.size();

  sender_id = -1;
  struct FabricAddrInfo addrs;
  for (auto& kv : ctrl_peer_fi_addr) {
    // Send local control info to remote
    FabricAddrInfo ctrl_info = {.sender_id = -1,
                                .receiver_id = kv.first,
                                .addr = ctrl_ep->fabric_ctx->addr};
    ctrl_ep->Send(&ctrl_info, sizeof(FabricAddrInfo), kAddrMsg, kv.second,
                  true);
    FabricAddrInfo fep_info = {
      .sender_id = -1, .receiver_id = kv.first, .addr = fep->fabric_ctx->addr};
    ctrl_ep->Send(&fep_info, sizeof(FabricAddrInfo), kAddrMsg, kv.second, true);
    ctrl_ep->Recv(&addrs, sizeof(FabricAddrInfo), kAddrMsg, FI_ADDR_UNSPEC,
                  true);
    // CHECK(addrs.receiver_id)
    peer_fi_addr[kv.first] = fep->AddPeerAddr(&addrs.addr);
    sender_id = addrs.sender_id;
    msg_queue_[kv.first] = std::make_shared<MessageQueue>(queue_size_);

    fep->Recv(handshake_buffer, handshake_msg.length(), kIgnoreMsg,
              FI_ADDR_UNSPEC);
    fep->Send(handshake_msg.c_str(), handshake_msg.length(), kIgnoreMsg,
              peer_fi_addr[kv.first]);
  }
  LOG(INFO) << "Finish Initialization";

  poll_thread = std::make_shared<std::thread>(&FabricSender::SendLoop, this);
  // ctrl_ep.reset();
  return true;
}

STATUS FabricSender::Send(Message msg, int recv_id) {
  CHECK_NOTNULL(msg.data);
  CHECK_GT(msg.size, 0);
  CHECK_GE(recv_id, 0);
  // Add data message to message queue
  // LOG(INFO) << "Queue addr:" << msg_queue_[recv_id].get();
  STATUS code = msg_queue_[recv_id]->Add(msg);
  return code;
}

void FabricSender::SendLoop() {
  // sleep(10);
  // { LOG(INFO) << "Start sendloop"; }
  bool exit = false;
  struct fi_cq_tagged_entry cq_entries[kMaxConcurrentWorkRequest];
  while (!exit) {
    for (auto& kv : msg_queue_) {
      Message* msg_ptr = new Message();
      STATUS code = kv.second->Remove(msg_ptr);
      // TODO(AZ): How to finalize queue
      if (code == QUEUE_EMPTY) {
        continue;
      } else if (code == QUEUE_CLOSE) {
        int64_t* exit_code = new int64_t(0);
        fep->Send(exit_code, sizeof(msg_ptr->size), kSizeMsg | sender_id,
                  peer_fi_addr[kv.first]);
        // LOG(INFO) << "Send exit code to:" << kv.first;
        delete msg_ptr;
        exit = true;
      } else if (code == REMOVE_SUCCESS) {
        // LOG(INFO) << "Send msgs size:" << msg_ptr->size;
        fep->Send(&msg_ptr->size, sizeof(msg_ptr->size), kSizeMsg | sender_id,
                  peer_fi_addr[kv.first]);
        fep->Send(msg_ptr->data, msg_ptr->size, kDataMsg | sender_id,
                  peer_fi_addr[kv.first], false, msg_ptr);
      }
    }
    PollCompletionQueue(cq_entries);
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
  if ((cq_entry.op_context) and ((cq_entry.tag & IdMask) == kDataMsg)) {
    Message* msg_ptr = reinterpret_cast<Message*>(cq_entry.op_context);
    if (msg_ptr->deallocator != nullptr) {
      msg_ptr->deallocator(msg_ptr);
    }
  }
};

}  // namespace network
}  // namespace dgl