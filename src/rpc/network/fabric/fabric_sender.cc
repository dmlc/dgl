#include <dgl/network/fabric/fabric_communicator.h>
#include <dgl/network/fabric/fabric_utils.h>
#include <netinet/in.h>
#include <unistd.h>

namespace dgl {
namespace network {

void FabricSender::AddReceiver(const char* addr, int recv_id) {
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
}

void FabricSender::Finalize() {
  // Send a signal to tell the msg_queue to finish its job
//   for (auto& mq : msg_queue_) {
//     // wait until queue is empty
//     while (mq.second->Empty() == false) {
// #ifdef _WIN32
//       // just loop
//       //
// #else  // !_WIN32
//       usleep(1000);
// #endif  // _WIN32
//     }
  //   int ID = mq.first;
  //   mq.second->SignalFinished(ID);
  // }
  // poll_thread->join();
}

bool FabricSender::Connect() {
  sender_id = -1;
  struct FabricAddrInfo addrs;
  for (auto& kv : ctrl_peer_fi_addr) {
    // Send local control info to remote
    FabricAddrInfo ctrl_info = {.sender_id = -1,
                                .receiver_id = kv.first,
                                .addr = ctrl_ep->fabric_ctx->addr};
    ctrl_ep->Send(&ctrl_info, sizeof(FabricAddrInfo), kCtrlAddrMsg, kv.second,
                  true);
    FabricAddrInfo fep_info = {
      .sender_id = -1, .receiver_id = kv.first, .addr = fep->fabric_ctx->addr};
    ctrl_ep->Send(&fep_info, sizeof(FabricAddrInfo), kFiAddrMsg, kv.second,
                  true);
    ctrl_ep->Recv(&addrs, sizeof(FabricAddrInfo), kFiAddrMsg, FI_ADDR_UNSPEC,
                  true);
    peer_fi_addr[kv.first] = fep->AddPeerAddr(&addrs.addr);
    sender_id = addrs.sender_id;
    // msg_queue_[kv.first] = std::make_shared<MessageQueue>(queue_size_);
  }
  // sleep(5);

  // poll_thread = std::make_shared<std::thread>(&FabricSender::SendLoop, this);
  return true;
}

STATUS FabricSender::Send(Message msg, int recv_id) {
  CHECK_NOTNULL(msg.data);
  CHECK_GT(msg.size, 0);
  CHECK_GE(recv_id, 0);
  // Add data message to message queue
  // STATUS code = msg_queue_[recv_id]->Add(msg);
  Message* msg2 = new Message();
  *msg2 = msg;
  fep->Send(&msg.size, sizeof(msg.size), kSizeMsg | sender_id,
            peer_fi_addr.at(recv_id));
  fep->Send(msg.data, msg.size, kDataMsg | sender_id,
            peer_fi_addr.at(recv_id), false, msg2);
  int count = 0;
  while (count<2){
    ssize_t ret = PollCompletionQueue(cq_entries);
    if (ret>0) count+=ret;
  }
  return ADD_SUCCESS;
}

// void FabricSender::SendLoop() {
//   bool exit = false;
//   struct fi_cq_tagged_entry cq_entries[kMaxConcurrentWorkRequest];
//   while (!exit) {
//     for (auto& kv : msg_queue_) {
//       Message* msg_ptr = new Message();
//       STATUS code = kv.second->Remove(msg_ptr, false);
//       if (code == QUEUE_EMPTY) {
//         continue;
//       } else if (code == REMOVE_SUCCESS) {
//         fep->Send(&msg_ptr->size, sizeof(msg_ptr->size), kSizeMsg | sender_id,
//                   peer_fi_addr[kv.first]);
//         fep->Send(msg_ptr->data, msg_ptr->size, kDataMsg | sender_id,
//                   peer_fi_addr[kv.first], false, msg_ptr);
//       } else if (code == QUEUE_CLOSE) {
//         int64_t* exit_code = new int64_t(0);
//         fep->Send(exit_code, sizeof(msg_ptr->size), kSizeMsg | sender_id,
//                   peer_fi_addr[kv.first]);
//         delete msg_ptr;
//         exit = true;
//       }
//     }
//     PollCompletionQueue(cq_entries);
//   }
// }

ssize_t FabricSender::PollCompletionQueue(struct fi_cq_tagged_entry* cq_entries) {
  ssize_t ret = fi_cq_read(fep->fabric_ctx->txcq.get(), cq_entries,
                       kMaxConcurrentWorkRequest);
  if (ret == -FI_EAGAIN) {
    return -FI_EAGAIN;
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
  return ret;
};

void FabricSender::HandleCompletionEvent(
  const struct fi_cq_tagged_entry& cq_entry) {
  if ((cq_entry.op_context) and ((cq_entry.tag & IdMask) == kDataMsg)) {
    Message* msg_ptr = reinterpret_cast<Message*>(cq_entry.op_context);
    if (msg_ptr->deallocator != nullptr) {
      msg_ptr->deallocator(msg_ptr);
    }
    delete msg_ptr;
  }
};

}  // namespace network
}  // namespace dgl