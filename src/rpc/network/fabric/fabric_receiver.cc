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
    msg_queue_[i] = std::make_shared<std::queue<Message>>();
  }
  
  for (size_t i = 0; i < peer_fi_addr.size(); i++) {
    // can be optimized with buffer pool
    // Will be freed in HandleCompletionEvent
    int64_t* size_buffer = new int64_t;
    // Issue recv events
    fep->Recv(size_buffer, sizeof(int64_t), kSizeMsg, FI_ADDR_UNSPEC, false,
              0xFFFF);
  }

  return true;
}

STATUS FabricReceiver::Recv(Message* msg, int* send_id) {
  while(true){
    PollCompletionQueue(cq_entries);
    for (auto& kv: msg_queue_){
      if (kv.second->size()>0){
          Message old_msg = kv.second->front();
          kv.second->pop();
          msg->data = old_msg.data;
          msg->size = old_msg.size;
          msg->deallocator = old_msg.deallocator;
          *send_id = kv.first;
          return REMOVE_SUCCESS;
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

bool FabricReceiver::PollCompletionQueue(
  struct fi_cq_tagged_entry* cq_entries) {
  // *source_addrs = 1;
  // fi_addr_t source_addrs2 = 999;
  int ret = fi_cq_read(fep->fabric_ctx->rxcq.get(), cq_entries,
                       kMaxConcurrentWorkRequest);
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
    for (int i = 0; i < ret; ++i) {
      if (!HandleCompletionEvent(cq_entries[i])) {
        keep_polling = false;
      };
    }
  }
  return keep_polling;
}

// bool FabricReceiver::Wait(const char* addr, int num_sender) {}

STATUS FabricReceiver::RecvFrom(Message* msg, int send_id) {
  // Get message from specified message queue
  auto queue = msg_queue_.at(send_id);
  while(true){
    PollCompletionQueue(cq_entries);
      if (queue->size()>0){
          Message old_msg = queue->front();
          msg->data = old_msg.data;
          msg->size = old_msg.size;
          msg->deallocator = old_msg.deallocator;
          queue->pop();
          return REMOVE_SUCCESS;
      }
  }
  // STATUS code = msg_queue_.at(send_id)->Remove(msg);
  // return code;
}

bool FabricReceiver::HandleCompletionEvent(
  const struct fi_cq_tagged_entry& cq_entry) {
    // LOG(INFO) << "Recv Msgs";
  uint64_t tag = cq_entry.tag & MsgTagMask;
  if (tag == kSizeMsg) {
    CHECK(cq_entry.len == sizeof(int64_t)) << "Invalid size message";
    int64_t data_size = *(int64_t*)cq_entry.buf;
    free(cq_entry.buf);  // Free size buffer
    char* buffer = nullptr;
    if (data_size == 0) {  // Indicate receiver should exit
      return false;
    }
    try {
      buffer = new char[data_size];
    } catch (const std::bad_alloc&) {
      LOG(FATAL) << "Cannot allocate enough memory for message, "
                 << "(message size: " << data_size << ")";
    }
    // Receive from specific sender
    fep->Recv(buffer, data_size, kDataMsg | (cq_entry.tag & IdMask),
              FI_ADDR_UNSPEC, false);
  } else if (tag == kDataMsg) {
    Message msg;
    msg.data = reinterpret_cast<char*>(cq_entry.buf);
    msg.size = cq_entry.len;
    msg.deallocator = DefaultMessageDeleter;
    int sender_id = cq_entry.tag & IdMask;
    msg_queue_.at(sender_id)->push(msg);
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

// void FabricReceiver::RecvLoop() {
//   for (size_t i = 0; i < peer_fi_addr.size(); i++) {
//     // can be optimized with buffer pool
//     // Will be freed in HandleCompletionEvent
//     int64_t* size_buffer = new int64_t;
//     // Issue recv events
//     fep->Recv(size_buffer, sizeof(int64_t), kSizeMsg, FI_ADDR_UNSPEC, false,
//               0xFFFF);
//   }

//   struct fi_cq_tagged_entry cq_entries[kMaxConcurrentWorkRequest];
//   int finished_count = 0;
//   while (finished_count != num_sender_) {
//     // If main thread had finished its job
//     for (auto& kv : msg_queue_) {
//       if (kv.second->EmptyAndNoMoreAdd()) {
//         return;  // exit loop thread
//       }
//     }
//     if (!PollCompletionQueue(cq_entries)) {
//       finished_count++;
//     };
//   }
// }

void FabricReceiver::Finalize() {
  // Send a signal to tell the message queue to finish its job
//   for (auto& mq : msg_queue_) {
//     // wait until queue is empty
//     while (mq.second->Empty() == false) {
// #ifdef _WIN32
//       // just loop
// #else   // !_WIN32
//       usleep(1000);
// #endif  // _WIN32
//     }
//     int ID = mq.first;
//     should_stop_polling_ = true;
//     mq.second->SignalFinished(ID);
//   }
//   // Block main thread until all socket-threads finish their jobs
//   poll_thread->join();
}
}  // namespace network
}  // namespace dgl
