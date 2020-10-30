#include <dgl/network/fabric/fabric_communicator.h>
#include <dgl/network/fabric/fabric_utils.h>
#include <unistd.h>

namespace dgl {
namespace network {

void FabricSender::AddReceiver(const char* addr, int recv_id) {
  socket_sender->AddReceiver(addr, recv_id);
  LOG(INFO) << "ADDRECEIVER:" << recv_id << ":" << addr;
  msg_queue_[recv_id] =  std::make_shared<MessageQueue>(queue_size_);
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
  // Clear all sockets
  for (auto& socket : sockets_) {
    socket.second->Close();
  }
}
bool FabricSender::Connect() {
  { LOG(INFO) << "Before connect"; }
  CHECK(socket_sender->Connect()) << "Fail to connect";
  { LOG(INFO) << "After connect"; }
  Message msg;
  for (int64_t i = 0; i < socket_sender->NumServer(); i++) {
    FabricAddrInfo info = {.id = i, .addr = fep->fabric_ctx->addr};
    msg.size = sizeof(info);
    msg.data = reinterpret_cast<char*>(&info);
    socket_sender->Send(msg, i);
  }
  struct FabricAddrInfo addrs[socket_sender->NumServer()];
  for (size_t i = 0; i < socket_sender->NumServer(); i++) {
    fep->Recv(&addrs[i], sizeof(FabricAddrInfo), kAddrMsg, FI_ADDR_UNSPEC,
              true);
  };
  for (size_t i = 0; i < socket_sender->NumServer(); i++) {
    LOG(INFO) << "ID" << addrs[i].id;
    LOG(INFO) << "ADDR ID" << addrs[i].addr.name;
    LOG(INFO) << "ADDR ID" << addrs[i].addr.len;
    addr_map[addrs[i].id] = fep->AddPeerAddr(&addrs[i].addr);
  }
  

  for (auto kv : addr_map) {
    LOG(INFO) << msg_queue_[kv.first].get();
    LOG(INFO) << kv.second;
    LOG(INFO) << kv.first;
    threads_[kv.first] = std::make_shared<std::thread>(
      &FabricSender::SendLoop, this, kv.second, msg_queue_[kv.first].get());
  }
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
  //   CHECK_NOTNULL(socket);
  // CHECK_NOTNULL(queue);
  bool exit = false;
  while (!exit) {
    Message msg;
    STATUS code = queue->Remove(&msg);
    if (code == QUEUE_CLOSE) {
      msg.size = 0;  // send an end-signal to receiver
      exit = true;
    }
    // LOG(INFO) << "send size";
    // LOG(INFO) << "send peer" << peer_addr;
    fep->Send(&msg.size, sizeof(msg.size), kSizeMsg, peer_addr);
    // LOG(INFO) << "send data";
    fep->Send(msg.data, msg.size, kDataMsg, peer_addr);
    // LOG(INFO) << "send data done";
  }
}
}  // namespace network
}  // namespace dgl