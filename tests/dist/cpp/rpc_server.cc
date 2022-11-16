#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

#include "rpc_base.h"

class RPCServer {
 public:
  explicit RPCServer(const std::string &ip, int num_machines)
      : ip_(ip), num_machines_(num_machines) {}
  void run() {
    std::vector<std::thread> threads;
    for (int i = 0; i < kNumReceiver; ++i) {
      threads.push_back(std::thread(&RPCServer::StartServer, this, i));
    }
    for (auto &&t : threads) {
      t.join();
    }
  }

 private:
  void StartServer(int id) {
    dgl::rpc::TPReceiver receiver(InitTPContext());
    std::string ip_addr =
        std::string{"tcp://"} + ip_ + ":" + std::to_string(kPort + id);
    if (!receiver.Wait(ip_addr, kNumSender * num_machines_, false)) {
      LOG(FATAL) << "Failed to wait on addr: " << ip_addr;
    }
    for (int n = 0; n < kNumSender * kNumMessage * num_machines_; ++n) {
      dgl::rpc::RPCMessage msg;
      if (receiver.Recv(&msg, 0) != dgl::rpc::kRPCSuccess) {
        LOG(FATAL) << "Failed to receive message on Server~" << id;
      }
      bool eq = msg.data == std::string("123456789");
      eq = eq && (msg.tensors.size() == kNumTensor);
      for (int j = 0; eq && j < kNumTensor; ++j) {
        eq = eq && (msg.tensors[j].ToVector<int>().size() == kSizeTensor);
      }
      if (!eq) {
        LOG(FATAL) << "Invalid received message";
      }
      if ((n + 1) % 1000 == 0) {
        LOG(INFO) << n + 1 << " RPCMessages have been received/verified on "
                  << ip_addr;
      }
    }
    receiver.Finalize();
  }
  const std::string ip_;
  const int num_machines_;
};

int main(int argc, char **argv) {
  if (argc != 3) {
    LOG(FATAL)
        << "Invalid call. Please call like this: ./rpc_server 4 127.0.0.1";
    return -1;
  }
  const int num_machines = std::atoi(argv[1]);
  const std::string ip{argv[2]};
  RPCServer server(ip, num_machines);
  server.run();

  return 0;
}
