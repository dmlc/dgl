#include <chrono>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <cstdlib>

#include "rpc_base.h"

class RPCClient {
public:
  explicit RPCClient(const std::string &ip_config) : ip_config_(ip_config) {
    parse_ips();
  }

  void run() {
    std::vector<std::thread> threads;
    for (int i = 0; i < kNumSender; ++i) {
      threads.push_back(std::thread(&RPCClient::start_client, this));
    }
    for (auto &&t : threads) {
      t.join();
    }
  }

private:
  void parse_ips() {
    std::ifstream ifs(ip_config_);
    if (!ifs) {
      LOG(FATAL) << "Failed to open ip_config: " + ip_config_;
    }
    for (std::string line; std::getline(ifs, line);) {
      std::cout << line << std::endl;
      ips_.push_back(line);
    }
  }
  void start_client() {
    dgl::rpc::TPSender sender(InitTPContext());
    int recv_id = 0;
    for (const auto &ip : ips_) {
      for (int i = 0; i < kNumReceiver; ++i) {
        const std::string ip_addr =
            std::string{"tcp://"} + ip + ":" + std::to_string(kPort + i);
        while (!sender.ConnectReceiver(ip_addr, recv_id)) {
          std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        ++recv_id;
      }
    }
    const int num_machines = ips_.size();
    for (int i = 0; i < kNumMessage; ++i) {
      for (int n = 0; n < recv_id; ++n) {
        dgl::rpc::RPCMessage msg;
        msg.data = "123456789";
        const auto tensor =
            dgl::runtime::NDArray::FromVector(std::vector<int>(kSizeTensor, 1));
        for (int j = 0; j < kNumTensor; ++j) {
          msg.tensors.push_back(tensor);
        }
        sender.Send(msg, n);
      }
    }
    sender.Finalize();
  }
  const std::string ip_config_;
  std::vector<std::string> ips_;
};

int main(int argc, char **argv) {
  if (argc != 2) {
    LOG(FATAL)
        << "Invalid call. Please call like this: ./rpc_client ip_config.txt";
    return -1;
  }
  RPCClient client(argv[1]);
  client.run();

  return 0;
}
