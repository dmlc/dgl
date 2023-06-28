/**
 *  Copyright (c) 2020 by Contributors
 * @file rpc/rpc.cc
 * @brief Implementation of RPC utilities used by both server and client sides.
 */
#if defined(__linux__)
#include "./rpc.h"

#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <dgl/random.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/parallel_for.h>
#include <dgl/zerocopy_serializer.h>
#include <unistd.h>

#include <csignal>
#include <future>

#include "../c_api_common.h"
#include "../runtime/resource_manager.h"

using dgl::network::StringPrintf;
using namespace dgl::runtime;

namespace dgl {
namespace rpc {

// Borrow from PyTorch

const char kSocketIfnameEnvVar[] = "TP_SOCKET_IFNAME";
const char kDefaultUvAddress[] = "127.0.0.1";

RPCStatus SendRPCMessage(const RPCMessage& msg, const int32_t target_id) {
  RPCContext::getInstance()->sender->Send(msg, target_id);
  return kRPCSuccess;
}

RPCStatus RecvRPCMessage(RPCMessage* msg, int32_t timeout) {
  static constexpr int32_t retry_timeout = 5 * 1000;  // milliseconds
  RPCStatus status;
  const int32_t real_timeout = timeout == 0 ? retry_timeout : timeout;
  do {
    status = RPCContext::getInstance()->receiver->Recv(msg, real_timeout);
    if (status == kRPCTimeOut) {
      static const std::string log_str = [real_timeout, timeout]() {
        std::ostringstream oss;
        oss << "Recv RPCMessage timeout in " << real_timeout << " ms."
            << (timeout == 0 ? " Retrying ..." : "");
        return oss.str();
      }();
      DLOG(WARNING) << log_str;
    }
  } while (timeout == 0 && status == kRPCTimeOut);
  return status;
}

//////////////////////////// C APIs ////////////////////////////
DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCReset")
    .set_body([](DGLArgs args, DGLRetValue* rv) { RPCContext::Reset(); });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCCreateSender")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      int64_t msg_queue_size = args[0];
      int max_thread_count = args[1];
      RPCContext::getInstance()->sender.reset(
          new network::SocketSender(msg_queue_size, max_thread_count));
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCCreateReceiver")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      int64_t msg_queue_size = args[0];
      int max_thread_count = args[1];
      RPCContext::getInstance()->receiver.reset(
          new network::SocketReceiver(msg_queue_size, max_thread_count));
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCFinalizeSender")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      RPCContext::getInstance()->sender->Finalize();
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCFinalizeReceiver")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      RPCContext::getInstance()->receiver->Finalize();
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCWaitForSenders")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      std::string ip = args[0];
      int port = args[1];
      int num_sender = args[2];
      std::string addr;
      addr = StringPrintf("tcp://%s:%d", ip.c_str(), port);
      if (RPCContext::getInstance()->receiver->Wait(addr, num_sender) ==
          false) {
        LOG(FATAL) << "Wait sender socket failed.";
      }
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCConnectReceiver")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      std::string ip = args[0];
      int port = args[1];
      int recv_id = args[2];
      std::string addr;
      addr = StringPrintf("tcp://%s:%d", ip.c_str(), port);
      *rv = RPCContext::getInstance()->sender->ConnectReceiver(addr, recv_id);
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCConnectReceiverFinalize")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int max_try_times = args[0];
      *rv = RPCContext::getInstance()->sender->ConnectReceiverFinalize(
          max_try_times);
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSetRank")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int32_t rank = args[0];
      RPCContext::getInstance()->rank = rank;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetRank")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      *rv = RPCContext::getInstance()->rank;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSetNumServer")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int32_t num_servers = args[0];
      *rv = RPCContext::getInstance()->num_servers = num_servers;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetNumServer")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      *rv = RPCContext::getInstance()->num_servers;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSetNumClient")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int32_t num_clients = args[0];
      *rv = RPCContext::getInstance()->num_clients = num_clients;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetNumClient")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      *rv = RPCContext::getInstance()->num_clients;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSetNumServerPerMachine")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int32_t num_servers = args[0];
      *rv = RPCContext::getInstance()->num_servers_per_machine = num_servers;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetNumServerPerMachine")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      *rv = RPCContext::getInstance()->num_servers_per_machine;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCIncrMsgSeq")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      *rv = (RPCContext::getInstance()->msg_seq)++;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetMsgSeq")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      *rv = RPCContext::getInstance()->msg_seq;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSetMsgSeq")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int64_t msg_seq = args[0];
      RPCContext::getInstance()->msg_seq = msg_seq;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetBarrierCount")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int32_t group_id = args[0];
      auto&& cnt = RPCContext::getInstance()->barrier_count;
      if (cnt.find(group_id) == cnt.end()) {
        cnt.emplace(group_id, 0x0);
      }
      *rv = cnt[group_id];
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSetBarrierCount")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int32_t count = args[0];
      const int32_t group_id = args[1];
      RPCContext::getInstance()->barrier_count[group_id] = count;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetMachineID")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      *rv = RPCContext::getInstance()->machine_id;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSetMachineID")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int32_t machine_id = args[0];
      RPCContext::getInstance()->machine_id = machine_id;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetNumMachines")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      *rv = RPCContext::getInstance()->num_machines;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSetNumMachines")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int32_t num_machines = args[0];
      RPCContext::getInstance()->num_machines = num_machines;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSendRPCMessage")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      RPCMessageRef msg = args[0];
      const int32_t target_id = args[1];
      *rv = SendRPCMessage(*(msg.sptr()), target_id);
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCRecvRPCMessage")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      int32_t timeout = args[0];
      RPCMessageRef msg = args[1];
      *rv = RecvRPCMessage(msg.sptr().get(), timeout);
    });

//////////////////////////// RPCMessage ////////////////////////////

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCCreateEmptyRPCMessage")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      std::shared_ptr<RPCMessage> rst(new RPCMessage);
      *rv = rst;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCCreateRPCMessage")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      std::shared_ptr<RPCMessage> rst(new RPCMessage);
      rst->service_id = args[0];
      rst->msg_seq = args[1];
      rst->client_id = args[2];
      rst->server_id = args[3];
      const std::string data =
          args[4];  // directly assigning string value raises errors :(
      rst->data = data;
      rst->tensors = ListValueToVector<NDArray>(args[5]);
      rst->group_id = args[6];
      *rv = rst;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCMessageGetServiceId")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const RPCMessageRef msg = args[0];
      *rv = msg->service_id;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCMessageGetMsgSeq")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const RPCMessageRef msg = args[0];
      *rv = msg->msg_seq;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCMessageGetClientId")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const RPCMessageRef msg = args[0];
      *rv = msg->client_id;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCMessageGetServerId")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const RPCMessageRef msg = args[0];
      *rv = msg->server_id;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCMessageGetData")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const RPCMessageRef msg = args[0];
      DGLByteArray barr{msg->data.c_str(), msg->data.size()};
      *rv = barr;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCMessageGetTensors")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const RPCMessageRef msg = args[0];
      List<Value> ret;
      for (size_t i = 0; i < msg->tensors.size(); ++i) {
        ret.push_back(Value(MakeValue(msg->tensors[i])));
      }
      *rv = ret;
    });

#if defined(__linux__)
/**
 * @brief The signal handler.
 * @param s signal
 */
void SigHandler(int s) {
  LOG(INFO) << "\nUser pressed Ctrl+C, Exiting";
  CleanupResources();
  exit(1);
}

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCHandleSignal")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      // Ctrl+C handler
      struct sigaction sigHandler;
      sigHandler.sa_handler = SigHandler;
      sigemptyset(&sigHandler.sa_mask);
      sigHandler.sa_flags = 0;
      sigaction(SIGINT, &sigHandler, nullptr);
      sigaction(SIGTERM, &sigHandler, nullptr);
    });
#endif

//////////////////////////// ServerState ////////////////////////////

DGL_REGISTER_GLOBAL("distributed.server_state._CAPI_DGLRPCGetServerState")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      auto st = RPCContext::getInstance()->server_state;
      if (st.get() == nullptr) {
        RPCContext::getInstance()->server_state =
            std::make_shared<ServerState>();
      }
      *rv = st;
    });

//////////////////////////// KVStore ////////////////////////////

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetGlobalIDFromLocalPartition")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      NDArray ID = args[0];
      NDArray part_id = args[1];
      int local_machine_id = args[2];
      int64_t* ID_data = static_cast<int64_t*>(ID->data);
      int64_t* part_id_data = static_cast<int64_t*>(part_id->data);
      int64_t ID_size = ID.GetSize() / sizeof(int64_t);
      std::vector<int64_t> global_id;
      for (int64_t i = 0; i < ID_size; ++i) {
        if (part_id_data[i] == local_machine_id) {
          global_id.push_back(ID_data[i]);
        }
      }
      NDArray res_tensor = dgl::aten::VecToIdArray<int64_t>(global_id);
      *rv = res_tensor;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCFastPull")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      // Input
      std::string name = args[0];
      int local_machine_id = args[1];
      int machine_count = args[2];
      int group_count = args[3];
      int client_id = args[4];
      int service_id = args[5];
      int64_t msg_seq = args[6];
      std::string pickle_data = args[7];
      NDArray ID = args[8];
      NDArray part_id = args[9];
      NDArray local_id = args[10];
      NDArray local_data = args[11];
      // Data
      dgl_id_t ID_size = ID.GetSize() / sizeof(dgl_id_t);
      dgl_id_t* ID_data = static_cast<dgl_id_t*>(ID->data);
      dgl_id_t* part_id_data = static_cast<dgl_id_t*>(part_id->data);
      dgl_id_t* local_id_data = static_cast<dgl_id_t*>(local_id->data);
      char* local_data_char = static_cast<char*>(local_data->data);
      std::vector<dgl_id_t> local_ids;
      std::vector<dgl_id_t> local_ids_orginal;
      std::vector<int64_t> local_data_shape;
      std::vector<std::vector<dgl_id_t>> remote_ids(machine_count);
      std::vector<std::vector<dgl_id_t>> remote_ids_original(machine_count);
      // Get row size (in bytes)
      int row_size = 1;
      for (int i = 0; i < local_data->ndim; ++i) {
        local_data_shape.push_back(local_data->shape[i]);
        if (i != 0) {
          row_size *= local_data->shape[i];
        }
      }
      row_size *= (local_data->dtype.bits / 8);
      size_t data_size = local_data.GetSize();
      CHECK_GT(local_data_shape.size(), 0);
      CHECK_EQ(row_size * local_data_shape[0], data_size);
      // Get local id (used in local machine) and
      // remote id (send to remote machine)
      dgl_id_t idx = 0;
      for (dgl_id_t i = 0; i < ID_size; ++i) {
        dgl_id_t p_id = part_id_data[i];
        if (static_cast<int>(p_id) == local_machine_id) {
          dgl_id_t l_id = local_id_data[idx++];
          CHECK_LT(l_id, local_data_shape[0]);
          CHECK_GE(l_id, 0);
          local_ids.push_back(l_id);
          local_ids_orginal.push_back(i);
        } else {
          CHECK_LT(p_id, machine_count) << "Invalid partition ID.";
          dgl_id_t id = ID_data[i];
          remote_ids[p_id].push_back(id);
          remote_ids_original[p_id].push_back(i);
        }
      }
      // Send remote id
      int msg_count = 0;
      for (size_t i = 0; i < remote_ids.size(); ++i) {
        if (remote_ids[i].size() != 0) {
          RPCMessage msg;
          msg.service_id = service_id;
          msg.msg_seq = msg_seq;
          msg.client_id = client_id;
          int lower = i * group_count;
          int upper = (i + 1) * group_count;
          msg.server_id =
              dgl::RandomEngine::ThreadLocal()->RandInt(lower, upper);
          msg.data = pickle_data;
          NDArray tensor = dgl::aten::VecToIdArray<dgl_id_t>(remote_ids[i]);
          msg.tensors.push_back(tensor);
          msg.group_id = RPCContext::getInstance()->group_id;
          SendRPCMessage(msg, msg.server_id);
          msg_count++;
        }
      }
      local_data_shape[0] = ID_size;
      NDArray res_tensor = NDArray::Empty(
          local_data_shape, local_data->dtype, DGLContext{kDGLCPU, 0});
      char* return_data = static_cast<char*>(res_tensor->data);
      // Copy local data
      parallel_for(0, local_ids.size(), [&](size_t b, size_t e) {
        for (auto i = b; i < e; ++i) {
          CHECK_GE(
              ID_size * row_size, local_ids_orginal[i] * row_size + row_size);
          CHECK_GE(data_size, local_ids[i] * row_size + row_size);
          CHECK_GE(local_ids[i], 0);
          memcpy(
              return_data + local_ids_orginal[i] * row_size,
              local_data_char + local_ids[i] * row_size, row_size);
        }
      });
      // Recv remote message
      int recv_cnt = 0;
      while (recv_cnt < msg_count) {
        RPCMessage msg;
        auto status = RecvRPCMessage(&msg, 0);
        CHECK_EQ(status, kRPCSuccess);
        ++recv_cnt;
        int part_id = msg.server_id / group_count;
        char* data_char = static_cast<char*>(msg.tensors[0]->data);
        dgl_id_t id_size = remote_ids[part_id].size();
        for (size_t n = 0; n < id_size; ++n) {
          memcpy(
              return_data + remote_ids_original[part_id][n] * row_size,
              data_char + n * row_size, row_size);
        }
      }
      *rv = res_tensor;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetGroupID")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      *rv = RPCContext::getInstance()->group_id;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSetGroupID")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int32_t group_id = args[0];
      RPCContext::getInstance()->group_id = group_id;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCMessageGetGroupId")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const RPCMessageRef msg = args[0];
      *rv = msg->group_id;
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCRegisterClient")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int32_t client_id = args[0];
      const int32_t group_id = args[1];
      *rv = RPCContext::getInstance()->RegisterClient(client_id, group_id);
    });

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetClient")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int32_t client_id = args[0];
      const int32_t group_id = args[1];
      *rv = RPCContext::getInstance()->GetClient(client_id, group_id);
    });

}  // namespace rpc
}  // namespace dgl

#endif
