/*!
 *  Copyright (c) 2020 by Contributors
 * \file rpc/rpc.cc
 * \brief Implementation of RPC utilities used by both server and client sides.
 */
#include "./rpc.h"

#include <csignal>
#if defined(__linux__)
#include <unistd.h>
#endif

#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>
#include <dgl/array.h>
#include <dgl/random.h>
#include <dgl/zerocopy_serializer.h>
#include "../c_api_common.h"

using dgl::network::StringPrintf;
using namespace dgl::runtime;

namespace dgl {
namespace rpc {

RPCStatus SendRPCMessage(const RPCMessage& msg, const int32_t target_id) {
  std::shared_ptr<std::string> zerocopy_blob(new std::string());
  StreamWithBuffer zc_write_strm(zerocopy_blob.get(), true);
  zc_write_strm.Write(msg);
  int32_t ndarray_count = msg.tensors.size();
  zerocopy_blob->append(
    reinterpret_cast<char*>(&ndarray_count),
    sizeof(int32_t));
  network::Message rpc_meta_msg;
  rpc_meta_msg.data = const_cast<char*>(zerocopy_blob->data());
  rpc_meta_msg.size = zerocopy_blob->size();
  rpc_meta_msg.deallocator = [zerocopy_blob](network::Message*) {};
  CHECK_EQ(RPCContext::ThreadLocal()->sender->Send(
    rpc_meta_msg, target_id), ADD_SUCCESS);
  // send real ndarray data
  for (auto ptr : zc_write_strm.buffer_list()) {
    network::Message ndarray_data_msg;
    ndarray_data_msg.data = reinterpret_cast<char*>(ptr.data);
    if (ptr.size == 0) {
      LOG(FATAL) << "Cannot send a empty NDArray.";
    }
    ndarray_data_msg.size = ptr.size;
    NDArray tensor = ptr.tensor;
    ndarray_data_msg.deallocator = [tensor](network::Message*) {};
    CHECK_EQ(RPCContext::ThreadLocal()->sender->Send(
      ndarray_data_msg, target_id), ADD_SUCCESS);
  }
  return kRPCSuccess;
}

RPCStatus RecvRPCMessage(RPCMessage* msg, int32_t timeout) {
  // ignore timeout now
  CHECK_EQ(timeout, 0) << "rpc cannot support timeout now.";
  network::Message rpc_meta_msg;
  int send_id;
  CHECK_EQ(RPCContext::ThreadLocal()->receiver->Recv(
    &rpc_meta_msg, &send_id), REMOVE_SUCCESS);
  char* count_ptr = rpc_meta_msg.data+rpc_meta_msg.size-sizeof(int32_t);
  int32_t ndarray_count = *(reinterpret_cast<int32_t*>(count_ptr));
  // Recv real ndarray data
  std::vector<void* > buffer_list(ndarray_count);
  for (int i = 0; i < ndarray_count; ++i) {
    network::Message ndarray_data_msg;
    CHECK_EQ(RPCContext::ThreadLocal()->receiver->RecvFrom(
        &ndarray_data_msg, send_id), REMOVE_SUCCESS);
    buffer_list[i] = ndarray_data_msg.data;
  }
  StreamWithBuffer zc_read_strm(rpc_meta_msg.data, rpc_meta_msg.size-sizeof(int32_t), buffer_list);
  zc_read_strm.Read(msg);
  rpc_meta_msg.deallocator(&rpc_meta_msg);
  return kRPCSuccess;
}

//////////////////////////// C APIs ////////////////////////////

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCCreateSender")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  int64_t msg_queue_size = args[0];
  std::string type = args[1];
  if (type.compare("socket") == 0) {
    RPCContext::ThreadLocal()->sender = std::make_shared<network::SocketSender>(msg_queue_size);
  } else {
    LOG(FATAL) << "Unknown communicator type for rpc receiver: " << type;
  }
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCCreateReceiver")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  int64_t msg_queue_size = args[0];
  std::string type = args[1];
  if (type.compare("socket") == 0) {
    RPCContext::ThreadLocal()->receiver = std::make_shared<network::SocketReceiver>(msg_queue_size);
  } else {
    LOG(FATAL) << "Unknown communicator type for rpc sender: " << type;
  }
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCFinalizeSender")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  RPCContext::ThreadLocal()->sender->Finalize();
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCFinalizeReceiver")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  RPCContext::ThreadLocal()->receiver->Finalize();
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCReceiverWait")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  std::string ip = args[0];
  int port = args[1];
  int num_sender = args[2];
  std::string addr;
  if (RPCContext::ThreadLocal()->receiver->Type() == "socket") {
    addr = StringPrintf("socket://%s:%d", ip.c_str(), port);
  } else {
    LOG(FATAL) << "Unknown communicator type: " << RPCContext::ThreadLocal()->receiver->Type();
  }
  if (RPCContext::ThreadLocal()->receiver->Wait(addr.c_str(), num_sender) == false) {
    LOG(FATAL) << "Wait sender socket failed.";
  }
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCAddReceiver")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  std::string ip = args[0];
  int port = args[1];
  int recv_id = args[2];
  std::string addr;
  if (RPCContext::ThreadLocal()->sender->Type() == "socket") {
    addr = StringPrintf("socket://%s:%d", ip.c_str(), port);
  } else {
    LOG(FATAL) << "Unknown communicator type: " << RPCContext::ThreadLocal()->sender->Type();
  }
  RPCContext::ThreadLocal()->sender->AddReceiver(addr.c_str(), recv_id);
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSenderConnect")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  if (RPCContext::ThreadLocal()->sender->Connect() == false) {
    LOG(FATAL) << "Sender connection failed.";
  }
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSetRank")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const int32_t rank = args[0];
  RPCContext::ThreadLocal()->rank = rank;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetRank")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  *rv = RPCContext::ThreadLocal()->rank;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSetNumServer")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const int32_t num_servers = args[0];
  *rv = RPCContext::ThreadLocal()->num_servers = num_servers;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetNumServer")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  *rv = RPCContext::ThreadLocal()->num_servers;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSetNumServerPerMachine")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const int32_t num_servers = args[0];
  *rv = RPCContext::ThreadLocal()->num_servers_per_machine = num_servers;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetNumServerPerMachine")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  *rv = RPCContext::ThreadLocal()->num_servers_per_machine;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCIncrMsgSeq")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  *rv = (RPCContext::ThreadLocal()->msg_seq)++;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetMsgSeq")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  *rv = RPCContext::ThreadLocal()->msg_seq;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSetMsgSeq")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const int64_t msg_seq = args[0];
  RPCContext::ThreadLocal()->msg_seq = msg_seq;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetMachineID")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  *rv = RPCContext::ThreadLocal()->machine_id;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSetMachineID")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const int32_t machine_id = args[0];
  RPCContext::ThreadLocal()->machine_id = machine_id;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetNumMachines")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  *rv = RPCContext::ThreadLocal()->num_machines;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSetNumMachines")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const int32_t num_machines = args[0];
  RPCContext::ThreadLocal()->num_machines = num_machines;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSendRPCMessage")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  RPCMessageRef msg = args[0];
  const int32_t target_id = args[1];
  *rv = SendRPCMessage(*(msg.sptr()), target_id);
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCRecvRPCMessage")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  int32_t timeout = args[0];
  RPCMessageRef msg = args[1];
  *rv = RecvRPCMessage(msg.sptr().get(), timeout);
});

//////////////////////////// RPCMessage ////////////////////////////

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCCreateEmptyRPCMessage")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  std::shared_ptr<RPCMessage> rst(new RPCMessage);
  *rv = rst;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCCreateRPCMessage")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  std::shared_ptr<RPCMessage> rst(new RPCMessage);
  rst->service_id = args[0];
  rst->msg_seq = args[1];
  rst->client_id = args[2];
  rst->server_id = args[3];
  const std::string data = args[4];  // directly assigning string value raises errors :(
  rst->data = data;
  rst->tensors = ListValueToVector<NDArray>(args[5]);
  *rv = rst;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCMessageGetServiceId")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const RPCMessageRef msg = args[0];
  *rv = msg->service_id;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCMessageGetMsgSeq")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const RPCMessageRef msg = args[0];
  *rv = msg->msg_seq;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCMessageGetClientId")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const RPCMessageRef msg = args[0];
  *rv = msg->client_id;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCMessageGetServerId")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const RPCMessageRef msg = args[0];
  *rv = msg->server_id;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCMessageGetData")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const RPCMessageRef msg = args[0];
  DGLByteArray barr{msg->data.c_str(), msg->data.size()};
  *rv = barr;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCMessageGetTensors")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const RPCMessageRef msg = args[0];
  List<Value> ret;
  for (size_t i = 0; i < msg->tensors.size(); ++i) {
    ret.push_back(Value(MakeValue(msg->tensors[i])));
  }
  *rv = ret;
});

#if defined(__linux__)
/*!
 * \brief CtrlCHandler, exits if Ctrl+C is pressed
 * \param s signal
 */
void CtrlCHandler(int s) {
  LOG(INFO) << "\nUser pressed Ctrl+C, Exiting";
  exit(1);
}

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCHandleCtrlC")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  // Ctrl+C handler
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = CtrlCHandler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, nullptr);
});
#endif

//////////////////////////// ServerState ////////////////////////////

DGL_REGISTER_GLOBAL("distributed.server_state._CAPI_DGLRPCGetServerState")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  auto st = RPCContext::ThreadLocal()->server_state;
  if (st.get() == nullptr) {
    RPCContext::ThreadLocal()->server_state = std::make_shared<ServerState>();
  }
  *rv = st;
});

//////////////////////////// KVStore ////////////////////////////

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetGlobalIDFromLocalPartition")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
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
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  // Input
  std::string name = args[0];
  int local_machine_id = args[1];
  int machine_count = args[2];
  int group_count = args[3];
  int client_id = args[4];
  int service_id = args[5];
  int msg_seq = args[6];
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
  std::vector<std::vector<dgl_id_t> > remote_ids(machine_count);
  std::vector<std::vector<dgl_id_t> > remote_ids_original(machine_count);
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
    if (p_id == local_machine_id) {
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
  for (int i = 0; i < remote_ids.size(); ++i) {
    if (remote_ids[i].size() != 0) {
      RPCMessage msg;
      msg.service_id = service_id;
      msg.msg_seq = msg_seq;
      msg.client_id = client_id;
      int lower = i*group_count;
      int upper = (i+1)*group_count;
      msg.server_id = dgl::RandomEngine::ThreadLocal()->RandInt(lower, upper);
      msg.data = pickle_data;
      NDArray tensor = dgl::aten::VecToIdArray<dgl_id_t>(remote_ids[i]);
      msg.tensors.push_back(tensor);
      SendRPCMessage(msg, msg.server_id);
      msg_count++;
    }
  }
  local_data_shape[0] = ID_size;
  NDArray res_tensor = NDArray::Empty(local_data_shape,
                                      local_data->dtype,
                                      DLContext{kDLCPU, 0});
  char* return_data = static_cast<char*>(res_tensor->data);
  // Copy local data
#pragma omp parallel for
  for (int64_t i = 0; i < local_ids.size(); ++i) {
    CHECK_GE(ID_size*row_size, local_ids_orginal[i]*row_size+row_size);
    CHECK_GE(data_size, local_ids[i] * row_size + row_size);
    CHECK_GE(local_ids[i], 0);
    memcpy(return_data + local_ids_orginal[i] * row_size,
           local_data_char + local_ids[i] * row_size,
           row_size);
  }
  // Recv remote message
  for (int i = 0; i < msg_count; ++i) {
    RPCMessage msg;
    RecvRPCMessage(&msg, 0);
    int part_id = msg.server_id / group_count;
    char* data_char = static_cast<char*>(msg.tensors[0]->data);
    dgl_id_t id_size = remote_ids[part_id].size();
    for (size_t n = 0; n < id_size; ++n) {
      memcpy(return_data + remote_ids_original[part_id][n] * row_size,
             data_char + n * row_size,
             row_size);
    }
  }
  *rv = res_tensor;
});

}  // namespace rpc
}  // namespace dgl
