/*!
 *  Copyright (c) 2020 by Contributors
 * \file rpc/rpc.cc
 * \brief Implementation of RPC utilities used by both server and client sides.
 */
#include "./rpc.h"

#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>
#include <dgl/zerocopy_serializer.h>
#include "../c_api_common.h"

using dgl::network::StringPrintf;
using namespace dgl::runtime;

namespace dgl {
namespace rpc {

RPCStatus SendRPCMessage(const RPCMessage& msg) {
  std::shared_ptr<std::string> zerocopy_blob(new std::string());
  StringStreamWithBuffer zc_write_strm(zerocopy_blob.get());
  static_cast<dmlc::Stream *>(&zc_write_strm)->Write(msg);
  network::Message rpc_meta_msg;
  rpc_meta_msg.data = zerocopy_blob->data();
  rpc_meta_msg.size = zerocopy_blob->size();
  rpc_meta_msg.deallocator = [zerocopy_blob](network::Message*) {};
  CHECK_EQ(RPCContext::ThreadLocal()->sender->Send(
    rpc_meta_msg, msg.server_id), ADD_SUCCESS);
  // send ndarray count
  char* ndarray_count = new char[sizeof(int32_t)];
  *(reinterpret_cast<int32_t*>(ndarray_count)) = msg.tensors.size();
  network::Message ndarray_count_msg;
  ndarray_count_msg.data = ndarray_count;
  ndarray_count_msg.size = sizeof(int32_t);
  ndarray_count_msg.deallocator = network::DefaultMessageDeleter;
  CHECK_EQ(RPCContext::ThreadLocal()->sender->Send(
    ndarray_count_msg, msg.server_id), ADD_SUCCESS);
  // send real ndarray data
  for (auto ptr : zc_write_strm.buffer_list()) {
    network::Message ndarray_data_msg;
    ndarray_data_msg.data = reinterpret_cast<char*>(ptr.data);
    ndarray_data_msg.size = ptr.size;
    NDArray tensor = ptr.tensor;
    ndarray_data_msg.deallocator = [tensor](network::Message*) {};
    CHECK_EQ(RPCContext::ThreadLocal()->sender->Send(
      ndarray_data_msg, msg.server_id), ADD_SUCCESS);
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
  // Copy the data for now, can be optimized later
  std::string zerocopy_blob(rpc_meta_msg.data, rpc_meta_msg.size);
  rpc_meta_msg.deallocator(&rpc_meta_msg);
  // Recv count
  network::Message ndarray_count_msg;
  CHECK_EQ(RPCContext::ThreadLocal()->receiver->RecvFrom(
    &ndarray_count_msg, send_id), REMOVE_SUCCESS);
  int32_t ndarray_count = *(reinterpret_cast<int32_t*>(ndarray_count_msg.data));
  ndarray_count_msg.deallocator(&ndarray_count_msg);
  // Recv real ndarray data
  std::vector<void* > buffer_list(ndarray_count);
  for (int i = 0; i < ndarray_count; ++i) {
    network::Message ndarray_data_msg;
    CHECK_EQ(RPCContext::ThreadLocal()->receiver->RecvFrom(
        &ndarray_data_msg, send_id), REMOVE_SUCCESS);
    buffer_list[i] = ndarray_data_msg.data;
  }
  StringStreamWithBuffer zc_read_strm(&zerocopy_blob, buffer_list);
  static_cast<dmlc::Stream *>(&zc_read_strm)->Read(msg);
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
  *rv = SendRPCMessage(*(msg.sptr()));
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

//////////////////////////// ServerState ////////////////////////////

DGL_REGISTER_GLOBAL("distributed.server_state._CAPI_DGLRPCGetServerState")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  auto st = RPCContext::ThreadLocal()->server_state;
  CHECK(st) << "Server state has not been initialized.";
  *rv = st;
});

}  // namespace rpc
}  // namespace dgl
