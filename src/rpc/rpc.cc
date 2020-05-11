/*!
 *  Copyright (c) 2020 by Contributors
 * \file rpc/rpc.cc
 * \brief Implementation of RPC utilities used by both server and client sides.
 */
#include "./rpc.h"

#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>
#include "../c_api_common.h"

using namespace dgl::runtime;

namespace dgl {
namespace rpc {

RPCStatus SendRPCMessage(const RPCMessage& msg) {
  // TODO
  return kRPCSuccess;
}

RPCStatus RecvRPCMessage(RPCMessage* msg, int32_t timeout) {
  // TODO
  return kRPCSuccess;
}

//////////////////////////// C APIs ////////////////////////////

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCSetRank")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const int32_t rank = args[0];
  RPCContext::ThreadLocal()->rank = rank;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetRank")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  *rv = RPCContext::ThreadLocal()->rank;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCIncrMsgSeq")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  *rv = (RPCContext::ThreadLocal()->msg_seq)++;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetMsgSeq")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  *rv = RPCContext::ThreadLocal()->msg_seq;
});

DGL_REGISTER_GLOBAL("distributed.rpc._CAPI_DGLRPCGetServerState")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  auto st = RPCContext::ThreadLocal()->server_state;
  CHECK(st) << "Server state has not been initialized.";
  *rv = st;
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

}  // namespace rpc
}  // namespace dgl
