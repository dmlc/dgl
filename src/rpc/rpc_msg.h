/**
 *  Copyright (c) 2020 by Contributors
 * @file rpc/rpc_msg.h
 * @brief Common headers for remote process call (RPC).
 */
#ifndef DGL_RPC_RPC_MSG_H_
#define DGL_RPC_RPC_MSG_H_

#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/object.h>
#include <dgl/zerocopy_serializer.h>

#include <string>
#include <vector>

namespace dgl {
namespace rpc {

/** @brief RPC message data structure
 *
 * This structure is exposed to Python and can be used as argument or return
 * value in C API.
 */
struct RPCMessage : public runtime::Object {
  /** @brief Service ID */
  int32_t service_id;

  /** @brief Sequence number of this message. */
  int64_t msg_seq;

  /** @brief Client ID. */
  int32_t client_id;

  /** @brief Server ID. */
  int32_t server_id;

  /** @brief Payload buffer carried by this request.*/
  std::string data;

  /** @brief Extra payloads in the form of tensors.*/
  std::vector<runtime::NDArray> tensors;

  /** @brief Group ID. */
  int32_t group_id{0};

  bool Load(dmlc::Stream* stream) {
    stream->Read(&service_id);
    stream->Read(&msg_seq);
    stream->Read(&client_id);
    stream->Read(&server_id);
    stream->Read(&data);
    stream->Read(&tensors);
    stream->Read(&group_id);
    return true;
  }

  void Save(dmlc::Stream* stream) const {
    stream->Write(service_id);
    stream->Write(msg_seq);
    stream->Write(client_id);
    stream->Write(server_id);
    stream->Write(data);
    stream->Write(tensors);
    stream->Write(group_id);
  }

  static constexpr const char* _type_key = "rpc.RPCMessage";
  DGL_DECLARE_OBJECT_TYPE_INFO(RPCMessage, runtime::Object);
};

DGL_DEFINE_OBJECT_REF(RPCMessageRef, RPCMessage);

/** @brief RPC status flag */
enum RPCStatus {
  kRPCSuccess = 0,
  kRPCTimeOut,
};

}  // namespace rpc
}  // namespace dgl

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, dgl::rpc::RPCMessage, true);
}  // namespace dmlc

#endif  // DGL_RPC_RPC_MSG_H_
