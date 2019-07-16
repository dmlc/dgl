/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/network.cc
 * \brief DGL networking related APIs
 */

#include "./network.h"

#include <unordered_map>

#include <dgl/immutable_graph.h>
#include <dgl/nodeflow.h>

#include "./network/communicator.h"
#include "./network/socket_communicator.h"
#include "./network/common.h"
#include "../c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::PackedFunc;
using dgl::runtime::NDArray;
using dgl::network::StringPrintf;

namespace dgl {
namespace network {


////////////////////////////////// Basic Networking Components ////////////////////////////////


DGL_REGISTER_GLOBAL("network._CAPI_DGLSenderCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string type = args[0];
    network::Sender* sender = nullptr;
    if (type == "socket") {
      sender = new network::SocketSender(kQueueSize);
    } else {
      LOG(FATAL) << "Unknown communicator type: " << type;
    }
    CommunicatorHandle chandle = static_cast<CommunicatorHandle>(sender);
    *rv = chandle;
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLReceiverCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string type = args[0];
    network::Receiver* receiver = nullptr;
    if (type == "socket") {
      receiver = new network::SocketReceiver(kQueueSize);
    } else {
      LOG(FATAL) << "Unknown communicator type: " << type;
    }
    CommunicatorHandle chandle = static_cast<CommunicatorHandle>(receiver);
    *rv = chandle;
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLFinalizeSender")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    sender->Finalize();
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLFinalizeReceiver")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Receiver* receiver = static_cast<network::SocketReceiver*>(chandle);
    receiver->Finalize();
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLSenderAddReceiver")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    std::string ip = args[1];
    int port = args[2];
    int recv_id = args[3];
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    std::string addr;
    if (sender->Type() == "socket") {
      addr = StringPrintf("socket://%s:%d", ip.c_str(), port);
    } else {
      LOG(FATAL) << "Unknown communicator type: " << sender->Type();
    }
    sender->AddReceiver(addr.c_str(), recv_id);
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLSenderConnect")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    if (sender->Connect() == false) {
      LOG(FATAL) << "Sender connection failed.";
    }
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLReceiverWait")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    std::string ip = args[1];
    int port = args[2];
    int num_sender = args[3];
    network::Receiver* receiver = static_cast<network::SocketReceiver*>(chandle);
    std::string addr;
    if (receiver->Type() == "socket") {
      addr = StringPrintf("socket://%s:%d", ip.c_str(), port);
    } else {
      LOG(FATAL) << "Unknown communicator type: " << receiver->Type();
    }
    if (receiver->Wait(addr.c_str(), num_sender) == false) {
      LOG(FATAL) << "Wait sender socket failed.";
    }
  });


////////////////////////// Distributed Sampler Components ////////////////////////////////


DGL_REGISTER_GLOBAL("network._CAPI_SenderSendNodeFlow")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    int recv_id = args[1];
    GraphHandle ghandle = args[2];
    const NDArray node_mapping = args[3];
    const NDArray edge_mapping = args[4];
    const NDArray layer_offsets = args[5];
    const NDArray flow_offsets = args[6];
    ImmutableGraph *ptr = static_cast<ImmutableGraph*>(ghandle);
    auto csr = ptr->GetInCSR();
    // Create a message for the meta data of ndarray
    Message msg(NF_MSG);
    msg.AddShape({node_mapping->shape[0]});
    msg.AddShape({edge_mapping->shape[0]});
    msg.AddShape({layer_offsets->shape[0]});
    msg.AddShape({flow_offsets->shape[0]});
    msg.AddShape({csr->indptr()->shape[0]});
    msg.AddShape({csr->indices()->shape[0]});
    msg.AddShape({csr->edge_ids()->shape[0]});
    // First we send meta message
    int64_t size = 0;
    char* data = msg.Serialize(&size);
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    sender->Send(data, size, recv_id);
    // Then we send a set of ndarray
    /*
    sender->Send(static_cast<char*>(node_mapping->data), node_mapping.GetSize(), recv_id);
    sender->Send(static_cast<char*>(edge_mapping->data), edge_mapping.GetSize(), recv_id);
    sender->Send(static_cast<char*>(layer_offsets->data), layer_offsets.GetSize(), recv_id);
    sender->Send(static_cast<char*>(flow_offsets->data), flow_offsets.GetSize(), recv_id);
    sender->Send(static_cast<char*>(csr->indptr()->data), csr->indptr().GetSize(), recv_id);
    sender->Send(static_cast<char*>(csr->indices()->data), csr->indices().GetSize(), recv_id);
    sender->Send(static_cast<char*>(csr->edge_ids()->data), csr->edge_ids().GetSize(), recv_id);*/
  });

DGL_REGISTER_GLOBAL("network._CAPI_SenderSendSamplerEndSignal")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    int recv_id = args[1];
    Message msg(END_MSG);
    int64_t size = 0;
    char* data = msg.Serialize(&size);
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    sender->Send(data, size, recv_id);
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverRecvNodeFlow")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Receiver* receiver = static_cast<network::SocketReceiver*>(chandle);
    int64_t data_size = 0;
    int send_id = 0;
    char* data = receiver->Recv(&data_size, &send_id);
    Message msg(data, data_size);
    if (msg.Type() == NF_MSG) {
      NodeFlow* nf = new NodeFlow();
      // node_mapping
      data = receiver->RecvFrom(&data_size, send_id);
      nf->node_mapping = NDArray::Empty(
        {msg.data_shape_[0]}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
      memcpy(static_cast<char*>(nf->node_mapping->data), data, data_size);
      delete [] data;
      // edge_mapping
      data = receiver->RecvFrom(&data_size, send_id);
      nf->edge_mapping = NDArray::Empty(
        {msg.data_shape_[1]}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
      memcpy(static_cast<char*>(nf->edge_mapping->data), data, data_size);
      delete [] data;
      // layer_offset
      data = receiver->RecvFrom(&data_size, send_id);
      nf->layer_offsets = NDArray::Empty(
        {msg.data_shape_[2]}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
      memcpy(static_cast<char*>(nf->layer_offsets->data), data, data_size);
      delete [] data;
      // flow_offset
      data = receiver->RecvFrom(&data_size, send_id);
      nf->flow_offsets = NDArray::Empty(
        {msg.data_shape_[3]}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
      memcpy(static_cast<char*>(nf->flow_offsets->data), data, data_size);
      delete [] data;
      // CSR indptr
      CSRPtr csr(new CSR(msg.data_shape_[0], msg.data_shape_[1], false));
      data = receiver->RecvFrom(&data_size, send_id);
      memcpy(static_cast<char*>(csr->indptr()->data), data, data_size);
      delete [] data;
      // CSR indice
      data = receiver->RecvFrom(&data_size, send_id);
      memcpy(static_cast<char*>(csr->indices()->data), data, data_size);
      delete [] data;
      // CSR edge_ids
      data = receiver->RecvFrom(&data_size, send_id);
      memcpy(static_cast<char*>(csr->edge_ids()->data), data, data_size);
      delete [] data;
      nf->graph = GraphPtr(new ImmutableGraph(csr, nullptr));
      *rv = nf;
    } else if (msg.Type() == END_MSG) {
      *rv = msg.Type();
    } else {
      LOG(FATAL) << "Unknown message type: " << msg.Type();
    }
  });

}  // namespace network
}  // namespace dgl
