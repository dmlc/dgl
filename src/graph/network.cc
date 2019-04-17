/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/network.cc
 * \brief DGL networking related APIs
 */

#include "./network.h"
#include "./network/communicator.h"
#include "./network/socket_communicator.h"
#include "./network/serialize.h"

#include "../c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::PackedFunc;
using dgl::runtime::NDArray;

namespace dgl {
namespace network {

static char* SEND_BUFFER = nullptr;
static char* RECV_BUFFER = nullptr;

DGL_REGISTER_GLOBAL("network._CAPI_DGLSenderCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    try {
      SEND_BUFFER = new char[kMaxBufferSize];
    } catch (const std::bad_alloc&) {
      LOG(FATAL) << "Not enough memory for sender buffer: " << kMaxBufferSize;
    }
    network::Sender* sender = new network::SocketSender();
    CommunicatorHandle chandle = static_cast<CommunicatorHandle>(sender);
    *rv = chandle;
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLFinalizeSender")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    sender->Finalize();
    delete [] SEND_BUFFER;
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLSenderAddReceiver")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    std::string ip = args[1];
    int port = args[2];
    int recv_id = args[3];
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    sender->AddReceiver(ip.c_str(), port, recv_id);
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLSenderConnect")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    if (sender->Connect() == false) {
      LOG(FATAL) << "Sender connection failed.";
    }
  });

DGL_REGISTER_GLOBAL("network._CAPI_SenderSendSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    int recv_id = args[1];
    GraphHandle ghandle = args[2];
    NodeFlow nf;
    DLManagedTensor *mt;

    nf.layer_offsets = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[3]));
    nf.flow_offsets = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[4]));
    nf.node_mapping = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[5]));
    if ((mt = CreateTmpDLManagedTensor(args[6])) == nullptr) {
      nf.edge_mapping_available = false;
    } else {
      nf.edge_mapping_available = true;
      nf.edge_mapping = IdArray::FromDLPack(mt);
    }
    std::string node_data_name = args[7];
    nf.node_data_name = std::move(node_data_name);
    if (!nf.node_data_name.empty())
      nf.node_data = NDArray::FromDLPack(CreateTmpDLManagedTensor(args[8]));
    std::string edge_data_name = args[9];
    nf.edge_data_name = std::move(edge_data_name);
    if (!nf.edge_data_name.empty())
      nf.edge_data = NDArray::FromDLPack(CreateTmpDLManagedTensor(args[10]));

    ImmutableGraph *ptr = static_cast<ImmutableGraph*>(ghandle);
    // Serialize nodeflow to data buffer
    int64_t data_size = network::SerializeNodeFlow(SEND_BUFFER, ptr, nf);
    CHECK_GT(data_size, 0);
    // Send msg via network
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    int64_t size = sender->Send(SEND_BUFFER, data_size, recv_id);
    if (size <= 0) {
      LOG(FATAL) << "Send message error (size: " << size << ")";
    }
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLReceiverCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    try {
      RECV_BUFFER = new char[kMaxBufferSize];
    } catch (const std::bad_alloc&) {
      LOG(FATAL) << "Not enough memory for receiver buffer: " << kMaxBufferSize;
    }
    network::Receiver* receiver = new network::SocketReceiver();
    CommunicatorHandle chandle = static_cast<CommunicatorHandle>(receiver);
    *rv = chandle;
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLFinalizeReceiver")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Receiver* receiver = static_cast<network::SocketReceiver*>(chandle);
    receiver->Finalize();
    delete [] RECV_BUFFER;
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLReceiverWait")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    std::string ip = args[1];
    int port = args[2];
    int num_sender = args[3];
    network::Receiver* receiver = static_cast<network::SocketReceiver*>(chandle);
    receiver->Wait(ip.c_str(), port, num_sender, kQueueSize);
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverRecvSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Receiver* receiver = static_cast<network::SocketReceiver*>(chandle);
    // Recv data from network
    int64_t size = receiver->Recv(RECV_BUFFER, kMaxBufferSize);
    if (size <= 0) {
      LOG(FATAL) << "Receive error: (size: " << size << ")";
    }
    NodeFlow* nf = new NodeFlow();
    // Deserialize nodeflow from recv_data_buffer
    network::DeserializeNodeFlow(RECV_BUFFER, nf);
    *rv = static_cast<NodeFlowHandle>(nf);
  });

}  // namespace network
}  // namespace dgl
