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

// Wrapper for Send api
static void SendData(network::Sender* sender,
                     const char* data,
                     int64_t size,
                     int recv_id) {
  int64_t send_size = sender->Send(data, size, recv_id);
  if (send_size <= 0) {
    LOG(FATAL) << "Send error (size: " << send_size << ")";
  }
}

// Wrapper for Recv api
static void RecvData(network::Receiver* receiver,
                     char* dest,
                     int64_t max_size) {
  int64_t recv_size = receiver->Recv(dest, max_size);
  if (recv_size <= 0) {
    LOG(FATAL) << "Receive error (size: " << recv_size << ")";
  }
}

DGL_REGISTER_GLOBAL("network._CAPI_DGLSenderCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    network::Sender* sender = new network::SocketSender();
    try {
      char* buffer = new char[kMaxBufferSize];
      sender->SetBuffer(buffer);
    } catch (const std::bad_alloc&) {
      LOG(FATAL) << "Not enough memory for sender buffer: " << kMaxBufferSize;
    }
    CommunicatorHandle chandle = static_cast<CommunicatorHandle>(sender);
    *rv = chandle;
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLFinalizeSender")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    sender->Finalize();
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
    const IdArray node_mapping = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[3]));
    const IdArray edge_mapping = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[4]));
    const IdArray layer_offsets = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[5]));
    const IdArray flow_offsets = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[6]));
    ImmutableGraph *ptr = static_cast<ImmutableGraph*>(ghandle);
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    auto csr = ptr->GetInCSR();
    // Write control message
    char* buffer = sender->GetBuffer();
    *buffer = CONTROL_NODEFLOW;
    // Serialize nodeflow to data buffer
    int64_t data_size = network::SerializeSampledSubgraph(
                             buffer+sizeof(CONTROL_NODEFLOW),
                             csr,
                             node_mapping,
                             edge_mapping,
                             layer_offsets,
                             flow_offsets);
    CHECK_GT(data_size, 0);
    data_size += sizeof(CONTROL_NODEFLOW);
    // Send msg via network
    SendData(sender, buffer, data_size, recv_id);
  });

DGL_REGISTER_GLOBAL("network._CAPI_SenderSendEndSignal")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    int recv_id = args[1];
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    char* buffer = sender->GetBuffer();
    *buffer = CONTROL_END_SIGNAL;
    // Send msg via network
    SendData(sender, buffer, sizeof(CONTROL_END_SIGNAL), recv_id);
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLReceiverCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    network::Receiver* receiver = new network::SocketReceiver();
    try {
      char* buffer = new char[kMaxBufferSize];
      receiver->SetBuffer(buffer);
    } catch (const std::bad_alloc&) {
      LOG(FATAL) << "Not enough memory for receiver buffer: " << kMaxBufferSize;
    }
    CommunicatorHandle chandle = static_cast<CommunicatorHandle>(receiver);
    *rv = chandle;
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLFinalizeReceiver")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Receiver* receiver = static_cast<network::SocketReceiver*>(chandle);
    receiver->Finalize();
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
    char* buffer = receiver->GetBuffer();
    RecvData(receiver, buffer, kMaxBufferSize);
    int control = *buffer;
    if (control == CONTROL_NODEFLOW) {
      NodeFlow* nf = new NodeFlow();
      ImmutableGraph::CSR::Ptr csr;
      // Deserialize nodeflow from recv_data_buffer
      network::DeserializeSampledSubgraph(buffer+sizeof(CONTROL_NODEFLOW),
                                          &(csr),
                                          &(nf->node_mapping),
                                          &(nf->edge_mapping),
                                          &(nf->layer_offsets),
                                          &(nf->flow_offsets));
      nf->graph = GraphPtr(new ImmutableGraph(csr, nullptr, false));
      std::vector<NodeFlow*> subgs(1);
      subgs[0] = nf;
      *rv = WrapVectorReturn(subgs);
    } else if (control == CONTROL_END_SIGNAL) {
      *rv = CONTROL_END_SIGNAL;
    } else {
      LOG(FATAL) << "Unknow control number: " << control;
    }
  });

}  // namespace network
}  // namespace dgl
