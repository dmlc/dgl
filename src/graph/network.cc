/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/network.cc
 * \brief DGL networking related APIs
 */

#include "./network.h"

#include <unordered_map>

#include "./network/communicator.h"
#include "./network/socket_communicator.h"
#include "./network/serialize.h"
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

static std::unordered_map<void*, char*> MemoryBuffer;


////////////////////////////////// Basic Networking Components ////////////////////////////////


// Wrapper for Communicator Send() API
static void SendData(network::Sender* sender,
                     const char* data,
                     int64_t size,
                     int recv_id) {
  int64_t send_size = sender->Send(data, size, recv_id);
  if (send_size <= 0) {
    LOG(FATAL) << "Send error (size: " << send_size << ")";
  }
}

// Wrapper for COmmunicator Recv() API
static void RecvData(network::Receiver* receiver,
                     char* buffer,
                     int64_t buff_size) {
  int64_t recv_size = receiver->Recv(buffer, buff_size);
  if (recv_size <= 0) {
    LOG(FATAL) << "Receive error (size: " << recv_size << ")";
  }
}

DGL_REGISTER_GLOBAL("network._CAPI_DGLSenderCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    network::Sender* sender = new network::SocketSender();
    CommunicatorHandle chandle = static_cast<CommunicatorHandle>(sender);
    try {
      char* buffer = new char[kMaxBufferSize];
      MemoryBuffer[chandle] = buffer;
    } catch (const std::bad_alloc&) {
      LOG(FATAL) << "Not enough memory for sender buffer: " << kMaxBufferSize;
    }
    *rv = chandle;
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLReceiverCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    network::Receiver* receiver = new network::SocketReceiver();
    CommunicatorHandle chandle = static_cast<CommunicatorHandle>(receiver);
    try {
      char* buffer = new char[kMaxBufferSize];
      MemoryBuffer[chandle] = buffer;
    } catch (const std::bad_alloc&) {
      LOG(FATAL) << "Not enough memory for receiver buffer: " << kMaxBufferSize;
    }
    *rv = chandle;
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLFinalizeSender")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    sender->Finalize();
    delete [] MemoryBuffer[chandle];
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLFinalizeReceiver")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Receiver* receiver = static_cast<network::SocketReceiver*>(chandle);
    receiver->Finalize();
    delete [] MemoryBuffer[chandle];
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLSenderAddReceiver")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    std::string ip = args[1];
    int port = args[2];
    int recv_id = args[3];
    std::string addr = StringPrintf("socket://%s:%d", ip.c_str(), port);
    network::Sender* sender = static_cast<network::Sender*>(chandle);
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
    std::string addr = StringPrintf("socket://%s:%d", ip.c_str(), port);
    network::Receiver* receiver = static_cast<network::SocketReceiver*>(chandle);
    receiver->Wait(addr.c_str(), num_sender, kQueueSize);
  });


////////////////////////// Distributed Sampler Components ////////////////////////////////


DGL_REGISTER_GLOBAL("network._CAPI_SenderSendSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    int recv_id = args[1];
    GraphHandle ghandle = args[2];
    const IdArray node_mapping = args[3];
    const IdArray edge_mapping = args[4];
    const IdArray layer_offsets = args[5];
    const IdArray flow_offsets = args[6];
    ImmutableGraph *ptr = static_cast<ImmutableGraph*>(ghandle);
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    auto csr = ptr->GetInCSR();
    // Write control message
    char* buffer = MemoryBuffer[chandle];
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
    char* buffer = MemoryBuffer[chandle];
    *buffer = CONTROL_END_SIGNAL;
    // Send msg via network
    SendData(sender, buffer, sizeof(CONTROL_END_SIGNAL), recv_id);
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverRecvSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Receiver* receiver = static_cast<network::SocketReceiver*>(chandle);
    // Recv data from network
    char* buffer = MemoryBuffer[chandle];
    RecvData(receiver, buffer, kMaxBufferSize);
    int control = *buffer;
    if (control == CONTROL_NODEFLOW) {
      NodeFlow* nf = new NodeFlow();
      CSRPtr csr;
      // Deserialize nodeflow from recv_data_buffer
      network::DeserializeSampledSubgraph(buffer+sizeof(CONTROL_NODEFLOW),
                                          &(csr),
                                          &(nf->node_mapping),
                                          &(nf->edge_mapping),
                                          &(nf->layer_offsets),
                                          &(nf->flow_offsets));
      nf->graph = GraphPtr(new ImmutableGraph(csr, nullptr));
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
