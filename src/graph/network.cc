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

static char* sender_data_buffer = nullptr;
static char* recv_data_buffer = nullptr;

///////////////////////// Distributed Sampler /////////////////////////

DGL_REGISTER_GLOBAL("network._CAPI_DGLSenderCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string ip = args[0];
    int port = args[1];
    network::Communicator* comm = new network::SocketCommunicator();
    if (comm->Initialize(IS_SENDER, ip.c_str(), port) == false) {
      LOG(FATAL) << "Initialize network communicator (sender) error.";
    }
    try {
      sender_data_buffer = new char[kMaxBufferSize];
    } catch (const std::bad_alloc&) {
      LOG(FATAL) << "Not enough memory for sender buffer: " << kMaxBufferSize;
    }
    CommunicatorHandle chandle = static_cast<CommunicatorHandle>(comm);
    *rv = chandle;
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLReceiverCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string ip = args[0];
    int port = args[1];
    int num_sender = args[2];
    network::Communicator* comm = new network::SocketCommunicator();
    if (comm->Initialize(IS_RECEIVER, ip.c_str(), port, num_sender, kQueueSize) == false) {
      LOG(FATAL) << "Initialize network communicator (receiver) error.";
    }
    try {
      recv_data_buffer = new char[kMaxBufferSize];
    } catch (const std::bad_alloc&) {
      LOG(FATAL) << "Not enough memory for receiver buffer: " << kMaxBufferSize;
    }
    CommunicatorHandle chandle = static_cast<CommunicatorHandle>(comm);
    *rv = chandle;
  });

DGL_REGISTER_GLOBAL("network._CAPI_SenderSendSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    GraphHandle ghandle = args[1];
    NodeFlow nf;
    DLManagedTensor *mt;

    nf.layer_offsets = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[2]));
    nf.flow_offsets = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[3]));
    nf.node_mapping = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[4]));
    if ((mt = CreateTmpDLManagedTensor(args[5])) == nullptr) {
      nf.edge_mapping_available = false;
    } else {
      nf.edge_mapping_available = true;
      nf.edge_mapping = IdArray::FromDLPack(mt);
    }
    nf.node_data_name = args[6];
    if (!nf.node_data_name.empty())
      nf.node_data = NDArray::FromDLPack(CreateTmpDLManagedTensor(args[7]));
    nf.edge_data_name = args[8];
    if (!nf.edge_data_name.empty())
      nf.edge_data = NDArray::FromDLPack(CreateTmpDLManagedTensor(args[9]));

    ImmutableGraph *ptr = static_cast<ImmutableGraph*>(ghandle);
    network::Communicator* comm = static_cast<network::Communicator*>(chandle);
    // Serialize nodeflow to data buffer
    int64_t data_size = network::SerializeNodeFlow(sender_data_buffer, ptr, nf);
    CHECK_GT(data_size, 0);
    // Send msg via network
    int64_t size = comm->Send(sender_data_buffer, data_size);
    if (size <= 0) {
      LOG(ERROR) << "Send message error (size: " << size << ")";
    }
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverRecvSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Communicator* comm = static_cast<network::Communicator*>(chandle);
    // Recv data from network
    int64_t size = comm->Receive(recv_data_buffer, kMaxBufferSize);
    if (size <= 0) {
      LOG(ERROR) << "Receive error: (size: " << size << ")";
    }
    NodeFlow* nf = new NodeFlow();
    // Deserialize nodeflow from recv_data_buffer
    network::DeserializeNodeFlow(recv_data_buffer, nf);
    *rv = static_cast<NodeFlowHandle>(nf);
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLFinalizeCommunicator")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Communicator* comm = static_cast<network::Communicator*>(chandle);
    comm->Finalize();
    delete [] sender_data_buffer;
    delete [] recv_data_buffer;
  });

}  // namespace network
}  // namespace dgl
