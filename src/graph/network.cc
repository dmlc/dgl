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

DGL_REGISTER_GLOBAL("network._CAPI_DGLSenderCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string ip = args[0];
    int port = args[1];
    network::Communicator* comm = new network::SocketCommunicator();
    if (comm->Initialize(IS_SENDER, ip.c_str(), port) == false) {
      LOG(FATAL) << "Initialize network communicator (sender) error.";
    }
    try {
      comm->SetBuffer(new char[kMaxBufferSize]);
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
      comm->SetBuffer(new char[kMaxBufferSize]);
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
    const IdArray node_mapping = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[2]));
    const IdArray edge_mapping = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[3]));
    const IdArray layer_offsets = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[4]));
    const IdArray flow_offsets = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[5]));
    ImmutableGraph *ptr = static_cast<ImmutableGraph*>(ghandle);
    network::Communicator* comm = static_cast<network::Communicator*>(chandle);
    auto csr = ptr->GetInCSR();
    // Serialize nodeflow to data buffer
    int64_t data_size = network::SerializeSampledSubgraph(
                             comm->GetBuffer(),
                             csr,
                             node_mapping,
                             edge_mapping,
                             layer_offsets,
                             flow_offsets);
    CHECK_GT(data_size, 0);
    // Send msg via network
    int64_t size = comm->Send(comm->GetBuffer(), data_size);
    if (size <= 0) {
      LOG(ERROR) << "Send message error (size: " << size << ")";
    }
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverRecvSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Communicator* comm = static_cast<network::Communicator*>(chandle);
    // Recv data from network
    int64_t size = comm->Receive(comm->GetBuffer(), kMaxBufferSize);
    if (size <= 0) {
      LOG(ERROR) << "Receive error: (size: " << size << ")";
    }
    NodeFlow* nf = new NodeFlow();
    ImmutableGraph::CSR::Ptr csr;
    // Deserialize nodeflow from recv_data_buffer
    network::DeserializeSampledSubgraph(comm->GetBuffer(),
                                        &(csr),
                                        &(nf->node_mapping),
                                        &(nf->edge_mapping),
                                        &(nf->layer_offsets),
                                        &(nf->flow_offsets));
    nf->graph = GraphPtr(new ImmutableGraph(csr, nullptr, false));
    std::vector<NodeFlow*> subgs(1);
    subgs[0] = nf;
    *rv = WrapVectorReturn(subgs);
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLFinalizeCommunicator")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Communicator* comm = static_cast<network::Communicator*>(chandle);
    comm->Finalize();
  });

}  // namespace network
}  // namespace dgl
