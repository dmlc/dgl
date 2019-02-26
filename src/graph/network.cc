/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/network.cc
 * \brief DGL networking related APIs
 */

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

const int kMaxBufferSize = 200 * 1024 * 1024;  // 200 MB

// Convert Sampled Subgraph structures to PackedFunc.
PackedFunc ConvertSubgraphToPackedFunc(const std::vector<NodeFlow>& sg) {
  auto body = [sg] (DGLArgs args, DGLRetValue* rv) {
      const uint64_t which = args[0];
      if (which < sg.size()) {
        GraphInterface* gptr = sg[which].graph->Reset();
        GraphHandle ghandle = gptr;
        *rv = ghandle;
      } else if (which >= sg.size() && which < sg.size() * 2) {
        *rv = std::move(sg[which - sg.size()].node_mapping);
      } else if (which >= sg.size() * 2 && which < sg.size() * 3) {
        *rv = std::move(sg[which - sg.size() * 2].edge_mapping);
      } else if (which >= sg.size() * 3 && which < sg.size() * 4) {
        *rv = std::move(sg[which - sg.size() * 3].layer_offsets);
      } else if (which >= sg.size() * 4 && which < sg.size() * 5) {
        *rv = std::move(sg[which - sg.size() * 4].flow_offsets);
      } else {
        LOG(FATAL) << "invalid choice";
      }
    };
  return PackedFunc(body);
}

DGL_REGISTER_GLOBAL("network._CAPI_DGLSenderCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string ip = args[0];
    int port = args[1];
    network::Communicator* comm = new network::SocketCommunicator();
    comm->Initialize(true, ip.c_str(), port);
    CommunicatorHandle chandle = static_cast<CommunicatorHandle>(comm);
    *rv = chandle;
  });

DGL_REGISTER_GLOBAL("network._CAPI_SenderSendSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    GraphHandle ghandle = args[1];
    ImmutableGraph *ptr = static_cast<ImmutableGraph*>(ghandle);
    network::Communicator* comm = static_cast<network::Communicator*>(chandle);
    auto csr = ptr->GetInCSR();  // We only care about in_csr
    const IdArray node_mapping = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[2]));
    const IdArray edge_mapping = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[3]));
    const IdArray layer_offsets = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[4]));
    const IdArray flow_offsets = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[5]));
    char* data_buffer = nullptr;
    int64_t data_size = network::SerializeSampledSubgraph(
                             &data_buffer,
                             csr,
                             node_mapping,
                             edge_mapping,
                             layer_offsets,
                             flow_offsets);
    // Send msg via network
    int size = comm->Send(data_buffer, data_size);
    if (size <= 0) {
      LOG(ERROR) << "Send message erro (size: " << size << ")";
    }
    if (data_buffer != nullptr) {
      delete [] data_buffer;
    }
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLReceiverCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string ip = args[0];
    int port = args[1];
    int num_sender = args[2];
    int queue_size = args[3];
    network::Communicator* comm = new network::SocketCommunicator();
    comm->Initialize(false, ip.c_str(), port, num_sender, queue_size);
    CommunicatorHandle chandle = static_cast<CommunicatorHandle>(comm);
    *rv = chandle;
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverRecvSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    // The data buffer will be allocated just once
    static char* data_buffer = new char[kMaxBufferSize];
    CommunicatorHandle chandle = args[0];
    network::Communicator* comm = static_cast<network::Communicator*>(chandle);
    int size = comm->Receive(data_buffer, kMaxBufferSize);
    if (size <= 0) {
      LOG(ERROR) << "Receive error: (size: " << size << ")";
    }
    NodeFlow nf;
    network::DeserializeSampledSubgraph(&nf, data_buffer);
    std::vector<NodeFlow> subgs(1);
    subgs[0] = nf;
    *rv = ConvertSubgraphToPackedFunc(subgs);
  });

}  // namespace network
}  // namespace dgl
