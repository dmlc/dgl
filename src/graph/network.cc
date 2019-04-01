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
    network::Communicator* comm = static_cast<network::Communicator*>(chandle);
    NodeFlowHandle nfhandle = args[1];
    GraphHandle ghandle = args[2];
    const NodeFlow *nf = static_cast<const NodeFlow *>(nfhandle);
    const GraphInterface *gptr = static_cast<const GraphInterface *>(ghandle);
    const ImmutableGraph *graph = dynamic_cast<const ImmutableGraph *>(gptr);

    int64_t data_size = nf->Serialize(sender_data_buffer, graph);
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
    int64_t decode_size;

    NodeFlowHandle nfhandle = NodeFlow::Deserialize(recv_data_buffer, &decode_size);
    *rv = nfhandle;
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
