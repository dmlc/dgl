/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/network.cc
 * \brief DGL networking related APIs
 */
#include "./network.h"

#include <dgl/runtime/container.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/packed_func_ext.h>
#include <dgl/immutable_graph.h>
#include <dgl/nodeflow.h>

#include <unordered_map>

#include "./network/communicator.h"
#include "./network/socket_communicator.h"
#include "./network/msg_queue.h"
#include "./network/common.h"

using dgl::network::StringPrintf;
using namespace dgl::runtime;

namespace dgl {
namespace network {

void MsgMeta::AddArray(const NDArray& array) {
  // We first write the ndim to the data_shape_
  data_shape_.push_back(static_cast<int64_t>(array->ndim));
  // Then we write the data shape
  for (int i = 0; i < array->ndim; ++i) {
    data_shape_.push_back(array->shape[i]);
  }
  ndarray_count_++;
}

char* MsgMeta::Serialize(int64_t* size) {
  char* buffer = nullptr;
  int64_t buffer_size = 0;
  buffer_size += sizeof(msg_type_);
  if (ndarray_count_ != 0) {
    buffer_size += sizeof(ndarray_count_);
    buffer_size += sizeof(data_shape_.size());
    buffer_size += sizeof(int64_t) * data_shape_.size();
  }
  buffer = new char[buffer_size];
  char* pointer = buffer;
  // Write msg_type_
  *(reinterpret_cast<int*>(pointer)) = msg_type_;
  pointer += sizeof(msg_type_);
  if (ndarray_count_ != 0) {
    // Write ndarray_count_
    *(reinterpret_cast<int*>(pointer)) = ndarray_count_;
    pointer += sizeof(ndarray_count_);
    // Write size of data_shape_
    *(reinterpret_cast<size_t*>(pointer)) = data_shape_.size();
    pointer += sizeof(data_shape_.size());
    // Write data of data_shape_
    memcpy(pointer,
        reinterpret_cast<char*>(data_shape_.data()),
        sizeof(int64_t) * data_shape_.size());
  }
  *size = buffer_size;
  return buffer;
}

void MsgMeta::Deserialize(char* buffer, int64_t size) {
  int64_t data_size = 0;
  // Read mesg_type_
  msg_type_ = *(reinterpret_cast<int*>(buffer));
  buffer += sizeof(int);
  data_size += sizeof(int);
  if (data_size < size) {
    // Read ndarray_count_
    ndarray_count_ = *(reinterpret_cast<int*>(buffer));
    buffer += sizeof(int);
    data_size += sizeof(int);
    // Read size of data_shape_
    size_t count = *(reinterpret_cast<size_t*>(buffer));
    buffer += sizeof(size_t);
    data_size += sizeof(size_t);
    data_shape_.resize(count);
    // Read data of data_shape_
    memcpy(data_shape_.data(), buffer,
        count * sizeof(int64_t));
    data_size += count * sizeof(int64_t);
  }
  CHECK_EQ(data_size, size);
}

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
    GraphRef g = args[2];
    NDArray node_mapping = args[3];
    NDArray edge_mapping = args[4];
    NDArray layer_offsets = args[5];
    NDArray flow_offsets = args[6];
    auto ptr = std::dynamic_pointer_cast<ImmutableGraph>(g.sptr());
    CHECK(ptr) << "only immutable graph is allowed in send/recv";
    auto csr = ptr->GetInCSR();
    // Create a message for the meta data of ndarray
    NDArray indptr = csr->indptr();
    NDArray indice = csr->indices();
    NDArray edge_ids = csr->edge_ids();
    MsgMeta msg(kNodeFlowMsg);
    msg.AddArray(node_mapping);
    msg.AddArray(edge_mapping);
    msg.AddArray(layer_offsets);
    msg.AddArray(flow_offsets);
    msg.AddArray(indptr);
    msg.AddArray(indice);
    msg.AddArray(edge_ids);
    // send meta message
    int64_t size = 0;
    char* data = msg.Serialize(&size);
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    Message send_msg;
    send_msg.data = data;
    send_msg.size = size;
    send_msg.deallocator = DefaultMessageDeleter;
    CHECK_NE(sender->Send(send_msg, recv_id), -1);
    // send node_mapping
    Message node_mapping_msg;
    node_mapping_msg.data = static_cast<char*>(node_mapping->data);
    node_mapping_msg.size = node_mapping.GetSize();
    // capture the array in the closure
    node_mapping_msg.deallocator = [node_mapping](Message*) {};
    CHECK_NE(sender->Send(node_mapping_msg, recv_id), -1);
    // send edege_mapping
    Message edge_mapping_msg;
    edge_mapping_msg.data = static_cast<char*>(edge_mapping->data);
    edge_mapping_msg.size = edge_mapping.GetSize();
    // capture the array in the closure
    edge_mapping_msg.deallocator = [edge_mapping](Message*) {};
    CHECK_NE(sender->Send(edge_mapping_msg, recv_id), -1);
    // send layer_offsets
    Message layer_offsets_msg;
    layer_offsets_msg.data = static_cast<char*>(layer_offsets->data);
    layer_offsets_msg.size = layer_offsets.GetSize();
    // capture the array in the closure
    layer_offsets_msg.deallocator = [layer_offsets](Message*) {};
    CHECK_NE(sender->Send(layer_offsets_msg, recv_id), -1);
    // send flow_offset
    Message flow_offsets_msg;
    flow_offsets_msg.data = static_cast<char*>(flow_offsets->data);
    flow_offsets_msg.size = flow_offsets.GetSize();
    // capture the array in the closure
    flow_offsets_msg.deallocator = [flow_offsets](Message*) {};
    CHECK_NE(sender->Send(flow_offsets_msg, recv_id), -1);
    // send csr->indptr
    Message indptr_msg;
    indptr_msg.data = static_cast<char*>(indptr->data);
    indptr_msg.size = indptr.GetSize();
    // capture the array in the closure
    indptr_msg.deallocator = [indptr](Message*) {};
    CHECK_NE(sender->Send(indptr_msg, recv_id), -1);
    // send csr->indices
    Message indices_msg;
    indices_msg.data = static_cast<char*>(indice->data);
    indices_msg.size = indice.GetSize();
    // capture the array in the closure
    indices_msg.deallocator = [indice](Message*) {};
    CHECK_NE(sender->Send(indices_msg, recv_id), -1);
    // send csr->edge_ids
    Message edge_ids_msg;
    edge_ids_msg.data = static_cast<char*>(edge_ids->data);
    edge_ids_msg.size = edge_ids.GetSize();
    // capture the array in the closure
    edge_ids_msg.deallocator = [edge_ids](Message*) {};
    CHECK_NE(sender->Send(edge_ids_msg, recv_id), -1);
  });

DGL_REGISTER_GLOBAL("network._CAPI_SenderSendSamplerEndSignal")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    int recv_id = args[1];
    MsgMeta msg(kEndMsg);
    int64_t size = 0;
    char* data = msg.Serialize(&size);
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    Message send_msg = {data, size};
    send_msg.deallocator = DefaultMessageDeleter;
    CHECK_NE(sender->Send(send_msg, recv_id), -1);
  });

static void ConstructNFTensor(DLTensor *tensor, char* data, int64_t shape_0) {
  tensor->data = data;
  tensor->ctx = DLContext{kDLCPU, 0};
  tensor->ndim = 1;
  tensor->dtype = DLDataType{kDLInt, 64, 1};
  tensor->shape = new int64_t[1];
  tensor->shape[0] = shape_0;
  tensor->byte_offset = 0;
}

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverRecvNodeFlow")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Receiver* receiver = static_cast<network::SocketReceiver*>(chandle);
    int send_id = 0;
    Message recv_msg;
    receiver->Recv(&recv_msg, &send_id);
    MsgMeta msg(recv_msg.data, recv_msg.size);
    recv_msg.deallocator(&recv_msg);
    if (msg.msg_type() == kNodeFlowMsg) {
      CHECK_EQ(msg.ndarray_count() * 2, msg.data_shape_.size());
      NodeFlow nf = NodeFlow::Create();
      // node_mapping
      Message array_0;
      CHECK_NE(receiver->RecvFrom(&array_0, send_id), -1);
      CHECK_EQ(msg.data_shape_[0], 1);
      DLTensor node_mapping_tensor;
      ConstructNFTensor(&node_mapping_tensor, array_0.data, msg.data_shape_[1]);
      DLManagedTensor *node_mapping_managed_tensor = new DLManagedTensor();
      node_mapping_managed_tensor->dl_tensor = node_mapping_tensor;
      nf->node_mapping = NDArray::FromDLPack(node_mapping_managed_tensor);
      // edge_mapping
      Message array_1;
      CHECK_NE(receiver->RecvFrom(&array_1, send_id), -1);
      CHECK_EQ(msg.data_shape_[2], 1);
      DLTensor edge_mapping_tensor;
      ConstructNFTensor(&edge_mapping_tensor, array_1.data, msg.data_shape_[3]);
      DLManagedTensor *edge_mapping_managed_tensor = new DLManagedTensor();
      edge_mapping_managed_tensor->dl_tensor = edge_mapping_tensor;
      nf->edge_mapping = NDArray::FromDLPack(edge_mapping_managed_tensor);
      // layer_offset
      Message array_2;
      CHECK_NE(receiver->RecvFrom(&array_2, send_id), -1);
      CHECK_EQ(msg.data_shape_[4], 1);
      DLTensor layer_offsets_tensor;
      ConstructNFTensor(&layer_offsets_tensor, array_2.data, msg.data_shape_[5]);
      DLManagedTensor *layer_offsets_managed_tensor = new DLManagedTensor();
      layer_offsets_managed_tensor->dl_tensor = layer_offsets_tensor;
      nf->layer_offsets = NDArray::FromDLPack(layer_offsets_managed_tensor);
      // flow_offset
      Message array_3;
      CHECK_NE(receiver->RecvFrom(&array_3, send_id), -1);
      CHECK_EQ(msg.data_shape_[6], 1);
      DLTensor flow_offsets_tensor;
      ConstructNFTensor(&flow_offsets_tensor, array_3.data, msg.data_shape_[7]);
      DLManagedTensor *flow_offsets_managed_tensor = new DLManagedTensor();
      flow_offsets_managed_tensor->dl_tensor = flow_offsets_tensor;
      nf->flow_offsets = NDArray::FromDLPack(flow_offsets_managed_tensor);
      // CSR indptr
      Message array_4;
      CHECK_NE(receiver->RecvFrom(&array_4, send_id), -1);
      CHECK_EQ(msg.data_shape_[8], 1);
      DLTensor indptr_tensor;
      ConstructNFTensor(&indptr_tensor, array_4.data, msg.data_shape_[9]);
      DLManagedTensor *indptr_managed_tensor = new DLManagedTensor();
      indptr_managed_tensor->dl_tensor = indptr_tensor;
      NDArray indptr = NDArray::FromDLPack(indptr_managed_tensor);
      // CSR indice
      Message array_5;
      CHECK_NE(receiver->RecvFrom(&array_5, send_id), -1);
      CHECK_EQ(msg.data_shape_[10], 1);
      DLTensor indice_tensor;
      ConstructNFTensor(&indice_tensor, array_5.data, msg.data_shape_[11]);
      DLManagedTensor *indice_managed_tensor = new DLManagedTensor();
      indice_managed_tensor->dl_tensor = indice_tensor;
      NDArray indice = NDArray::FromDLPack(indice_managed_tensor);
      // CSR edge_ids
      Message array_6;
      CHECK_NE(receiver->RecvFrom(&array_6, send_id), -1);
      CHECK_EQ(msg.data_shape_[12], 1);
      DLTensor edge_id_tensor;
      ConstructNFTensor(&edge_id_tensor, array_6.data, msg.data_shape_[13]);
      DLManagedTensor *edge_id_managed_tensor = new DLManagedTensor();
      edge_id_managed_tensor->dl_tensor = edge_id_tensor;
      NDArray edge_ids = NDArray::FromDLPack(edge_id_managed_tensor);
      // Create CSR
      CSRPtr csr(new CSR(indptr, indice, edge_ids));
      nf->graph = GraphPtr(new ImmutableGraph(csr, nullptr));
      *rv = nf;
    } else if (msg.msg_type() == kEndMsg) {
      *rv = msg.msg_type();
    } else {
      LOG(FATAL) << "Unknown message type: " << msg.msg_type();
    }
  });

}  // namespace network
}  // namespace dgl
