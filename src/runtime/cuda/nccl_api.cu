/*!
 *  Copyright (c) 2021 by Contributors
 * \file nccl_api.cc
 * \brief Implementation of wrapper around NCCL routines. 
 */

#include "nccl_api.h"

#include <cuda_fp16.h>

#include <dgl/packed_func_ext.h>
#include <dgl/runtime/registry.h>

namespace dgl {
namespace runtime {
namespace cuda {

namespace {

template<typename T> ncclDataType_t NCCLType();
template<> ncclDataType_t NCCLType<int32_t>() {
    return ncclInt32; 
}
template<> ncclDataType_t NCCLType<int64_t>() {
    return ncclInt64; 
}
template<> ncclDataType_t NCCLType<__half>() {
    return ncclHalf; 
}
template<> ncclDataType_t NCCLType<float>() {
    return ncclFloat32; 
}
template<> ncclDataType_t NCCLType<double>() {
    return ncclFloat64; 
}

}

/* NCCLUniqueId **************************************************************/

NCCLUniqueId::NCCLUniqueId() :
  id_()
{
  // this ID is unique to the process, not to each call of this function
  auto r = ncclGetUniqueId(&id_);
  CHECK_EQ(r, ncclSuccess);
}

ncclUniqueId NCCLUniqueId::Get() const
{
  return id_;
}


/* NCCLCommunicator **********************************************************/

NCCLCommunicator::NCCLCommunicator(
    const int size,
    const int rank,
    ncclUniqueId id) :
  comm_(),
  size_(size),
  rank_(rank)
{
  CHECK_LT(rank, size);
  CHECK_GE(rank, 0);

  auto r = ncclCommInitRank(&comm_, size_, id, rank_);
  CHECK_EQ(r, ncclSuccess);
}

NCCLCommunicator::~NCCLCommunicator()
{
  ncclCommDestroy(comm_);
}

ncclComm_t NCCLCommunicator::Get()
{
  return comm_;
}

void NCCLCommunicator::AllToAll(
    const void * send,
    const int64_t size,
    void * const recv,
    const ncclDataType_t type,
    cudaStream_t stream)
{
  const uint8_t * const send_data = static_cast<const uint8_t*>(send);
  uint8_t * const recv_data = static_cast<uint8_t*>(recv);

  ncclGroupStart();
  for (int r = 0; r < size_; ++r) {
    ncclSend(send_data+(r*size), size, type, r, comm_, stream);
    ncclRecv(recv_data+(r*size), size, type, r, comm_, stream);
  }
  ncclGroupEnd();
}

void NCCLCommunicator::AllToAllV(
    const void * const * const send,
    const int64_t * send_size,
    void * const * const recv,
    const int64_t * recv_size,
    const ncclDataType_t type,
    cudaStream_t stream)
{ 
  ncclGroupStart();
  for (int r = 0; r < size_; ++r) {
    if (send_size[r] > 0) {
      ncclSend(send[r], send_size[r], type, r, comm_, stream);
    }
    if (recv_size[r] > 0) {
      ncclRecv(recv[r], recv_size[r], type, r, comm_, stream);
    }
  }
  ncclGroupEnd();
}

template<typename IdType, typename DType>
void NCCLCommunicator::SparseAllToAll(
      const IdType * const send_idx,
      const DType * const send_value,
      const int64_t * const send_prefix,
      IdType * const recv_idx,
      DType * const recv_value,
      const int64_t * const recv_prefix,
      cudaStream_t stream)
{
  const ncclDataType_t idx_type = NCCLType<IdType>;
  const ncclDataType_t value_type = NCCLType<IdType>;

  ncclGroupStart();
  for (int r = 0; r < size_; ++r) {
    const int64_t send_size = send_prefix[r+1]-send_prefix[r];
    if (send_size > 0) {
      ncclSend(send_idx+send_prefix[r], send_size, idx_type, r, comm_, stream);
      ncclSend(send_value+send_prefix[r], send_size, value_type, r, comm_, stream);
    }
    const int64_t recv_size = recv_prefix[r+1]-recv_prefix[r];
    if (recv_size > 0) {
      ncclRecv(recv_idx+recv_prefix[r], recv_size, idx_type, r, comm_, stream);
      ncclRecv(recv_value+recv_prefix[r], recv_size, value_type, r, comm_, stream);
    }
  }
  ncclGroupEnd();
}

template<>
void NCCLCommunicator::SparseAllToAll<int32_t, __half>(
      const int32_t * const send_idx,
      const __half * const send_value,
      const int64_t * const send_prefix,
      int32_t * const recv_idx,
      __half * const recv_value,
      const int64_t * const recv_prefix,
      cudaStream_t stream);

template<>
void NCCLCommunicator::SparseAllToAll<int64_t, __half>(
      const int64_t * const send_idx,
      const __half * const send_value,
      const int64_t * const send_prefix,
      int64_t * const recv_idx,
      __half * const recv_value,
      const int64_t * const recv_prefix,
      cudaStream_t stream);




/* CAPI **********************************************************************/

DGL_REGISTER_GLOBAL("cuda.nccl._CAPI_DGLNCCLGetUniqueId")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    *rv = NCCLUniqueIdRef(std::make_shared<NCCLUniqueId>());
});

DGL_REGISTER_GLOBAL("cuda.nccl._CAPI_DGLNCCLCreateComm")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const int size = args[0];
    const int rank = args[1];
    NCCLUniqueIdRef idObj = args[2];

    *rv = NCCLCommunicatorRef(std::make_shared<NCCLCommunicator>(size, rank,
          idObj->Get()));
});

}
}
}
