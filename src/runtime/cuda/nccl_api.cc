/*!
 *  Copyright (c) 2021 by Contributors
 * \file nccl_api.cc
 * \brief Implementation of wrapper around NCCL routines. 
 */

#include "nccl_api.h"

#include <dgl/packed_func_ext.h>
#include <dgl/runtime/registry.h>


namespace dgl {
namespace runtime {
namespace cuda {

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
