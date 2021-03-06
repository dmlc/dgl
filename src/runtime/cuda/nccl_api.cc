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
  comm_()
{
  auto r = ncclCommInitRank(&comm_, size, id, rank);
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

    CHECK_GE(rank, 0);
    CHECK_LT(rank, size);

    *rv = NCCLCommunicatorRef(std::make_shared<NCCLCommunicator>(size, rank,
          idObj->Get()));
});

}
}
}
