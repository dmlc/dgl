/*!
 *  Copyright (c) 2021 by Contributors
 * \file nccl_api.h
 * \brief Wrapper around NCCL routines. 
 */

#include "nccl.h"

#include <dgl/runtime/object.h>

namespace dgl {
namespace runtime {
namespace cuda {

class NCCLUniqueId : public runtime::Object {
 public:
  NCCLUniqueId();

  static constexpr const char* _type_key = "cuda.NCCLUniqueId";
  DGL_DECLARE_OBJECT_TYPE_INFO(NCCLUniqueId, Object);

  ncclUniqueId Get() const;

 private:
  ncclUniqueId id_;
};

DGL_DEFINE_OBJECT_REF(NCCLUniqueIdRef, NCCLUniqueId);

class NCCLCommunicator : public runtime::Object {
 public:
  NCCLCommunicator(
      int size,
      int rank,
      ncclUniqueId id);

  ~NCCLCommunicator();

  // disable copying
  NCCLCommunicator(const NCCLCommunicator& other) = delete;
  NCCLCommunicator& operator=(
      const NCCLCommunicator& other);

  ncclComm_t Get();

  static constexpr const char* _type_key = "cuda.NCCLCommunicator";
  DGL_DECLARE_OBJECT_TYPE_INFO(NCCLCommunicator, Object);

 private:
  ncclComm_t comm_;
};

DGL_DEFINE_OBJECT_REF(NCCLCommunicatorRef, NCCLCommunicator);

}
}
}
