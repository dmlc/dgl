/*!
 *  Copyright (c) 2018 by Contributors
 * \file scheduler/scheduler_apis.cc
 * \brief DGL scheduler APIs
 */
#include <dgl/graph.h>
#include <dgl/scheduler.h>
#include "../c_api_common.h"

using tvm::runtime::TVMArgs;
using tvm::runtime::TVMRetValue;
using tvm::runtime::NDArray;

namespace dgl {

TVM_REGISTER_GLOBAL("scheduler._CAPI_DGLDegreeBucketing")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    const IdArray vids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[0]));
    *rv = ConvertNDArrayVectorToPackedFunc(sched::DegreeBucketing(vids));
  });

TVM_REGISTER_GLOBAL("scheduler._CAPI_DGLDegreeBucketingFromGraph")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const auto& edges = gptr->Edges(false);
    *rv = ConvertNDArrayVectorToPackedFunc(sched::DegreeBucketing(edges.dst));
  });

}  // namespace dgl
