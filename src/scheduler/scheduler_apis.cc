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

TVM_REGISTER_GLOBAL("runtime.scheduler._CAPI_DGLDegreeBucketingForEdges")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    const IdArray vids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[0]));
    // XXX: better way to do arange?
    int64_t n_msgs = vids->shape[0];
    IdArray msg_ids = IdArray::Empty({n_msgs}, vids->dtype, vids->ctx);
    int64_t* mid_data = static_cast<int64_t*>(msg_ids->data);
    for (int64_t i = 0; i < n_msgs; ++i) {
        mid_data[i] = i;
    }
    *rv = ConvertNDArrayVectorToPackedFunc(sched::DegreeBucketing(msg_ids, vids, vids));
  });

TVM_REGISTER_GLOBAL("runtime.scheduler._CAPI_DGLDegreeBucketingForRecvNodes")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const IdArray vids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    const auto& edges = gptr->InEdges(vids);
    *rv = ConvertNDArrayVectorToPackedFunc(sched::DegreeBucketing(edges.id, edges.dst, vids));
  });

TVM_REGISTER_GLOBAL("runtime.scheduler._CAPI_DGLDegreeBucketingForFullGraph")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const auto& edges = gptr->Edges(false);
    int64_t n_vertices = gptr->NumVertices();
    IdArray nids = IdArray::Empty({n_vertices}, edges.dst->dtype, edges.dst->ctx);
    int64_t* nid_data = static_cast<int64_t*>(nids->data);
    for (int64_t i = 0; i < n_vertices; ++i) {
        nid_data[i] = i;
    }
    *rv = ConvertNDArrayVectorToPackedFunc(sched::DegreeBucketing(edges.id, edges.dst, nids));
  });
}  // namespace dgl
