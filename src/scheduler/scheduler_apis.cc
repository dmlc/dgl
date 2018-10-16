#include <dgl/c_api_common.h>
#include <dgl/graph.h>
#include <dgl/scheduler.h>

namespace dgl {

// Graph handler type
typedef void* GraphHandle;

TVM_REGISTER_GLOBAL("scheduler._CAPI_DGLDegreeBucketing")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    const IdArray vids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[0]));
    *rv = ConvertNDArrayVectorToPackedFunc(scheduler::DegreeBucketing(vids));
  });

TVM_REGISTER_GLOBAL("scheduler._CAPI_DGLDegreeBucketingFromGraph")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    auto edges = gptr->Edges(false);
    *rv = ConvertNDArrayVectorToPackedFunc(scheduler::DegreeBucketing(edges.dst));
  });

} // namespace dgl
