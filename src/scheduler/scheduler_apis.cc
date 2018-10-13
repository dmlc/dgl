#include <dgl/c_api_common.h>
#include <dgl/scheduler.h>

namespace dgl {

TVM_REGISTER_GLOBAL("scheduler._CAPI_DGLDegreeBucket")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    const IdArray vids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[0]));
    *rv = ConvertNDArrayVectorToPackedFunc(scheduler::DegreeBucketing(vids));
  });

} // namespace dgl
