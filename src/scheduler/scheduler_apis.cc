/*!
 *  Copyright (c) 2018 by Contributors
 * \file scheduler/scheduler_apis.cc
 * \brief DGL scheduler APIs
 */
#include <dgl/graph.h>
#include <dgl/scheduler.h>
#include "../c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLRetValue;
using dgl::runtime::NDArray;

namespace dgl {

DGL_REGISTER_GLOBAL("runtime.degree_bucketing._CAPI_DGLDegreeBucketing")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const IdArray msg_ids = args[0];
    const IdArray vids = args[1];
    const IdArray nids = args[2];
    *rv = ConvertNDArrayVectorToPackedFunc(sched::DegreeBucketing(msg_ids, vids, nids));
  });

DGL_REGISTER_GLOBAL("runtime.degree_bucketing._CAPI_DGLGroupEdgeByNodeDegree")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const IdArray uids = args[0];
    const IdArray vids = args[1];
    const IdArray eids = args[2];
    *rv = ConvertNDArrayVectorToPackedFunc(
            sched::GroupEdgeByNodeDegree(uids, vids, eids));
  });

}  // namespace dgl
