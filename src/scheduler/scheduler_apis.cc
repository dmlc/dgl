/**
 *  Copyright (c) 2018 by Contributors
 * @file scheduler/scheduler_apis.cc
 * @brief DGL scheduler APIs
 */
#include <dgl/array.h>
#include <dgl/graph.h>
#include <dgl/scheduler.h>

#include "../array/cpu/array_utils.h"
#include "../c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLRetValue;
using dgl::runtime::NDArray;

namespace dgl {

DGL_REGISTER_GLOBAL(
    "_deprecate.runtime.degree_bucketing._CAPI_DGLDegreeBucketing")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const IdArray msg_ids = args[0];
      const IdArray vids = args[1];
      const IdArray nids = args[2];
      CHECK_SAME_DTYPE(msg_ids, vids);
      CHECK_SAME_DTYPE(msg_ids, nids);
      ATEN_ID_TYPE_SWITCH(msg_ids->dtype, IdType, {
        *rv = ConvertNDArrayVectorToPackedFunc(
            sched::DegreeBucketing<IdType>(msg_ids, vids, nids));
      });
    });

DGL_REGISTER_GLOBAL(
    "_deprecate.runtime.degree_bucketing._CAPI_DGLGroupEdgeByNodeDegree")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const IdArray uids = args[0];
      const IdArray vids = args[1];
      const IdArray eids = args[2];
      CHECK_SAME_DTYPE(uids, vids);
      CHECK_SAME_DTYPE(uids, eids);
      ATEN_ID_TYPE_SWITCH(uids->dtype, IdType, {
        *rv = ConvertNDArrayVectorToPackedFunc(
            sched::GroupEdgeByNodeDegree<IdType>(uids, vids, eids));
      });
    });

}  // namespace dgl
