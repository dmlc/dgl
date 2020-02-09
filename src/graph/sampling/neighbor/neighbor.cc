/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/sampling/neighbor.cc
 * \brief Definition of neighborhood-based sampler APIs.
 */

#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>
#include <dgl/array.h>
#include <dgl/sampling/neighbor.h>
#include "../../../c_api_common.h"
#include "./neighbor_impl.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {
namespace sampling {

HeteroGraphPtr SampleNeighbors(
    const HeteroGraphPtr hg,
    const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& fanouts,
    EdgeDir dir,
    const std::vector<FloatArray>& prob,
    bool replace) {
  // sanity check
  CHECK_EQ(nodes.size(), hg->NumVertexTypes())
    << "Number of node ID tensors must match the number of node types.";
  CHECK_EQ(fanouts.size(), hg->NumEdgeTypes())
    << "Number of fanout values must match the number of edge types.";
  CHECK_EQ(prob.size(), hg->NumEdgeTypes())
    << "Number of probability tensors must match the number of edge types.";

  HeteroGraphPtr ret;
  ATEN_XPU_SWITCH(hg->Context().device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(hg->DataType(), IdxType, {
      ret = impl::SampleNeighbors<XPU, IdxType>(
        hg, nodes, fanouts, dir, prob, replace);
    });
  });
  return ret;
}

HeteroGraphPtr SampleNeighborsTopk(
    const HeteroGraphPtr hg,
    const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& k,
    EdgeDir dir,
    const std::vector<FloatArray>& weight) {
  // sanity check
  CHECK_EQ(nodes.size(), hg->NumVertexTypes())
    << "Number of node ID tensors must match the number of node types.";
  CHECK_EQ(k.size(), hg->NumEdgeTypes())
    << "Number of k values must match the number of edge types.";
  CHECK_EQ(weight.size(), hg->NumEdgeTypes())
    << "Number of weight tensors must match the number of edge types.";

  HeteroGraphPtr ret;
  ATEN_XPU_SWITCH(hg->Context().device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(hg->DataType(), IdxType, {
      ret = impl::SampleNeighborsTopk<XPU, IdxType>(
        hg, nodes, k, dir, weight);
    });
  });
  return ret;
}

DGL_REGISTER_GLOBAL("sampling.neighbor._CAPI_DGLSampleNeighbors")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    const auto& nodes = ListValueToVector<IdArray>(args[1]);
    const auto& fanouts = ListValueToVector<int64_t>(args[2]);
    const std::string dir_str = args[3];
    const auto& prob = ListValueToVector<FloatArray>(args[4]);
    const bool replace = args[5];

    CHECK(dir_str == "in" || dir_str == "out")
      << "Invalid edge direction. Must be \"in\" or \"out\".";
    EdgeDir dir = (dir_str == "in")? EdgeDir::kIn : EdgeDir::kOut;

    HeteroGraphPtr ret = sampling::SampleNeighbors(
        hg.sptr(), nodes, fanouts, dir, prob, replace);

    *rv = HeteroGraphRef(ret);
  });

DGL_REGISTER_GLOBAL("sampling.neighbor._CAPI_DGLSampleNeighborsTopk")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    const auto& nodes = ListValueToVector<IdArray>(args[1]);
    const auto& k = ListValueToVector<int64_t>(args[2]);
    const std::string dir_str = args[3];
    const auto& weight = ListValueToVector<FloatArray>(args[4]);

  CHECK(dir_str == "in" || dir_str == "out")
    << "Invalid edge direction. Must be \"in\" or \"out\".";
    EdgeDir dir = (dir_str == "in")? EdgeDir::kIn : EdgeDir::kOut;

    HeteroGraphPtr ret = sampling::SampleNeighborsTopk(
        hg.sptr(), nodes, k, dir, weight);

    *rv = HeteroGraphRef(ret);
  });

}  // namespace sampling
}  // namespace dgl
