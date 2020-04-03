/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampling/randomwalks.cc
 * \brief Dispatcher of different DGL random walks by device type
 */

#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>
#include <dgl/array.h>
#include <dgl/sampling/randomwalks.h>
#include <utility>
#include <tuple>
#include <vector>
#include "../../../c_api_common.h"
#include "randomwalks_impl.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {

namespace sampling {

namespace {

void CheckRandomWalkInputs(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob) {
  CHECK_INT(seeds, "seeds");
  CHECK_INT(metapath, "metapath");
  CHECK_NDIM(seeds, 1, "seeds");
  CHECK_NDIM(metapath, 1, "metapath");
  for (uint64_t i = 0; i < prob.size(); ++i) {
    FloatArray p = prob[i];
    CHECK_FLOAT(p, "probability");
    if (p.GetSize() != 0)
      CHECK_NDIM(p, 1, "probability");
  }
}

};  // namespace

std::pair<IdArray, TypeArray> RandomWalk(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob) {
  CheckRandomWalkInputs(hg, seeds, metapath, prob);

  TypeArray vtypes;
  IdArray vids;
  ATEN_XPU_SWITCH(hg->Context().device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(seeds->dtype, IdxType, {
      vtypes = impl::GetNodeTypesFromMetapath<XPU, IdxType>(hg, metapath);
      vids = impl::RandomWalk<XPU, IdxType>(hg, seeds, metapath, prob);
    });
  });

  return std::make_pair(vids, vtypes);
}

std::pair<IdArray, TypeArray> RandomWalkWithRestart(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob) {
  CheckRandomWalkInputs(hg, seeds, metapath, prob);
  CHECK(restart_prob >= 0 && restart_prob < 1) << "restart probability must belong to [0, 1)";

  TypeArray vtypes;
  IdArray vids;
  ATEN_XPU_SWITCH(hg->Context().device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(seeds->dtype, IdxType, {
      vtypes = impl::GetNodeTypesFromMetapath<XPU, IdxType>(hg, metapath);
      vids = impl::RandomWalkWithRestart<XPU, IdxType>(hg, seeds, metapath, prob, restart_prob);
    });
  });

  return std::make_pair(vids, vtypes);
}

std::pair<IdArray, TypeArray> RandomWalkWithStepwiseRestart(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    FloatArray restart_prob) {
  CheckRandomWalkInputs(hg, seeds, metapath, prob);
  // TODO(BarclayII): check the elements of restart probability

  TypeArray vtypes;
  IdArray vids;
  ATEN_XPU_SWITCH(hg->Context().device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(seeds->dtype, IdxType, {
      vtypes = impl::GetNodeTypesFromMetapath<XPU, IdxType>(hg, metapath);
      vids = impl::RandomWalkWithStepwiseRestart<XPU, IdxType>(
          hg, seeds, metapath, prob, restart_prob);
    });
  });

  return std::make_pair(vids, vtypes);
}

};  // namespace sampling

DGL_REGISTER_GLOBAL("sampling.randomwalks._CAPI_DGLSamplingRandomWalk")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    IdArray seeds = args[1];
    TypeArray metapath = args[2];
    List<Value> prob = args[3];

    const auto& prob_vec = ListValueToVector<FloatArray>(prob);

    auto result = sampling::RandomWalk(hg.sptr(), seeds, metapath, prob_vec);
    List<Value> ret;
    ret.push_back(Value(MakeValue(result.first)));
    ret.push_back(Value(MakeValue(result.second)));
    *rv = ret;
  });

DGL_REGISTER_GLOBAL("sampling.randomwalks._CAPI_DGLSamplingRandomWalkWithRestart")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    IdArray seeds = args[1];
    TypeArray metapath = args[2];
    List<Value> prob = args[3];
    double restart_prob = args[4];

    const auto& prob_vec = ListValueToVector<FloatArray>(prob);

    auto result = sampling::RandomWalkWithRestart(
        hg.sptr(), seeds, metapath, prob_vec, restart_prob);
    List<Value> ret;
    ret.push_back(Value(MakeValue(result.first)));
    ret.push_back(Value(MakeValue(result.second)));
    *rv = ret;
  });

DGL_REGISTER_GLOBAL("sampling.randomwalks._CAPI_DGLSamplingRandomWalkWithStepwiseRestart")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    IdArray seeds = args[1];
    TypeArray metapath = args[2];
    List<Value> prob = args[3];
    FloatArray restart_prob = args[4];

    const auto& prob_vec = ListValueToVector<FloatArray>(prob);

    auto result = sampling::RandomWalkWithStepwiseRestart(
        hg.sptr(), seeds, metapath, prob_vec, restart_prob);
    List<Value> ret;
    ret.push_back(Value(MakeValue(result.first)));
    ret.push_back(Value(MakeValue(result.second)));
    *rv = ret;
  });

DGL_REGISTER_GLOBAL("sampling.randomwalks._CAPI_DGLSamplingPackTraces")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    IdArray vids = args[0];
    TypeArray vtypes = args[1];

    IdArray concat_vids, concat_vtypes, lengths, offsets;
    std::tie(concat_vids, lengths, offsets) = Pack(vids, -1);
    std::tie(concat_vtypes, std::ignore) = ConcatSlices(vtypes, lengths);

    List<Value> ret;
    ret.push_back(Value(MakeValue(concat_vids)));
    ret.push_back(Value(MakeValue(concat_vtypes)));
    ret.push_back(Value(MakeValue(lengths)));
    ret.push_back(Value(MakeValue(offsets)));
    *rv = ret;
  });

};  // namespace dgl
