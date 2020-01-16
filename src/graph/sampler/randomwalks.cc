/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampler/randomwalks.cc
 * \brief Dispatcher of different DGL random walks by device type
 */

#include <dgl/runtime/container.h>
#include <dgl/array.h>
#include <utility>
#include "../../c_api_common.h"
#include "randomwalks.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {

namespace sampling {

std::pair<IdArray, TypeArray> RandomWalk(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray etypes,
    const List<Value> &prob) {
  EXPECT_INT(seeds, "seeds");
  EXPECT_INT(etypes, "etypes");
  EXPECT_NDIM(seeds, 1, "seeds");
  EXPECT_NDIM(etypes, 1, "etypes");
  for (int i = 0; i < prob.size(); ++i) {
    FloatArray p = prob[i]->data;
    EXPECT_FLOAT(p, "probability");
    if (p->ndim != 0) {
      EXPECT_NDIM(p, 1, "probability");
    }
  }

  std::pair<IdArray, TypeArray> result;
  ATEN_XPU_SWITCH(hg->Context().device_type, XPU, {
    result = impl::RandomWalkImpl<XPU>(hg, seeds, etypes, prob);
  });

  return result;
}

};  // namespace sampling

DGL_REGISTER_GLOBAL("sampler.randomwalks._CAPI_DGLRandomWalk")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    IdArray seeds = args[1];
    TypeArray etypes = args[2];
    List<Value> prob = args[3];

    auto result = sampling::RandomWalk(hg.sptr(), seeds, etypes, prob);
    List<Value> ret;
    ret.push_back(Value(MakeValue(result.first)));
    ret.push_back(Value(MakeValue(result.second)));
    *rv = ret;
  });

};  // namespace dgl
