/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampler/randomwalks.cc
 * \brief Dispatcher of different DGL random walks by device type
 */

#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>
#include <dgl/array.h>
#include <utility>
#include "../../c_api_common.h"
#include "randomwalks.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {

namespace sampling {

namespace {

/*!
 * \brief Check if dsttype of metapath[i-1] == srctype of metapath[i].
 *
 * \return 0 if the metapath is consistent, or the index where the source and destination type
 * does not match.
 */
};  // namespace

std::pair<IdArray, TypeArray> RandomWalk(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob) {
  EXPECT_INT(seeds, "seeds");
  EXPECT_INT(metapath, "metapath");
  EXPECT_NDIM(seeds, 1, "seeds");
  EXPECT_NDIM(metapath, 1, "metapath");
  for (uint64_t i = 0; i < prob.size(); ++i) {
    FloatArray p = prob[i];
    EXPECT_FLOAT(p, "probability");
    if (!IsEmpty(p)) {
      EXPECT_NDIM(p, 1, "probability");
    }
  }

  TypeArray vtypes;
  IdArray vids;
  ATEN_XPU_SWITCH(hg->Context().device_type, XPU, {
    vtypes = impl::GetNodeTypesFromMetapath<XPU>(hg, metapath);
    vids = impl::RandomWalk<XPU>(hg, seeds, metapath, prob);
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

    std::vector<FloatArray> prob_vec;
    prob_vec.reserve(prob.size());
    for (Value val : prob)
      prob_vec.push_back(val->data);

    auto result = sampling::RandomWalk(hg.sptr(), seeds, metapath, prob_vec);
    List<Value> ret;
    ret.push_back(Value(MakeValue(result.first)));
    ret.push_back(Value(MakeValue(result.second)));
    *rv = ret;
  });

};  // namespace dgl
