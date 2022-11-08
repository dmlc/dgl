/**
 *  Copyright (c) 2021 by Contributors
 * @file graph/sampling/node2vec.cc
 * @brief Dispatcher of DGL node2vec random walks
 */

#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>

#include "../../../c_api_common.h"
#include "node2vec_impl.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {

namespace sampling {

namespace {

void CheckNode2vecInputs(
    const HeteroGraphPtr hg, const IdArray seeds, const double p,
    const double q, const int64_t walk_length, const FloatArray &prob) {
  CHECK_INT(seeds, "seeds");
  CHECK_NDIM(seeds, 1, "seeds");
  CHECK_FLOAT(prob, "probability");
  CHECK_NDIM(prob, 1, "probability");
}

std::pair<IdArray, IdArray> Node2vec(
    const HeteroGraphPtr hg, const IdArray seeds, const double p,
    const double q, const int64_t walk_length, const FloatArray &prob) {
  CheckNode2vecInputs(hg, seeds, p, q, walk_length, prob);

  std::pair<IdArray, IdArray> result;
  ATEN_XPU_SWITCH(hg->Context().device_type, XPU, "Node2vec", {
    ATEN_ID_TYPE_SWITCH(seeds->dtype, IdxType, {
      result = impl::Node2vec<XPU, IdxType>(hg, seeds, p, q, walk_length, prob);
    });
  });

  return result;
}

DGL_REGISTER_GLOBAL("sampling.randomwalks._CAPI_DGLSamplingNode2vec")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphRef hg = args[0];
      IdArray seeds = args[1];
      double p = args[2];
      double q = args[3];
      int64_t walk_length = args[4];
      FloatArray prob = args[5];

      auto result =
          sampling::Node2vec(hg.sptr(), seeds, p, q, walk_length, prob);

      List<Value> ret;
      ret.push_back(Value(MakeValue(result.first)));
      ret.push_back(Value(MakeValue(result.second)));
      *rv = ret;
    });

}  // namespace

}  // namespace sampling

}  // namespace dgl
