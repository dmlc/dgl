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
#include <dgl/profile.h>

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {

namespace sampling {

namespace {

/*!
 * \brief Check if dsttype of metapath[i-1] == srctype of metapath[i].
 *
 * \return -1 if the metapath is consistent, or the index where the source and destination type
 * does not match.
 */
int64_t IsInconsistent(const HeteroGraphPtr hg, const TypeArray metapath) {
  uint64_t num_etypes = metapath->shape[0];
  dgl_type_t etype = IndexSelect<int64_t>(metapath, 0);
  dgl_type_t srctype = hg->GetEndpointTypes(etype).first;
  dgl_type_t curr_type = srctype;
  for (uint64_t i = 0; i < num_etypes; ++i) {
    etype = IndexSelect<int64_t>(metapath, i);
    auto src_dst_type = hg->GetEndpointTypes(etype);
    dgl_type_t srctype = src_dst_type.first;
    dgl_type_t dsttype = src_dst_type.second;

    if (srctype != curr_type)
      return i;
    curr_type = dsttype;
  }
  return -1;
}

};  // namespace

std::pair<IdArray, TypeArray> RandomWalk(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const List<Value> &prob) {
  TIMEIT_ALLOC(T, iter, 10);

  TIMEIT(T, 0, {
    EXPECT_INT(seeds, "seeds");
    EXPECT_INT(metapath, "metapath");
    EXPECT_NDIM(seeds, 1, "seeds");
    EXPECT_NDIM(metapath, 1, "metapath");
    for (uint64_t i = 0; i < prob.size(); ++i) {
      FloatArray p = prob[i]->data;
      EXPECT_FLOAT(p, "probability");
      if (!IsEmpty(p)) {
        EXPECT_NDIM(p, 1, "probability");
      }
    }
  });

  TIMEIT(T, 1, {
    auto inconsistent_idx = IsInconsistent(hg, metapath);
    CHECK(inconsistent_idx == -1) << "metapath inconsistent between position " <<
      (inconsistent_idx - 1) << " and " << inconsistent_idx;
  });

  std::pair<IdArray, TypeArray> result;
  TIMEIT(T, 2, {
    ATEN_XPU_SWITCH(hg->Context().device_type, XPU, {
      result = impl::RandomWalkImpl<XPU>(hg, seeds, metapath, prob);
    });
  });

  TIMEIT_CHECK(T, iter, 3, 10, "");

  return result;
}

};  // namespace sampling

DGL_REGISTER_GLOBAL("sampling.randomwalks._CAPI_DGLSamplingRandomWalk")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    IdArray seeds = args[1];
    TypeArray metapath = args[2];
    List<Value> prob = args[3];

    auto result = sampling::RandomWalk(hg.sptr(), seeds, metapath, prob);
    List<Value> ret;
    ret.push_back(Value(MakeValue(result.first)));
    ret.push_back(Value(MakeValue(result.second)));
    *rv = ret;
  });

};  // namespace dgl
