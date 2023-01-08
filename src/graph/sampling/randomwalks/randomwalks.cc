/**
 *  Copyright (c) 2018 by Contributors
 * @file graph/sampling/randomwalks.cc
 * @brief Dispatcher of different DGL random walks by device type
 */

#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/sampling/randomwalks.h>

#include <tuple>
#include <utility>
#include <vector>

#include "../../../c_api_common.h"
#include "randomwalks_impl.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {

namespace sampling {

namespace {

void CheckRandomWalkInputs(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob) {
  CHECK_INT(seeds, "seeds");
  CHECK_INT(metapath, "metapath");
  CHECK_NDIM(seeds, 1, "seeds");
  CHECK_NDIM(metapath, 1, "metapath");
  // (Xin): metapath is copied to GPU in CUDA random walk code
  // CHECK_SAME_CONTEXT(seeds, metapath);

  if (hg->IsPinned()) {
    CHECK_EQ(seeds->ctx.device_type, kDGLCUDA)
        << "Expected seeds (" << seeds->ctx << ")"
        << " to be on the GPU when the graph is pinned.";
  } else if (hg->Context() != seeds->ctx) {
    LOG(FATAL) << "Expected seeds (" << seeds->ctx << ")"
               << " to have the same "
               << "context as graph (" << hg->Context() << ").";
  }
  for (uint64_t i = 0; i < prob.size(); ++i) {
    FloatArray p = prob[i];
    CHECK_EQ(hg->Context(), p->ctx)
        << "Expected prob (" << p->ctx << ")"
        << " to have the same "
        << "context as graph (" << hg->Context() << ").";
    CHECK_FLOAT(p, "probability");
    if (p.GetSize() != 0) {
      CHECK_EQ(hg->IsPinned(), p.IsPinned())
          << "The prob array should have the same pinning status as the graph";
      CHECK_NDIM(p, 1, "probability");
    }
  }
}

};  // namespace

std::tuple<IdArray, IdArray, TypeArray> RandomWalk(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob) {
  CheckRandomWalkInputs(hg, seeds, metapath, prob);

  TypeArray vtypes;
  std::pair<IdArray, IdArray> result;
  ATEN_XPU_SWITCH_CUDA(seeds->ctx.device_type, XPU, "RandomWalk", {
    ATEN_ID_TYPE_SWITCH(seeds->dtype, IdxType, {
      vtypes = impl::GetNodeTypesFromMetapath<XPU, IdxType>(hg, metapath);
      result = impl::RandomWalk<XPU, IdxType>(hg, seeds, metapath, prob);
    });
  });

  return std::make_tuple(result.first, result.second, vtypes);
}

std::tuple<IdArray, IdArray, TypeArray> RandomWalkWithRestart(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, double restart_prob) {
  CheckRandomWalkInputs(hg, seeds, metapath, prob);
  CHECK(restart_prob >= 0 && restart_prob < 1)
      << "restart probability must belong to [0, 1)";

  TypeArray vtypes;
  std::pair<IdArray, IdArray> result;
  ATEN_XPU_SWITCH_CUDA(seeds->ctx.device_type, XPU, "RandomWalkWithRestart", {
    ATEN_ID_TYPE_SWITCH(seeds->dtype, IdxType, {
      vtypes = impl::GetNodeTypesFromMetapath<XPU, IdxType>(hg, metapath);
      result = impl::RandomWalkWithRestart<XPU, IdxType>(
          hg, seeds, metapath, prob, restart_prob);
    });
  });

  return std::make_tuple(result.first, result.second, vtypes);
}

std::tuple<IdArray, IdArray, TypeArray> RandomWalkWithStepwiseRestart(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, FloatArray restart_prob) {
  CheckRandomWalkInputs(hg, seeds, metapath, prob);
  // TODO(BarclayII): check the elements of restart probability

  TypeArray vtypes;
  std::pair<IdArray, IdArray> result;
  ATEN_XPU_SWITCH_CUDA(
      seeds->ctx.device_type, XPU, "RandomWalkWithStepwiseRestart", {
        ATEN_ID_TYPE_SWITCH(seeds->dtype, IdxType, {
          vtypes = impl::GetNodeTypesFromMetapath<XPU, IdxType>(hg, metapath);
          result = impl::RandomWalkWithStepwiseRestart<XPU, IdxType>(
              hg, seeds, metapath, prob, restart_prob);
        });
      });

  return std::make_tuple(result.first, result.second, vtypes);
}

std::tuple<IdArray, IdArray, IdArray> SelectPinSageNeighbors(
    const IdArray src, const IdArray dst, const int64_t num_samples_per_node,
    const int64_t k) {
  assert(
      (src->ndim == 1) && (dst->ndim == 1) &&
      (src->shape[0] % num_samples_per_node == 0) &&
      (src->shape[0] == dst->shape[0]));
  std::tuple<IdArray, IdArray, IdArray> result;

  ATEN_XPU_SWITCH_CUDA((src->ctx).device_type, XPU, "SelectPinSageNeighbors", {
    ATEN_ID_TYPE_SWITCH(src->dtype, IdxType, {
      result = impl::SelectPinSageNeighbors<XPU, IdxType>(
          src, dst, num_samples_per_node, k);
    });
  });

  return result;
}

};  // namespace sampling

DGL_REGISTER_GLOBAL("sampling.randomwalks._CAPI_DGLSamplingRandomWalk")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphRef hg = args[0];
      IdArray seeds = args[1];
      TypeArray metapath = args[2];
      List<Value> prob = args[3];

      const auto &prob_vec = ListValueToVector<FloatArray>(prob);

      auto result = sampling::RandomWalk(hg.sptr(), seeds, metapath, prob_vec);
      List<Value> ret;
      ret.push_back(Value(MakeValue(std::get<0>(result))));
      ret.push_back(Value(MakeValue(std::get<1>(result))));
      ret.push_back(Value(MakeValue(std::get<2>(result))));
      *rv = ret;
    });

DGL_REGISTER_GLOBAL("sampling.pinsage._CAPI_DGLSamplingSelectPinSageNeighbors")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      IdArray src = args[0];
      IdArray dst = args[1];
      int64_t num_travelsals = static_cast<int64_t>(args[2]);
      int64_t k = static_cast<int64_t>(args[3]);

      auto result =
          sampling::SelectPinSageNeighbors(src, dst, num_travelsals, k);

      List<Value> ret;
      ret.push_back(Value(MakeValue(std::get<0>(result))));
      ret.push_back(Value(MakeValue(std::get<1>(result))));
      ret.push_back(Value(MakeValue(std::get<2>(result))));
      *rv = ret;
    });

DGL_REGISTER_GLOBAL(
    "sampling.randomwalks._CAPI_DGLSamplingRandomWalkWithRestart")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphRef hg = args[0];
      IdArray seeds = args[1];
      TypeArray metapath = args[2];
      List<Value> prob = args[3];
      double restart_prob = args[4];

      const auto &prob_vec = ListValueToVector<FloatArray>(prob);

      auto result = sampling::RandomWalkWithRestart(
          hg.sptr(), seeds, metapath, prob_vec, restart_prob);
      List<Value> ret;
      ret.push_back(Value(MakeValue(std::get<0>(result))));
      ret.push_back(Value(MakeValue(std::get<1>(result))));
      ret.push_back(Value(MakeValue(std::get<2>(result))));
      *rv = ret;
    });

DGL_REGISTER_GLOBAL(
    "sampling.randomwalks._CAPI_DGLSamplingRandomWalkWithStepwiseRestart")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphRef hg = args[0];
      IdArray seeds = args[1];
      TypeArray metapath = args[2];
      List<Value> prob = args[3];
      FloatArray restart_prob = args[4];

      const auto &prob_vec = ListValueToVector<FloatArray>(prob);

      auto result = sampling::RandomWalkWithStepwiseRestart(
          hg.sptr(), seeds, metapath, prob_vec, restart_prob);
      List<Value> ret;
      ret.push_back(Value(MakeValue(std::get<0>(result))));
      ret.push_back(Value(MakeValue(std::get<1>(result))));
      ret.push_back(Value(MakeValue(std::get<2>(result))));
      *rv = ret;
    });

DGL_REGISTER_GLOBAL("sampling.randomwalks._CAPI_DGLSamplingPackTraces")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
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
