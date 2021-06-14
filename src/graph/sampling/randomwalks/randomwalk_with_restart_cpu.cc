/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampling/randomwalk_with_restart_cpu.cc
 * \brief DGL sampler - CPU implementation of metapath-based random walk with restart with OpenMP
 */

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/random.h>
#include <utility>
#include <vector>
#include "randomwalks_impl.h"
#include "randomwalks_cpu.h"
#include "metapath_randomwalk.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

template<DLDeviceType XPU, typename IdxType>
std::pair<IdArray, IdArray> RandomWalkWithRestart(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob) {
  TerminatePredicate<IdxType> terminate =
    [restart_prob] (IdxType *data, dgl_id_t curr, int64_t len) {
      return RandomEngine::ThreadLocal()->Uniform<double>() < restart_prob;
    };
  return MetapathBasedRandomWalk<XPU, IdxType>(hg, seeds, metapath, prob, terminate);
}

template
std::pair<IdArray, IdArray> RandomWalkWithRestart<kDLCPU, int32_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob);
template
std::pair<IdArray, IdArray> RandomWalkWithRestart<kDLCPU, int64_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob);

template<DLDeviceType XPU, typename IdxType>
std::pair<IdArray, IdArray> RandomWalkWithStepwiseRestart(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    FloatArray restart_prob) {
  std::pair<IdArray, IdArray> result;

  ATEN_FLOAT_TYPE_SWITCH(restart_prob->dtype, DType, "restart probability", {
    DType *restart_prob_data = static_cast<DType *>(restart_prob->data);
    TerminatePredicate<IdxType> terminate =
      [restart_prob_data] (IdxType *data, dgl_id_t curr, int64_t len) {
        return RandomEngine::ThreadLocal()->Uniform<DType>() < restart_prob_data[len];
      };
    result = MetapathBasedRandomWalk<XPU, IdxType>(hg, seeds, metapath, prob, terminate);
  });

  return result;
}

template
std::pair<IdArray, IdArray> RandomWalkWithStepwiseRestart<kDLCPU, int32_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    FloatArray restart_prob);
template
std::pair<IdArray, IdArray> RandomWalkWithStepwiseRestart<kDLCPU, int64_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    FloatArray restart_prob);

};  // namespace impl

};  // namespace sampling

};  // namespace dgl
