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
#include "randomwalks.h"
#include "randomwalks_cpu.h"
#include "metapath_randomwalk.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

template
IdArray RandomWalkWithRestart<kDLCPU, int32_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob) {
  TerminatePredicate terminate =
    [prob] (void *data, dgl_id_t curr, int64_t len) {
      return RandomEngine::ThreadLocal()->Uniform<double>() < restart_prob;
    }
  return RandomWalk(hg, seeds, metapath, prob, terminate);
}

template
IdArray RandomWalkWithRestart<kDLCPU, int32_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob);
template
IdArray RandomWalkWithRestart<kDLCPU, int64_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob);

};  // namespace impl

};  // namespace sampling

};  // namespace dgl
