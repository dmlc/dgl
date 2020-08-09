/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampling/randomwalk_cpu.cc
 * \brief DGL sampler - CPU implementation of metapath-based random walk with OpenMP
 */

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
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
IdArray RandomWalk(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob) {
  TerminatePredicate<IdxType> terminate =
    [] (IdxType *data, dgl_id_t curr, int64_t len) {
      return false;
    };

  return MetapathBasedRandomWalk<XPU, IdxType>(hg, seeds, metapath, prob, terminate);
}

template
IdArray RandomWalk<kDLCPU, int32_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob);
template
IdArray RandomWalk<kDLCPU, int64_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob);

};  // namespace impl

};  // namespace sampling

};  // namespace dgl
