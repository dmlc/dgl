/**
 *  Copyright (c) 2021 by Contributors
 * @file graph/sampling/node2vec_cpu.cc
 * @brief DGL sampler - CPU implementation of node2vec random walk with OpenMP
 */

#include <dgl/array.h>
#include <dgl/base_heterograph.h>

#include <utility>

#include "node2vec_randomwalk.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

template <DGLDeviceType XPU, typename IdxType>
std::pair<IdArray, IdArray> Node2vec(
    const HeteroGraphPtr hg, const IdArray seeds, const double p,
    const double q, const int64_t walk_length, const FloatArray &prob) {
  TerminatePredicate<IdxType> terminate = [](IdxType *data, dgl_id_t curr,
                                             int64_t len) { return false; };

  return Node2vecRandomWalk<XPU, IdxType>(
      hg, seeds, p, q, walk_length, prob, terminate);
}

template std::pair<IdArray, IdArray> Node2vec<kDGLCPU, int32_t>(
    const HeteroGraphPtr hg, const IdArray seeds, const double p,
    const double q, const int64_t walk_length, const FloatArray &prob);
template std::pair<IdArray, IdArray> Node2vec<kDGLCPU, int64_t>(
    const HeteroGraphPtr hg, const IdArray seeds, const double p,
    const double q, const int64_t walk_length, const FloatArray &prob);

};  // namespace impl

};  // namespace sampling

};  // namespace dgl
