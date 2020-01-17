/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampler/randomwalks_cpu.cc
 * \brief DGL sampler - CPU implementation of random walks with OpenMP
 */

#include <dgl/runtime/container.h>
#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <utility>
#include "randomwalks_impl.h"

namespace dgl {

namespace sampling {

namespace impl {

template<DLDeviceType XPU>
std::pair<IdArray, TypeArray> RandomWalkImpl(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const List<Value> &prob) {
  int64_t num_seeds = seeds->shape[0];
  int64_t trace_length = metapath->shape[0] + 1;

  IdArray vids = IdArray::Empty(
    {num_seeds, trace_length}, seeds->dtype, seeds->ctx);
  TypeArray vtypes = TypeArray::Empty(
    {num_seeds, trace_length}, metapath->dtype, metapath->ctx);

#pragma omp parallel for
  for (int64_t i = 0; i < num_seeds; ++i) {
    IdArray vids_i = vids.CreateView(
      {trace_length}, vids->dtype, i * trace_length * vids->dtype.bits / 8);
    TypeArray vtypes_i = vtypes.CreateView(
      {trace_length}, vtypes->dtype, i * trace_length * vtypes->dtype.bits / 8);

    RandomWalkOneSeed(
        hg, IndexSelect<int64_t>(seeds, i), metapath, prob, vids_i, vtypes_i, 0.);
  }

  return std::make_pair(vids, vtypes);
}

template
std::pair<IdArray, TypeArray> RandomWalkImpl<kDLCPU>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const List<Value> &prob);

};  // namespace impl

};  // namespace sampling

};  // namespace dgl
