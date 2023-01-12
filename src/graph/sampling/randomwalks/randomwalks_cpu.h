/**
 *  Copyright (c) 2018 by Contributors
 * @file graph/sampler/generic_randomwalk_cpu.h
 * @brief DGL sampler - templated implementation definition of random walks on
 * CPU
 */

#ifndef DGL_GRAPH_SAMPLING_RANDOMWALKS_RANDOMWALKS_CPU_H_
#define DGL_GRAPH_SAMPLING_RANDOMWALKS_RANDOMWALKS_CPU_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/runtime/parallel_for.h>

#include <tuple>
#include <utility>

#include "randomwalks_impl.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

namespace {

/**
 * @brief Generic Random Walk.
 * @param seeds A 1D array of seed nodes, with the type the source type of the
 * first edge type in the metapath.
 * @param max_num_steps The maximum number of steps of a random walk path.
 * @param step The random walk step function with type \c StepFunc.
 * @param max_nodes Throws an error if one of the values in \c seeds exceeds
 * this argument.
 * @return A 2D array of shape (len(seeds), max_num_steps + 1) with node IDs.
 * @note The graph itself should be bounded in the closure of \c step.
 */
template <DGLDeviceType XPU, typename IdxType>
std::pair<IdArray, IdArray> GenericRandomWalk(
    const IdArray seeds, int64_t max_num_steps, StepFunc<IdxType> step,
    int64_t max_nodes) {
  int64_t num_seeds = seeds->shape[0];
  int64_t trace_length = max_num_steps + 1;
  IdArray traces =
      IdArray::Empty({num_seeds, trace_length}, seeds->dtype, seeds->ctx);
  IdArray eids =
      IdArray::Empty({num_seeds, max_num_steps}, seeds->dtype, seeds->ctx);

  const IdxType *seed_data = seeds.Ptr<IdxType>();
  IdxType *traces_data = traces.Ptr<IdxType>();
  IdxType *eids_data = eids.Ptr<IdxType>();

  runtime::parallel_for(0, num_seeds, [&](size_t seed_begin, size_t seed_end) {
    for (auto seed_id = seed_begin; seed_id < seed_end; seed_id++) {
      int64_t i;
      dgl_id_t curr = seed_data[seed_id];
      traces_data[seed_id * trace_length] = curr;

      CHECK_LT(curr, max_nodes)
          << "Seed node ID exceeds the maximum number of nodes.";

      for (i = 0; i < max_num_steps; ++i) {
        const auto &succ = step(traces_data + seed_id * trace_length, curr, i);
        traces_data[seed_id * trace_length + i + 1] = curr = std::get<0>(succ);
        eids_data[seed_id * max_num_steps + i] = std::get<1>(succ);
        if (std::get<2>(succ)) break;
      }

      for (; i < max_num_steps; ++i) {
        traces_data[seed_id * trace_length + i + 1] = -1;
        eids_data[seed_id * max_num_steps + i] = -1;
      }
    }
  });

  return std::make_pair(traces, eids);
}

};  // namespace

};  // namespace impl

};  // namespace sampling

};  // namespace dgl

#endif  // DGL_GRAPH_SAMPLING_RANDOMWALKS_RANDOMWALKS_CPU_H_
