/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampler/generic_randomwalk_cpu.h
 * \brief DGL sampler - templated implementation definition of random walks on CPU
 */

#ifndef DGL_GRAPH_SAMPLING_RANDOMWALKS_RANDOMWALKS_CPU_H_
#define DGL_GRAPH_SAMPLING_RANDOMWALKS_RANDOMWALKS_CPU_H_

#include <dgl/base_heterograph.h>
#include <dgl/array.h>
#include "randomwalks_impl.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

namespace {

/*!
 * \brief Generic Random Walk.
 * \param seeds A 1D array of seed nodes, with the type the source type of the first
 *        edge type in the metapath.
 * \param max_num_steps The maximum number of steps of a random walk path.
 * \param step The random walk step function with type \c StepFunc.
 * \return A 2D array of shape (len(seeds), max_num_steps + 1) with node IDs.
 * \note The graph itself should be bounded in the closure of \c step.
 */
template<DLDeviceType XPU, typename IdxType>
IdArray GenericRandomWalk(
    const IdArray seeds,
    int64_t max_num_steps,
    StepFunc<IdxType> step) {
  int64_t num_seeds = seeds->shape[0];
  int64_t trace_length = max_num_steps + 1;
  IdArray traces = IdArray::Empty({num_seeds, trace_length}, seeds->dtype, seeds->ctx);

  const IdxType *seed_data = static_cast<IdxType *>(seeds->data);
  IdxType *traces_data = static_cast<IdxType *>(traces->data);

#pragma omp parallel for
  for (int64_t seed_id = 0; seed_id < num_seeds; ++seed_id) {
    int64_t i;
    dgl_id_t curr = seed_data[seed_id];
    traces_data[seed_id * trace_length] = curr;

    for (i = 0; i < max_num_steps; ++i) {
      const auto &succ = step(traces_data + seed_id * max_num_steps, curr, i);
      traces_data[seed_id * trace_length + i + 1] = curr = succ.first;
      if (succ.second)
        break;
    }

    for (; i < max_num_steps; ++i)
      traces_data[seed_id * trace_length + i + 1] = -1;
  }

  return traces;
}

};  // namespace

};  // namespace impl

};  // namespace sampling

};  // namespace dgl

#endif  // DGL_GRAPH_SAMPLING_RANDOMWALKS_RANDOMWALKS_CPU_H_
