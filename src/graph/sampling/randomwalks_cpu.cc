/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampler/randomwalks_cpu.cc
 * \brief DGL sampler - CPU implementation of random walks with OpenMP
 */

#include <dgl/packed_func_ext.h>
#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/random.h>
#include <utility>
#include "randomwalks.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

template<>
TypeArray GetNodeTypesFromMetapath<kDLCPU>(
    const HeteroGraphPtr hg,
    const TypeArray metapath) {
  uint64_t num_etypes = metapath->shape[0];
  TypeArray result = TypeArray::Empty(
      {metapath->shape[0] + 1}, metapath->dtype, metapath->ctx);

  const dgl_type_t *metapath_data = static_cast<dgl_type_t *>(metapath->data);
  dgl_type_t *result_data = static_cast<dgl_type_t *>(result->data);

  dgl_type_t etype = metapath_data[0];
  dgl_type_t srctype = hg->GetEndpointTypes(etype).first;
  dgl_type_t curr_type = srctype;
  result_data[0] = curr_type;

  for (uint64_t i = 0; i < num_etypes; ++i) {
    etype = metapath_data[i];
    auto src_dst_type = hg->GetEndpointTypes(etype);
    dgl_type_t srctype = src_dst_type.first;
    dgl_type_t dsttype = src_dst_type.second;

    if (srctype != curr_type) {
      LOG(FATAL) << "source of edge type #" << i <<
        " does not match destination of edge type #" << i - 1;
      return result;
    }
    curr_type = dsttype;
    result_data[i + 1] = dsttype;
  }
  return result;
}

template<>
IdArray RandomWalk<kDLCPU>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob) {
  int64_t trace_length = metapath->shape[0] + 1;
  int64_t num_seeds = seeds->shape[0];
  IdArray traces = IdArray::Empty({num_seeds, trace_length}, seeds->dtype, seeds->ctx);

  const dgl_type_t *metapath_data = static_cast<dgl_type_t *>(metapath->data);
  const dgl_id_t *seed_data = static_cast<dgl_id_t *>(seeds->data);
  dgl_id_t *traces_data = static_cast<dgl_id_t *>(traces->data);

#pragma omp parallel for
  for (int64_t seed_id = 0; seed_id < num_seeds; ++seed_id) {
    int64_t i;
    dgl_id_t curr = seed_data[seed_id];
    traces_data[seed_id * trace_length] = curr;

    for (i = 0; i < metapath->shape[0]; ++i) {
      dgl_type_t etype = metapath_data[i];

      const auto &succ = hg->SuccVec(etype, curr);
      int64_t size = succ.size();
      if (size == 0)
        // no successor, stop
        break;

      FloatArray prob_etype = prob[etype];
      if (prob_etype->shape[0] == 0) {
        // empty probability array; assume uniform
        curr = succ[RandomEngine::ThreadLocal()->RandInt(size)];
      } else {
        // non-uniform random walk
        const auto eids = hg->OutEdgeVec(etype, curr);
        FloatArray prob_selected = FloatArray::Empty(
            {size}, prob_etype->dtype, prob_etype->ctx);

        // do an IndexSelect on OutEdgeVec which is not an NDArray
        ATEN_FLOAT_TYPE_SWITCH(prob_etype->dtype, DType, "probability", {
          const DType *prob_etype_data = static_cast<DType *>(prob_etype->data);
          DType *prob_selected_data = static_cast<DType *>(prob_selected->data);
          for (int64_t i = 0; i < size; ++i)
            prob_selected_data[i] = prob_etype_data[eids[i]];
        });

        curr = succ[RandomEngine::ThreadLocal()->Choice<int64_t>(prob_selected)];
      }

      traces_data[seed_id * trace_length + i + 1] = curr;

      // restart probability
      if (restart_prob > 0 && RandomEngine::ThreadLocal()->Uniform<double>() < restart_prob)
        break;
    }

    // pad
    for (; i < metapath->shape[0]; ++i)
      traces_data[seed_id * trace_length + i + 1] = -1;
  }

  return traces;
}

};  // namespace impl

};  // namespace sampling

};  // namespace dgl
