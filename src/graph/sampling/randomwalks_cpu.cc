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

template<DLDeviceType XPU, typename IdxType>
TypeArray GetNodeTypesFromMetapath(
    const HeteroGraphPtr hg,
    const TypeArray metapath) {
  uint64_t num_etypes = metapath->shape[0];
  TypeArray result = TypeArray::Empty(
      {metapath->shape[0] + 1}, metapath->dtype, metapath->ctx);

  const IdxType *metapath_data = static_cast<IdxType *>(metapath->data);
  IdxType *result_data = static_cast<IdxType *>(result->data);

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

template
TypeArray GetNodeTypesFromMetapath<kDLCPU, int32_t>(
    const HeteroGraphPtr hg,
    const TypeArray metapath);
template
TypeArray GetNodeTypesFromMetapath<kDLCPU, int64_t>(
    const HeteroGraphPtr hg,
    const TypeArray metapath);

template<DLDeviceType XPU, typename IdxType>
IdArray RandomWalk(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob) {
  int64_t trace_length = metapath->shape[0] + 1;
  int64_t num_seeds = seeds->shape[0];
  IdArray traces = IdArray::Empty({num_seeds, trace_length}, seeds->dtype, seeds->ctx);

  const IdxType *metapath_data = static_cast<IdxType *>(metapath->data);
  const IdxType *seed_data = static_cast<IdxType *>(seeds->data);
  IdxType *traces_data = static_cast<IdxType *>(traces->data);

#pragma omp parallel for
  for (int64_t seed_id = 0; seed_id < num_seeds; ++seed_id) {
    int64_t i;
    dgl_id_t curr = seed_data[seed_id];
    traces_data[seed_id * trace_length] = curr;

    for (i = 0; i < metapath->shape[0]; ++i) {
      dgl_type_t etype = metapath_data[i];

#ifdef NO_SUCCVEC
      const auto &etype_csr = hg->GetAdj(etype, true, "csr");
      const IdxType *offsets = static_cast<IdxType *>(etype_csr[0]->data);
      const IdxType *all_succ = static_cast<IdxType *>(etype_csr[1]->data);
      const IdxType *all_eids = static_cast<IdxType *>(etype_csr[2]->data);
      const IdxType *succ = all_succ + offsets[curr];
      const IdxType *eids = all_eids + offsets[curr];

      int64_t size = offsets[curr + 1] - offsets[curr];
#else
      const auto &succ = hg->SuccVec(etype, curr);
      int64_t size = succ.size();
#endif
      if (size == 0)
        // no successor, stop
        break;

      FloatArray prob_etype = prob[etype];
      if (prob_etype->shape[0] == 0) {
        // empty probability array; assume uniform
        curr = succ[RandomEngine::ThreadLocal()->RandInt(size)];
      } else {
#ifndef NO_SUCCVEC
        const auto &eids = hg->OutEdgeVec(etype, curr);
#endif
        // non-uniform random walk
        ATEN_FLOAT_TYPE_SWITCH(prob_etype->dtype, DType, "probability", {
          const DType *prob_etype_data = static_cast<DType *>(prob_etype->data);
          std::vector<DType> prob_selected;
          for (int64_t i = 0; i < size; ++i)
            prob_selected.push_back(prob_etype_data[eids[i]]);

          curr = succ[RandomEngine::ThreadLocal()->Choice<int64_t>(prob_selected)];
        });
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

template
IdArray RandomWalk<kDLCPU, int32_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob);
template
IdArray RandomWalk<kDLCPU, int64_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob);

};  // namespace impl

};  // namespace sampling

};  // namespace dgl
