/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampler/randomwalks_cpu.cc
 * \brief DGL sampler - CPU implementation of random walks with OpenMP
 */

#include <dgl/packed_func_ext.h>
#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/random.h>
#include <dmlc/omp.h>
#include <utility>
#include <iomanip>
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
IdArray GenericRandomWalk(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    int64_t max_num_steps,
    StepFunc step) {
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
      traces_data[seed_id * max_num_steps + i + 1] = -1;
  }

  return traces;
}

template<DLDeviceType XPU, typename IdxType>
std::pair<dgl_id_t, bool> _MetapathRandomWalkStep(
    void *data,
    dgl_id_t curr,
    int64_t len,
    const std::vector<std::vector<IdArray> > &edges_by_type,
    const IdxType *metapath_data,
    const std::vector<FloatArray> &prob,
    double restart_prob) {
  dgl_type_t etype = metapath_data[len];

  // Note that since the selection of successors is very lightweight (especially in the
  // uniform case), we want to reduce the overheads (even from object copies or object
  // construction) as much as possible.
  // Using Successors() slows down by 2x.
  // Using OutEdges() slows down by 10x.
  const auto &csr_arrays = edges_by_type[etype];
  const IdxType *offsets = static_cast<IdxType *>(csr_arrays[0]->data);
  const IdxType *all_succ = static_cast<IdxType *>(csr_arrays[1]->data);
  const IdxType *succ = all_succ + offsets[curr];

  int64_t size = offsets[curr + 1] - offsets[curr];
  if (size == 0)
    return std::make_pair(-1, true);

  FloatArray prob_etype = prob[etype];
  if (prob_etype->shape[0] == 0) {
    // empty probability array; assume uniform
    IdxType idx = RandomEngine::ThreadLocal()->RandInt(size);
    curr = succ[idx];
  } else {
    // non-uniform random walk
    const IdxType *all_eids = static_cast<IdxType *>(csr_arrays[2]->data);
    const IdxType *eids = all_eids + offsets[curr];

    ATEN_FLOAT_TYPE_SWITCH(prob_etype->dtype, DType, "probability", {
      const DType *prob_etype_data = static_cast<DType *>(prob_etype->data);
      std::vector<DType> prob_selected;
      for (int64_t j = 0; j < size; ++j)
        prob_selected.push_back(prob_etype_data[eids[j]]);

      curr = succ[RandomEngine::ThreadLocal()->Choice<int64_t>(prob_selected)];
    });
  }

  bool restart = (
      restart_prob > 0 &&
      RandomEngine::ThreadLocal()->Uniform<double>() < restart_prob);
  return std::make_pair(curr, restart);
}

template<DLDeviceType XPU, typename IdxType>
IdArray RandomWalk(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob) {
  int64_t max_num_steps = metapath->shape[0];
  const IdxType *metapath_data = static_cast<IdxType *>(metapath->data);

  // Prefetch all edges.
  // This forces the heterograph to materialize all OutCSR's before the OpenMP loop;
  // otherwise data races will happen.
  // TODO(BarclayII): should we later on materialize COO/CSR/CSC anyway unless told otherwise?
  std::vector<std::vector<IdArray> > edges_by_type;
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype)
    edges_by_type.push_back(hg->GetAdj(etype, true, "csr"));

  StepFunc step =
    [&edges_by_type, metapath_data, &prob, restart_prob]
    (void *data, dgl_id_t curr, int64_t len) {
      return _MetapathRandomWalkStep<XPU, IdxType>(
          data, curr, len, edges_by_type, metapath_data, prob, restart_prob);
    };

  return GenericRandomWalk<XPU, IdxType>(hg, seeds, max_num_steps, step);
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
