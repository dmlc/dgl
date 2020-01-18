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

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

int64_t RandomWalkOneSeed(
    const HeteroGraphPtr hg,
    dgl_id_t seed,
    const TypeArray metapath,
    const std::vector<FloatArray> &transition_prob,
    IdArray vids,
    IdArray vtypes,
    double restart_prob) {
  uint64_t num_etypes = metapath->shape[0];

  dgl_type_t etype = IndexSelect<int64_t>(metapath, 0);
  dgl_type_t curr_type = hg->GetEndpointTypes(etype).first;
  dgl_id_t curr_id = seed;
  uint64_t i = 0;

  Assign(vtypes, 0, curr_type);
  Assign(vids, 0, curr_id);

  // Perform random walk
  for (; i < num_etypes; ++i) {
    // get edge type and endpoint node types
    etype = IndexSelect<int64_t>(metapath, i);
    auto src_dst_type = hg->GetEndpointTypes(etype);
    dgl_type_t dsttype = src_dst_type.second;

    // find all successors
    EdgeArray edges = hg->OutEdges(etype, curr_id);
    IdArray succs = edges.dst;
    IdArray eids = edges.id;
    int64_t size = succs->shape[0];
    if (size == 0)
      // no successors; halt and pad
      break;

    // pick one successor
    int64_t sel;
    FloatArray p_etype = transition_prob[etype];
    if (IsEmpty(p_etype)) {
      // uniform if empty prob array is given
      sel = RandomEngine::ThreadLocal()->RandInt(size);
    } else {
      FloatArray selected_probs = IndexSelect(p_etype, eids);
      sel = RandomEngine::ThreadLocal()->Choice<int64_t>(selected_probs);
    }
    curr_id = IndexSelect<int64_t>(succs, sel);
    curr_type = dsttype;

    Assign(vtypes, i + 1, curr_type);
    Assign(vids, i + 1, curr_id);

    // determine if terminate the trace
    double p = RandomEngine::ThreadLocal()->Uniform<double>();
    if (p < restart_prob)
      break;
  }

  int64_t len = i;  // record and return number of hops jumped
  // pad
  for (; i < num_etypes; ++i) {
    Assign(vtypes, i + 1, -1);
    Assign(vids, i + 1, -1);
  }

  return len;
}

template<DLDeviceType XPU>
std::pair<IdArray, TypeArray> RandomWalkImpl(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob) {
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
    const std::vector<FloatArray> &prob);

};  // namespace impl

};  // namespace sampling

};  // namespace dgl
