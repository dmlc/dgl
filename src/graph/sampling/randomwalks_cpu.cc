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
#include <dgl/profile.h>

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

int64_t RandomWalkOneSeed(
    const HeteroGraphPtr hg,
    dgl_id_t seed,
    const TypeArray metapath,
    const List<Value> &transition_prob,
    IdArray vids,
    IdArray vtypes,
    double restart_prob) {
  uint64_t num_etypes = metapath->shape[0];
  int64_t len;
  TIMEIT_ALLOC(T, iter, 10);
  TIMEIT(T, 7, {

  dgl_type_t etype, curr_type, dsttype;
  dgl_id_t curr_id;
  uint64_t i = 0;
  TIMEIT(T, 0, {
    etype = IndexSelect<int64_t>(metapath, 0);
    curr_type = hg->GetEndpointTypes(etype).first;
    curr_id = seed;
  });

  TIMEIT(T, 1, {
  Assign(vtypes, 0, curr_type);
  Assign(vids, 0, curr_id);
  });

  // Perform random walk
  for (; i < num_etypes; ++i) {
    // get edge type and endpoint node types
    TIMEIT(T, 2, {
    etype = IndexSelect<int64_t>(metapath, i);
    auto src_dst_type = hg->GetEndpointTypes(etype);
    dsttype = src_dst_type.second;
    });

    // find all successors
    EdgeArray edges;
    IdArray succs, eids;
    TIMEIT(T, 3, {
    edges = hg->OutEdges(etype, curr_id, false);
    succs = edges.dst;
    eids = edges.id;
    });
    int64_t size = succs->shape[0];
    if (size == 0)
      // no successors; halt and pad
      break;

    // pick one successor
    TIMEIT(T, 4, {
    int64_t sel;
    FloatArray p_etype = transition_prob[etype]->data;
    if (IsEmpty(p_etype)) {
      // uniform if empty prob array is given
      sel = RandomEngine::ThreadLocal()->RandInt(size);
    } else {
      FloatArray selected_probs = IndexSelect(p_etype, eids);
      sel = RandomEngine::ThreadLocal()->Choice<int64_t>(selected_probs);
    }
    curr_id = IndexSelect<int64_t>(succs, sel);
    curr_type = dsttype;
    });

    TIMEIT(T, 5, {
    Assign(vtypes, i + 1, curr_type);
    Assign(vids, i + 1, curr_id);
    });

    // determine if terminate the trace
    double p = RandomEngine::ThreadLocal()->Uniform<double>();
    if (p < restart_prob)
      break;
  }

  len = i;  // record and return number of hops jumped
  // pad
  TIMEIT(T, 6, {
  for (; i < num_etypes; ++i) {
    Assign(vtypes, i + 1, -1);
    Assign(vids, i + 1, -1);
  }
  });

  });

  TIMEIT_CHECK(T, iter, 8, 2000, "\t\t");

  return len;
}

template<DLDeviceType XPU>
std::pair<IdArray, TypeArray> RandomWalkImpl(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const List<Value> &prob) {
  TIMEIT_ALLOC(T, iter, 10);
  int64_t num_seeds = seeds->shape[0];
  int64_t trace_length = metapath->shape[0] + 1;
  IdArray vids, vids_i;
  TypeArray vtypes, vtypes_i;

  TIMEIT(T, 0, {
  vids = IdArray::Empty(
    {num_seeds, trace_length}, seeds->dtype, seeds->ctx);
  vtypes = TypeArray::Empty(
    {num_seeds, trace_length}, metapath->dtype, metapath->ctx);
  });

  for (int64_t i = 0; i < num_seeds; ++i) {
    TIMEIT(T, 1, {
    vids_i = vids.CreateView(
      {trace_length}, vids->dtype, i * trace_length * vids->dtype.bits / 8);
    vtypes_i = vtypes.CreateView(
      {trace_length}, vtypes->dtype, i * trace_length * vtypes->dtype.bits / 8);
    });

    TIMEIT(T, 2, {
    RandomWalkOneSeed(
        hg, IndexSelect<int64_t>(seeds, i), metapath, prob, vids_i, vtypes_i, 0.);
    });
  }

  TIMEIT_CHECK(T, iter, 3, 10, "\t");

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
