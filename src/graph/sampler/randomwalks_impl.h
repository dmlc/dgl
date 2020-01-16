/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampler/randomwalks_impl.h
 * \brief DGL sampler - templated implementation of random walks
 */

#ifndef SAMPLER_RANDOMWALKS_IMPL_H_
#define SAMPLER_RANDOMWALKS_IMPL_H_

#include <dgl/base_heterograph.h>
#include <dgl/runtime/container.h>
#include <dgl/array.h>
#include <dgl/random.h>

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

int64_t RandomWalkOneSeed(
    const HeteroGraphPtr hg,
    dgl_id_t seed,
    const TypeArray etypes,
    const List<Value> &transition_prob,
    IdArray vids,
    IdArray vtypes,
    double restart_prob) {
  int64_t num_etypes = etypes->shape[0];

  dgl_type_t etype = IndexSelect<int64_t>(etypes, 0);
  dgl_type_t srctype = hg->GetEndpointTypes(etype).first;
  dgl_type_t curr_type = srctype;
  dgl_id_t curr_id = seed;
  int64_t i = 0;

  Assign(vtypes, 0, curr_type);
  Assign(vids, 0, curr_id);

  // Perform random walk
  for (; i < num_etypes; ++i) {
    // get edge type and endpoint node types
    etype = IndexSelect<int64_t>(etypes, i);
    auto src_dst_type = hg->GetEndpointTypes(etype);
    dgl_type_t srctype = src_dst_type.first;
    dgl_type_t dsttype = src_dst_type.second;

    if (srctype != curr_type) {
      LOG(FATAL) << "Node type mismatch in metapath";
      return 0;
    }

    // find all successors
    EdgeArray edges = hg->OutEdges(etype, seed);
    IdArray succs = edges.dst;
    IdArray eids = edges.id;
    int64_t size = succs->shape[0];
    if (succs->shape[0] == 0)
      // no successors; halt and pad
      break;

    // pick one successor
    int64_t sel;
    FloatArray p_etype = transition_prob[etype]->data;
    if (p_etype->ndim == 0) {     // uniform if empty prob array is given
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

};  // namespace impl

};  // namespace sampling

};  // namespace dgl

#endif  // SAMPLER_RANDOMWALKS_IMPL_H_
