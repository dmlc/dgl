/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampler/cpu/randomwalks.h
 * \brief DGL sampler templated implementation of random walks
 */

#include <dgl/base_heterograph.h>
#include <dgl/array.h>
#include <dgl/random.h>

namespace dgl {

namespace sampling {

namespace impl {

template<DLDeviceType XPU>
std::pair<IdArray, TypeArray> RandomWalkImpl(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray etypes,
    const FloatArray prob);

int64_t RandomWalkOneSeed(
    const HeteroGraphPtr hg,
    dgl_id_t seed,
    const TypeArray etypes,
    const FloatArray transition_prob,
    IdArray vids,
    IdArray vtypes,
    double restart_prob) {
  bool uniform = (transition_prob->shape[0] == 0);

  int64_t num_seeds = seeds->shape[0];
  int64_t num_etypes = etypes->shape[0];

  dgl_type_t etype = IndexSelect(etypes, 0);
  dgl_type_t srctype = hg->GetEndpointTypes(etype).first;
  dgl_type_t curr_type = srctype;
  dgl_id_t curr_id = seed;
  int64_t i = 0;

  Assign(vtypes, 0, curr_type);
  Assign(vids, 0, curr_id);

  // Perform random walk
  for (; i < num_etypes; ++i) {
    etype = IndexSelect(etypes, i);
    auto src_dst_type = hg->GetEndpointTypes(etype);
    dgl_type_t srctype = src_dst_type.first;
    dgl_type_t dsttype = src_dst_type.second;

    if (srctype != curr_type) {
      LOG(FATAL) << "Node type mismatch in metapath";
      return 0;
    }

    EdgeArray edges = hg->OutEdges(etype, seed);
    IdArray succs = edges->dst;
    IdArray eids = edges->id;
    int64_t size = succs->shape[0];
    if (succs->shape[0] == 0)
      // no successors; halt and pad
      break;

    int64_t sel;
    if (uniform) {
      sel = RandomEngine::ThreadLocal()->RandInt(size);
    } else {
      FloatArray selected_probs = IndexSelect(prob, eids);
      sel = RandomEngine::ThreadLocal()->Choice(selected_probs);
    }
    curr_id = IndexSelect(succs, RandomEngine::ThreadLocal()->RandInt(size));
    curr_type = dsttype;

    Assign(vtypes, i + 1, curr_type);
    Assign(vids, i + 1, curr_id);

    double p = RandomEngine::ThreadLocal()->Uniform();
    if (p < restart_prob):
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

