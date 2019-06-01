/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/heterograph.cc
 * \brief DGL heterogeneous graph index implementation
 */

#include <dgl/heterograph.h>
#include <dgl/sampler.h>

namespace dgl {

HeteroGraphPtr HeteroGraph::Slice(const RelationType &r) {
  auto idx = _subgraphs[r.get_id()];
  CHECK_EQ(idx->NumEdgeType(), 1);
  return HeteroGraphPtr(new HeteroGraph({idx}, idx->NumNodeType(), idx->NumEdgeType()));
}

HeteroGraphPtr HeteroGraph::Slice(const std::vector<RelationType> &rs) {
  return nullptr;
}

void HeteroGraph::SampleNeighbors(const CSR &csr, dgl_id_t local_vid, int k,
                                  std::vector<dgl_id_t> *sampled) {
  // TODO(zhengda) we need to fix the seed.
  static thread_local unsigned int time_seed = randseed();
  DGLIdIters it = csr.SuccVec(local_vid);
  if (it.size() <= k) {
    for (int i = 0; i < it.size(); i++)
      sampled->push_back(it[i]);
  } else {
    std::vector<size_t> idxs;
    idxs.reserve(k);
    RandomSample(it.size(), k, &idxs, &time_seed);
    std::sort(idxs.begin(), idxs.end());
    for (auto idx : idxs)
      sampled->push_back(it[idx]);
  }
}

}  // namespace dgl
