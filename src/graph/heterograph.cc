/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/heterograph.cc
 * \brief DGL heterogeneous graph index implementation
 */

#include <dgl/heterograph.h>
#include <dgl/sampler.h>

HeteroGraphPtr HeteroGraph::Slice(const RelationType &r) {
  return subgraphs[r.get_id()];
}

HeteroGraphPtr HeteroGraph::Slice(const std::vector<RelationType> &rs) {
  return nullptr;
}

void HeteroGraph::SampleNeighbors(const CSR &csr, dgl_id_t local_vid, int k,
                                  std::vector<dgl_id_t> *sampled) {
  DGLIdIters it = csr.SuccVec(local_vid);
  if (it.size() <= k) {
    for (int i = 0; i < it.size(); i++)
      sampled.push_back(it[i]);
  } else {
    std::vector<size_t> idxs;
    idxs.reserve(k);
    RandomSample(it.size(), k, &idxs, &seed);
    std::sort(idxs.begin(), idxs.end());
    for (auto idx : idxs)
      sampled->push_back(it[idx]);
  }
}
