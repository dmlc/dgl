/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/immutable_graph.cc
 * \brief DGL immutable graph index implementation
 */

#include <dgl/immutable_graph.h>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "../c_api_common.h"

namespace dgl {

/*
 * ArrayHeap is used to sample elements from vector
 */
class ArrayHeap {
 public:
  explicit ArrayHeap(const std::vector<float>& prob) {
    vec_size_ = prob.size();
    bit_len_ = ceil(log2(vec_size_));
    limit_ = 1 << bit_len_;
    // allocate twice the size
    heap_.resize(limit_ << 1, 0);
    // allocate the leaves
    for (int i = limit_; i < vec_size_+limit_; ++i) {
      heap_[i] = prob[i-limit_];
    }
    // iterate up the tree (this is O(m))
    for (int i = bit_len_-1; i >= 0; --i) {
      for (int j = (1 << i); j < (1 << (i + 1)); ++j) {
        heap_[j] = heap_[j << 1] + heap_[(j << 1) + 1];
      }
    }
  }
  ~ArrayHeap() {}

  /*
   * Remove term from index (this costs O(log m) steps)
   */
  void Delete(size_t index) {
    size_t i = index + limit_;
    float w = heap_[i];
    for (int j = bit_len_; j >= 0; --j) {
      heap_[i] -= w;
      i = i >> 1;
    }
  }

  /*
   * Add value w to index (this costs O(log m) steps)
   */
  void Add(size_t index, float w) {
    size_t i = index + limit_;
    for (int j = bit_len_; j >= 0; --j) {
      heap_[i] += w;
      i = i >> 1;
    }
  }

  /*
   * Sample from arrayHeap
   */
  size_t Sample(unsigned int* seed) {
    float xi = heap_[1] * (rand_r(seed)%100/101.0);
    int i = 1;
    while (i < limit_) {
      i = i << 1;
      if (xi >= heap_[i]) {
        xi -= heap_[i];
        i += 1;
      }
    }
    return i - limit_;
  }

  /*
   * Sample a vector by given the size n
   */
  void SampleWithoutReplacement(size_t n, std::vector<size_t>* samples, unsigned int* seed) {
    // sample n elements
    for (size_t i = 0; i < n; ++i) {
      samples->at(i) = this->Sample(seed);
      this->Delete(samples->at(i));
    }
  }

 private:
  int vec_size_;  // sample size
  int bit_len_;   // bit size
  int limit_;
  std::vector<float> heap_;
};

/*
 * Uniformly sample integers from [0, set_size) without replacement.
 */
static void RandomSample(size_t set_size,
                         size_t num,
                         std::vector<size_t>* out,
                         unsigned int* seed) {
  std::unordered_set<size_t> sampled_idxs;
  while (sampled_idxs.size() < num) {
    sampled_idxs.insert(rand_r(seed) % set_size);
  }
  out->clear();
  for (auto it = sampled_idxs.begin(); it != sampled_idxs.end(); it++) {
    out->push_back(*it);
  }
}

/*
 * For a sparse array whose non-zeros are represented by nz_idxs,
 * negate the sparse array and outputs the non-zeros in the negated array.
 */
static void NegateArray(const std::vector<size_t> &nz_idxs,
                        size_t arr_size,
                        std::vector<size_t>* out) {
  // nz_idxs must have been sorted.
  auto it = nz_idxs.begin();
  size_t i = 0;
  CHECK_GT(arr_size, nz_idxs.back());
  for (; i < arr_size && it != nz_idxs.end(); i++) {
    if (*it == i) {
      it++;
      continue;
    }
    out->push_back(i);
  }
  for (; i < arr_size; i++) {
    out->push_back(i);
  }
}

/*
 * Uniform sample vertices from a list of vertices.
 */
static void GetUniformSample(const dgl_id_t* val_list,
                             const dgl_id_t* ver_list,
                             const size_t ver_len,
                             const size_t max_num_neighbor,
                             std::vector<dgl_id_t>* out_ver,
                             std::vector<dgl_id_t>* out_edge,
                             unsigned int* seed) {
  // Copy ver_list to output
  if (ver_len <= max_num_neighbor) {
    for (size_t i = 0; i < ver_len; ++i) {
      out_ver->push_back(ver_list[i]);
      out_edge->push_back(val_list[i]);
    }
    return;
  }
  // If we just sample a small number of elements from a large neighbor list.
  std::vector<size_t> sorted_idxs;
  if (ver_len > max_num_neighbor * 2) {
    sorted_idxs.reserve(max_num_neighbor);
    RandomSample(ver_len, max_num_neighbor, &sorted_idxs, seed);
    std::sort(sorted_idxs.begin(), sorted_idxs.end());
  } else {
    std::vector<size_t> negate;
    negate.reserve(ver_len - max_num_neighbor);
    RandomSample(ver_len, ver_len - max_num_neighbor,
                 &negate, seed);
    std::sort(negate.begin(), negate.end());
    NegateArray(negate, ver_len, &sorted_idxs);
  }
  // verify the result.
  CHECK_EQ(sorted_idxs.size(), max_num_neighbor);
  for (size_t i = 1; i < sorted_idxs.size(); i++) {
    CHECK_GT(sorted_idxs[i], sorted_idxs[i - 1]);
  }
  for (auto idx : sorted_idxs) {
    out_ver->push_back(ver_list[idx]);
    out_edge->push_back(val_list[idx]);
  }
}

/*
 * Non-uniform sample via ArrayHeap
 */
static void GetNonUniformSample(const float* probability,
                                const dgl_id_t* val_list,
                                const dgl_id_t* ver_list,
                                const size_t ver_len,
                                const size_t max_num_neighbor,
                                std::vector<dgl_id_t>* out_ver,
                                std::vector<dgl_id_t>* out_edge,
                                unsigned int* seed) {
  // Copy ver_list to output
  if (ver_len <= max_num_neighbor) {
    for (size_t i = 0; i < ver_len; ++i) {
      out_ver->push_back(ver_list[i]);
      out_edge->push_back(val_list[i]);
    }
    return;
  }
  // Make sample
  std::vector<size_t> sp_index(max_num_neighbor);
  std::vector<float> sp_prob(ver_len);
  for (size_t i = 0; i < ver_len; ++i) {
    sp_prob[i] = probability[ver_list[i]];
  }
  ArrayHeap arrayHeap(sp_prob);
  arrayHeap.SampleWithoutReplacement(max_num_neighbor, &sp_index, seed);
  out_ver->resize(max_num_neighbor);
  out_edge->resize(max_num_neighbor);
  for (size_t i = 0; i < max_num_neighbor; ++i) {
    size_t idx = sp_index[i];
    out_ver->at(i) = ver_list[idx];
    out_edge->at(i) = val_list[idx];
  }
  sort(out_ver->begin(), out_ver->end());
  sort(out_edge->begin(), out_edge->end());
}

/*
 * Used for subgraph sampling
 */
struct neigh_list {
  std::vector<dgl_id_t> neighs;
  std::vector<dgl_id_t> edges;
  neigh_list(const std::vector<dgl_id_t> &_neighs,
             const std::vector<dgl_id_t> &_edges)
    : neighs(_neighs), edges(_edges) {}
};

SampledSubgraph ImmutableGraph::SampleSubgraph(IdArray seed_arr,
                                               const float* probability,
                                               const std::string &neigh_type,
                                               int num_hops,
                                               size_t num_neighbor) const {
  unsigned int time_seed = time(nullptr);
  size_t num_seeds = seed_arr->shape[0];
  auto orig_csr = neigh_type == "in" ? GetInCSR() : GetOutCSR();
  const dgl_id_t* val_list = orig_csr->edge_ids.data();
  const dgl_id_t* col_list = orig_csr->indices.data();
  const int64_t* indptr = orig_csr->indptr.data();
  const dgl_id_t* seed = static_cast<dgl_id_t*>(seed_arr->data);

  // BFS traverse the graph and sample vertices
  // <vertex_id, layer_id>
  std::unordered_set<dgl_id_t> sub_ver_map;
  std::vector<std::pair<dgl_id_t, int> > sub_vers;
  sub_vers.reserve(num_seeds * 10);
  // add seed vertices
  for (size_t i = 0; i < num_seeds; ++i) {
    auto ret = sub_ver_map.insert(seed[i]);
    // If the vertex is inserted successfully.
    if (ret.second) {
      sub_vers.emplace_back(seed[i], 0);
    }
  }
  std::vector<dgl_id_t> tmp_sampled_src_list;
  std::vector<dgl_id_t> tmp_sampled_edge_list;
  // ver_id, position
  std::vector<std::pair<dgl_id_t, size_t> > neigh_pos;
  neigh_pos.reserve(num_seeds);
  std::vector<dgl_id_t> neighbor_list;
  int64_t num_edges = 0;

  // sub_vers is used both as a node collection and a queue.
  // In the while loop, we iterate over sub_vers and new nodes are added to the vector.
  // A vertex in the vector only needs to be accessed once. If there is a vertex behind idx
  // isn't in the last level, we will sample its neighbors. If not, the while loop terminates.
  size_t idx = 0;
  while (idx < sub_vers.size()) {
    dgl_id_t dst_id = sub_vers[idx].first;
    int cur_node_level = sub_vers[idx].second;
    idx++;
    // If the node is in the last level, we don't need to sample neighbors
    // from this node.
    if (cur_node_level >= num_hops)
      continue;

    tmp_sampled_src_list.clear();
    tmp_sampled_edge_list.clear();
    dgl_id_t ver_len = *(indptr+dst_id+1) - *(indptr+dst_id);
    if (probability == nullptr) {  // uniform-sample
      GetUniformSample(val_list + *(indptr + dst_id),
                       col_list + *(indptr + dst_id),
                       ver_len,
                       num_neighbor,
                       &tmp_sampled_src_list,
                       &tmp_sampled_edge_list,
                       &time_seed);
    } else {  // non-uniform-sample
      GetNonUniformSample(probability,
                       val_list + *(indptr + dst_id),
                       col_list + *(indptr + dst_id),
                       ver_len,
                       num_neighbor,
                       &tmp_sampled_src_list,
                       &tmp_sampled_edge_list,
                       &time_seed);
    }
    CHECK_EQ(tmp_sampled_src_list.size(), tmp_sampled_edge_list.size());
    size_t pos = neighbor_list.size();
    neigh_pos.emplace_back(dst_id, pos);
    // First we push the size of neighbor vector
    neighbor_list.push_back(tmp_sampled_edge_list.size());
    // Then push the vertices
    for (size_t i = 0; i < tmp_sampled_src_list.size(); ++i) {
      neighbor_list.push_back(tmp_sampled_src_list[i]);
    }
    // Finally we push the edge list
    for (size_t i = 0; i < tmp_sampled_edge_list.size(); ++i) {
      neighbor_list.push_back(tmp_sampled_edge_list[i]);
    }
    num_edges += tmp_sampled_src_list.size();
    for (size_t i = 0; i < tmp_sampled_src_list.size(); ++i) {
      // We need to add the neighbor in the hashtable here. This ensures that
      // the vertex in the queue is unique. If we see a vertex before, we don't
      // need to add it to the queue again.
      auto ret = sub_ver_map.insert(tmp_sampled_src_list[i]);
      // If the sampled neighbor is inserted to the map successfully.
      if (ret.second)
        sub_vers.emplace_back(tmp_sampled_src_list[i], cur_node_level + 1);
    }
  }
  // Let's check if there is a vertex that we haven't sampled its neighbors.
  for (; idx < sub_vers.size(); idx++) {
    if (sub_vers[idx].second < num_hops) {
      LOG(WARNING)
        << "The sampling is truncated because we have reached the max number of vertices\n"
        << "Please use a smaller number of seeds or a small neighborhood";
      break;
    }
  }

  // Copy sub_ver_map to output[0]
  // Copy layer
  uint64_t num_vertices = sub_ver_map.size();
  std::sort(sub_vers.begin(), sub_vers.end(),
            [](const std::pair<dgl_id_t, dgl_id_t> &a1, const std::pair<dgl_id_t, dgl_id_t> &a2) {
    return a1.first < a2.first;
  });

  SampledSubgraph subg;
  subg.induced_vertices = IdArray::Empty({static_cast<int64_t>(num_vertices)},
                                         DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  subg.induced_edges = IdArray::Empty({static_cast<int64_t>(num_edges)},
                                      DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  subg.layer_ids = IdArray::Empty({static_cast<int64_t>(num_vertices)},
                                  DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  subg.sample_prob = runtime::NDArray::Empty({static_cast<int64_t>(num_vertices)},
                                             DLDataType{kDLFloat, 32, 1}, DLContext{kDLCPU, 0});

  dgl_id_t *out = static_cast<dgl_id_t *>(subg.induced_vertices->data);
  dgl_id_t *out_layer = static_cast<dgl_id_t *>(subg.layer_ids->data);
  for (size_t i = 0; i < sub_vers.size(); i++) {
    out[i] = sub_vers[i].first;
    out_layer[i] = sub_vers[i].second;
  }

  // Copy sub_probability
  float *sub_prob = static_cast<float *>(subg.sample_prob->data);
  if (probability != nullptr) {
    for (size_t i = 0; i < sub_ver_map.size(); ++i) {
      dgl_id_t idx = out[i];
      sub_prob[i] = probability[idx];
    }
  }

  // Construct sub_csr_graph
  auto subg_csr = std::make_shared<CSR>(num_vertices, num_edges);
  subg_csr->indices.resize(num_edges);
  subg_csr->edge_ids.resize(num_edges);
  dgl_id_t* val_list_out = static_cast<dgl_id_t *>(subg.induced_edges->data);
  dgl_id_t* col_list_out = subg_csr->indices.data();
  int64_t* indptr_out = subg_csr->indptr.data();
  size_t collected_nedges = 0;

  // Both the out array and neigh_pos are sorted. By scanning the two arrays, we can see
  // which vertices have neighbors and which don't.
  std::sort(neigh_pos.begin(), neigh_pos.end(),
            [](const std::pair<dgl_id_t, size_t> &a1, const std::pair<dgl_id_t, size_t> &a2) {
    return a1.first < a2.first;
  });
  size_t idx_with_neigh = 0;
  for (size_t i = 0; i < num_vertices; i++) {
    dgl_id_t dst_id = *(out + i);
    // If a vertex is in sub_ver_map but not in neigh_pos, this vertex must not
    // have edges.
    size_t edge_size = 0;
    if (idx_with_neigh < neigh_pos.size() && dst_id == neigh_pos[idx_with_neigh].first) {
      size_t pos = neigh_pos[idx_with_neigh].second;
      CHECK_LT(pos, neighbor_list.size());
      edge_size = neighbor_list[pos];
      CHECK_LE(pos + edge_size * 2 + 1, neighbor_list.size());

      std::copy_n(neighbor_list.begin() + pos + 1,
                  edge_size,
                  col_list_out + collected_nedges);
      std::copy_n(neighbor_list.begin() + pos + edge_size + 1,
                  edge_size,
                  val_list_out + collected_nedges);
      collected_nedges += edge_size;
      idx_with_neigh++;
    }
    indptr_out[i+1] = indptr_out[i] + edge_size;
  }

  for (size_t i = 0; i < subg_csr->edge_ids.size(); i++)
    subg_csr->edge_ids[i] = i;

  if (neigh_type == "in")
    subg.graph = GraphPtr(new ImmutableGraph(subg_csr, nullptr, IsMultigraph()));
  else
    subg.graph = GraphPtr(new ImmutableGraph(nullptr, subg_csr, IsMultigraph()));

  return subg;
}

void CompactSubgraph(ImmutableGraph::CSR *subg,
                     const std::unordered_map<dgl_id_t, dgl_id_t> &id_map) {
  for (size_t i = 0; i < subg->indices.size(); i++) {
    auto it = id_map.find(subg->indices[i]);
    CHECK(it != id_map.end());
    subg->indices[i] = it->second;
  }
}

void ImmutableGraph::CompactSubgraph(IdArray induced_vertices) {
  // The key is the old id, the value is the id in the subgraph.
  std::unordered_map<dgl_id_t, dgl_id_t> id_map;
  const dgl_id_t *vdata = static_cast<dgl_id_t *>(induced_vertices->data);
  size_t len = induced_vertices->shape[0];
  for (size_t i = 0; i < len; i++)
    id_map.insert(std::pair<dgl_id_t, dgl_id_t>(vdata[i], i));
  if (in_csr_)
    dgl::CompactSubgraph(in_csr_.get(), id_map);
  if (out_csr_)
    dgl::CompactSubgraph(out_csr_.get(), id_map);
}

SampledSubgraph ImmutableGraph::NeighborUniformSample(IdArray seeds,
                                                      const std::string &neigh_type,
                                                      int num_hops, int expand_factor) const {
  auto ret = SampleSubgraph(seeds,                 // seed vector
                            nullptr,               // sample_id_probability
                            neigh_type,
                            num_hops,
                            expand_factor);
  std::static_pointer_cast<ImmutableGraph>(ret.graph)->CompactSubgraph(ret.induced_vertices);
  return ret;
}

dgl_id_t ImmutableGraph::GetRandomSuccessor(dgl_id_t vid,
                                            unsigned int *seed) const {
  const DGLIdIters &succ = this->GetOutCSR()->GetIndexRef(vid);
  const size_t size = succ.size();
  return succ[rand_r(seed) % size];
}

IdArray ImmutableGraph::RandomWalk(IdArray seeds,
                                   int num_traces,
                                   int num_hops) const {
  const int num_nodes = seeds->shape[0];
  const dgl_id_t *seed_ids = static_cast<dgl_id_t *>(seeds->data);
  // TODO: any ways getting rid of these sign casting?
  IdArray traces = IdArray::Empty(
      {num_nodes, num_traces, num_hops + 1},
      DLDataType{kDLInt, 64, 1},
      DLContext{kDLCPU, 0});
  dgl_id_t *trace_data = static_cast<dgl_id_t *>(traces->data);
  unsigned int random_seed = time(nullptr);

  // TODO: openmp support
  for (int i = 0; i < num_nodes; ++i) {
    const dgl_id_t seed_id = seed_ids[i];

    for (int j = 0; j < num_traces; ++j) {
      dgl_id_t cur = seed_id;
      const int kmax = num_hops + 1;

      for (int k = 0; k < kmax; ++k) {
        size_t offset = ((size_t)i * num_traces + j) * kmax + k;

        trace_data[offset] = cur;
        cur = GetRandomSuccessor(cur, &random_seed);
      }
    }
  }

  return traces;
}

}  // namespace dgl
