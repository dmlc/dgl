/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampler.cc
 * \brief DGL sampler implementation
 */

#include <dgl/sampler.h>
#include <dgl/immutable_graph.h>
#include <algorithm>

#ifdef _MSC_VER
// rand in MS compiler works well in multi-threading.
int rand_r(unsigned *seed) {
  return rand();
}
#endif

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

NodeFlow ImmutableGraph::SampleSubgraph(IdArray seed_arr,
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

  std::unordered_set<dgl_id_t> sub_ver_map;  // The vertex Ids in a layer.
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
  std::vector<size_t> layer_offsets(num_hops + 1);
  int64_t num_edges = 0;

  layer_offsets[0] = 0;
  layer_offsets[1] = sub_vers.size();
  size_t idx = 0;
  for (size_t layer_id = 1; layer_id < num_hops; layer_id++) {
    // We need to avoid resampling the same node in a layer, but we allow a node
    // to be resampled in multiple layers. We use `sub_ver_map` to keep track of
    // sampled nodes in a layer, and clear it when entering a new layer.
    sub_ver_map.clear();
    // sub_vers is used both as a node collection and a queue.
    // In the while loop, we iterate over sub_vers and new nodes are added to the vector.
    // A vertex in the vector only needs to be accessed once. If there is a vertex behind idx
    // isn't in the last level, we will sample its neighbors. If not, the while loop terminates.
    while (idx < sub_vers.size() && layer_id - 1 == sub_vers[idx].second) {
      dgl_id_t dst_id = sub_vers[idx].first;
      int cur_node_level = sub_vers[idx].second;
      idx++;

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
        if (ret.second) {
          sub_vers.emplace_back(tmp_sampled_src_list[i], cur_node_level + 1);
        }
      }
    }
    layer_offsets[layer_id + 1] = layer_offsets[layer_id] + sub_ver_map.size();
  }

  uint64_t num_vertices = sub_vers.size();
  NodeFlow nf;
  nf.node_mapping = IdArray::Empty({static_cast<int64_t>(num_vertices)},
                                   DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  nf.edge_mapping = IdArray::Empty({static_cast<int64_t>(num_edges)},
                                   DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  nf.layer_offsets = IdArray::Empty({static_cast<int64_t>(num_hops + 1)},
                                    DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  nf.flow_offsets = IdArray::Empty({static_cast<int64_t>(num_hops)},
                                    DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});

  dgl_id_t *out = static_cast<dgl_id_t *>(nf.node_mapping->data);
  dgl_id_t *out_layer = static_cast<dgl_id_t *>(nf.layer_offsets->data);
  dgl_id_t *out_flow = static_cast<dgl_id_t *>(nf.flow_offsets->data);
  dgl_id_t* val_list_out = static_cast<dgl_id_t *>(nf.edge_mapping->data);

  // Construct sub_csr_graph
  auto subg_csr = std::make_shared<CSR>(num_vertices, num_edges);
  subg_csr->indices.resize(num_edges);
  subg_csr->edge_ids.resize(num_edges);
  dgl_id_t* col_list_out = subg_csr->indices.data();
  int64_t* indptr_out = subg_csr->indptr.data();
  size_t collected_nedges = 0;

  // The data from the previous steps:
  // * node data: sub_vers (vid, layer), neigh_pos,
  // * edge data: neighbor_list, probability.
  // * layer_offsets: the offset in sub_vers.
  dgl_id_t ver_id = 0;
  std::vector<std::unordered_map<dgl_id_t, dgl_id_t>> layer_ver_maps;
  layer_ver_maps.resize(num_hops + 1);
  for (size_t layer_id = 0; layer_id < num_hops; layer_id++) {
    // We sort the vertices in a layer so that we don't need to sort the neighbor Ids
    // after remap to a subgraph.
    std::sort(sub_vers.begin() + layer_offsets[layer_id],
              sub_vers.begin() + layer_offsets[layer_id + 1],
              [](const std::pair<dgl_id_t, dgl_id_t> &a1,
                 const std::pair<dgl_id_t, dgl_id_t> &a2) {
      return a1.first < a2.first;
    });

    // Save the sampled vertices and its layer Id.
    for (size_t i = layer_offsets[layer_id]; i < layer_offsets[layer_id + 1]; i++) {
      out[i] = sub_vers[i].first;
      layer_ver_maps[layer_id].insert(std::pair<dgl_id_t, dgl_id_t>(sub_vers[i].first, ver_id++));
      assert(sub_vers[i].second == layer_id);
    }
  }
  std::copy(layer_offsets.begin(), layer_offsets.end(), out_layer);

  // Remap the neighbors.
  int64_t last_off = 0;
  for (size_t layer_id = 0; layer_id < num_hops - 1; layer_id++) {
    std::sort(neigh_pos.begin() + layer_offsets[layer_id],
              neigh_pos.begin() + layer_offsets[layer_id + 1],
              [](const std::pair<dgl_id_t, size_t> &a1, const std::pair<dgl_id_t, size_t> &a2) {
                return a1.first < a2.first;
              });

    for (size_t i = layer_offsets[layer_id]; i < layer_offsets[layer_id + 1]; i++) {
      dgl_id_t dst_id = *(out + i);
      assert(dst_id == neigh_pos[i].first);
      size_t pos = neigh_pos[i].second;
      CHECK_LT(pos, neighbor_list.size());
      size_t num_edges = neighbor_list[pos];
      CHECK_LE(pos + num_edges * 2 + 1, neighbor_list.size());

      // We need to map the Ids of the neighbors to the subgraph.
      auto neigh_it = neighbor_list.begin() + pos + 1;
      for (size_t i = 0; i < num_edges; i++) {
        dgl_id_t neigh = *(neigh_it + i);
        col_list_out[collected_nedges + i] = layer_ver_maps[layer_id + 1][neigh];
      }
      // We can simply copy the edge Ids.
      std::copy_n(neighbor_list.begin() + pos + num_edges + 1,
                  num_edges,
                  val_list_out + collected_nedges);
      collected_nedges += num_edges;
      indptr_out[i+1] = indptr_out[i] + num_edges;
      last_off = indptr_out[i+1];
    }
  }

  for (size_t i = layer_offsets[num_hops - 1]; i < subg_csr->indptr.size(); i++)
    indptr_out[i] = last_off;

  // Copy flow offsets.
  out_flow = 0;
  for (size_t i = 0; i < layer_offsets.size() - 2; i++) {
    size_t num_edges = subg_csr->GetDegree(layer_offsets[i + 1], layer_offsets[i]);
    assert(i + 1 < num_hops);
    out_flow[i + 1] = out_flow[i] + num_edges;
  }

  for (size_t i = 0; i < subg_csr->edge_ids.size(); i++)
    subg_csr->edge_ids[i] = i;

  if (neigh_type == "in")
    nf.graph = GraphPtr(new ImmutableGraph(subg_csr, nullptr, IsMultigraph()));
  else
    nf.graph = GraphPtr(new ImmutableGraph(nullptr, subg_csr, IsMultigraph()));

  return nf;
}

NodeFlow ImmutableGraph::NeighborUniformSample(IdArray seeds,
                                               const std::string &neigh_type,
                                               int num_hops, int expand_factor) const {
  return SampleSubgraph(seeds,                 // seed vector
                        nullptr,               // sample_id_probability
                        neigh_type,
                        num_hops + 1,
                        expand_factor);
}

}  // namespace dgl
