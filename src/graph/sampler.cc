/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampler.cc
 * \brief DGL sampler implementation
 */
#include <dgl/sampler.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>
#include <dgl/random.h>
#include <dmlc/omp.h>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include "../c_api_common.h"
#include "../array/common.h"  // for ATEN_FLOAT_TYPE_SWITCH

using namespace dgl::runtime;

namespace dgl {

namespace {
/*
 * ArrayHeap is used to sample elements from vector
 */
template<typename ValueType>
class ArrayHeap {
 public:
  explicit ArrayHeap(const std::vector<ValueType>& prob) {
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
    ValueType w = heap_[i];
    for (int j = bit_len_; j >= 0; --j) {
      heap_[i] -= w;
      i = i >> 1;
    }
  }

  /*
   * Add value w to index (this costs O(log m) steps)
   */
  void Add(size_t index, ValueType w) {
    size_t i = index + limit_;
    for (int j = bit_len_; j >= 0; --j) {
      heap_[i] += w;
      i = i >> 1;
    }
  }

  /*
   * Sample from arrayHeap
   */
  size_t Sample() {
    ValueType xi = heap_[1] * RandomEngine::ThreadLocal()->Uniform<float>();
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
  void SampleWithoutReplacement(size_t n, std::vector<size_t>* samples) {
    // sample n elements
    for (size_t i = 0; i < n; ++i) {
      samples->at(i) = this->Sample();
      this->Delete(samples->at(i));
    }
  }

 private:
  int vec_size_;  // sample size
  int bit_len_;   // bit size
  int limit_;
  std::vector<ValueType> heap_;
};

/*
 * Uniformly sample integers from [0, set_size) without replacement.
 */
void RandomSample(size_t set_size, size_t num, std::vector<size_t>* out) {
  if (num < set_size) {
    std::unordered_set<size_t> sampled_idxs;
    while (sampled_idxs.size() < num) {
      sampled_idxs.insert(RandomEngine::ThreadLocal()->RandInt(set_size));
    }
    out->insert(out->end(), sampled_idxs.begin(), sampled_idxs.end());
  } else {
    // If we need to sample all elements in the set, we don't need to
    // generate random numbers.
    for (size_t i = 0; i < set_size; i++)
      out->push_back(i);
  }
}

void RandomSample(size_t set_size, size_t num, const std::vector<size_t> &exclude,
                  std::vector<size_t>* out) {
  std::unordered_map<size_t, int> sampled_idxs;
  for (auto v : exclude) {
    sampled_idxs.insert(std::pair<size_t, int>(v, 0));
  }
  if (num + exclude.size() < set_size) {
    while (sampled_idxs.size() < num + exclude.size()) {
      size_t rand = RandomEngine::ThreadLocal()->RandInt(set_size);
      sampled_idxs.insert(std::pair<size_t, int>(rand, 1));
    }
    for (auto it = sampled_idxs.begin(); it != sampled_idxs.end(); it++) {
      if (it->second) {
        out->push_back(it->first);
      }
    }
  } else {
    // If we need to sample all elements in the set, we don't need to
    // generate random numbers.
    for (size_t i = 0; i < set_size; i++) {
      // If the element doesn't exist in exclude.
      if (sampled_idxs.find(i) == sampled_idxs.end()) {
        out->push_back(i);
      }
    }
  }
}

/*
 * For a sparse array whose non-zeros are represented by nz_idxs,
 * negate the sparse array and outputs the non-zeros in the negated array.
 */
void NegateArray(const std::vector<size_t> &nz_idxs,
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
void GetUniformSample(const dgl_id_t* edge_id_list,
                      const dgl_id_t* vid_list,
                      const size_t ver_len,
                      const size_t max_num_neighbor,
                      std::vector<dgl_id_t>* out_ver,
                      std::vector<dgl_id_t>* out_edge) {
  // Copy vid_list to output
  if (ver_len <= max_num_neighbor) {
    out_ver->insert(out_ver->end(), vid_list, vid_list + ver_len);
    out_edge->insert(out_edge->end(), edge_id_list, edge_id_list + ver_len);
    return;
  }
  // If we just sample a small number of elements from a large neighbor list.
  std::vector<size_t> sorted_idxs;
  if (ver_len > max_num_neighbor * 2) {
    sorted_idxs.reserve(max_num_neighbor);
    RandomSample(ver_len, max_num_neighbor, &sorted_idxs);
    std::sort(sorted_idxs.begin(), sorted_idxs.end());
  } else {
    std::vector<size_t> negate;
    negate.reserve(ver_len - max_num_neighbor);
    RandomSample(ver_len, ver_len - max_num_neighbor, &negate);
    std::sort(negate.begin(), negate.end());
    NegateArray(negate, ver_len, &sorted_idxs);
  }
  // verify the result.
  CHECK_EQ(sorted_idxs.size(), max_num_neighbor);
  for (size_t i = 1; i < sorted_idxs.size(); i++) {
    CHECK_GT(sorted_idxs[i], sorted_idxs[i - 1]);
  }
  for (auto idx : sorted_idxs) {
    out_ver->push_back(vid_list[idx]);
    out_edge->push_back(edge_id_list[idx]);
  }
}

/*
 * Non-uniform sample via ArrayHeap
 *
 * \param probability Transition probability on the entire graph, indexed by edge ID
 */
template<typename ValueType>
void GetNonUniformSample(const ValueType* probability,
                         const dgl_id_t* edge_id_list,
                         const dgl_id_t* vid_list,
                         const size_t ver_len,
                         const size_t max_num_neighbor,
                         std::vector<dgl_id_t>* out_ver,
                         std::vector<dgl_id_t>* out_edge) {
  // Copy vid_list to output
  if (ver_len <= max_num_neighbor) {
    out_ver->insert(out_ver->end(), vid_list, vid_list + ver_len);
    out_edge->insert(out_edge->end(), edge_id_list, edge_id_list + ver_len);
    return;
  }
  // Make sample
  std::vector<size_t> sp_index(max_num_neighbor);
  std::vector<ValueType> sp_prob(ver_len);
  for (size_t i = 0; i < ver_len; ++i) {
    sp_prob[i] = probability[edge_id_list[i]];
  }
  ArrayHeap<ValueType> arrayHeap(sp_prob);
  arrayHeap.SampleWithoutReplacement(max_num_neighbor, &sp_index);
  out_ver->resize(max_num_neighbor);
  out_edge->resize(max_num_neighbor);
  for (size_t i = 0; i < max_num_neighbor; ++i) {
    size_t idx = sp_index[i];
    out_ver->at(i) = vid_list[idx];
    out_edge->at(i) = edge_id_list[idx];
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

struct neighbor_info {
  dgl_id_t id;
  size_t pos;
  size_t num_edges;

  neighbor_info(dgl_id_t id, size_t pos, size_t num_edges) {
    this->id = id;
    this->pos = pos;
    this->num_edges = num_edges;
  }
};

NodeFlow ConstructNodeFlow(std::vector<dgl_id_t> neighbor_list,
                           std::vector<dgl_id_t> edge_list,
                           std::vector<size_t> layer_offsets,
                           std::vector<std::pair<dgl_id_t, int> > *sub_vers,
                           std::vector<neighbor_info> *neigh_pos,
                           const std::string &edge_type,
                           int64_t num_edges, int num_hops, bool is_multigraph) {
  NodeFlow nf = NodeFlow::Create();
  uint64_t num_vertices = sub_vers->size();
  nf->node_mapping = aten::NewIdArray(num_vertices);
  nf->edge_mapping = aten::NewIdArray(num_edges);
  nf->layer_offsets = aten::NewIdArray(num_hops + 1);
  nf->flow_offsets = aten::NewIdArray(num_hops);

  dgl_id_t *node_map_data = static_cast<dgl_id_t *>(nf->node_mapping->data);
  dgl_id_t *layer_off_data = static_cast<dgl_id_t *>(nf->layer_offsets->data);
  dgl_id_t *flow_off_data = static_cast<dgl_id_t *>(nf->flow_offsets->data);
  dgl_id_t *edge_map_data = static_cast<dgl_id_t *>(nf->edge_mapping->data);

  // Construct sub_csr_graph
  // TODO(minjie): is nodeflow a multigraph?
  auto subg_csr = CSRPtr(new CSR(num_vertices, num_edges, is_multigraph));
  dgl_id_t* indptr_out = static_cast<dgl_id_t*>(subg_csr->indptr()->data);
  dgl_id_t* col_list_out = static_cast<dgl_id_t*>(subg_csr->indices()->data);
  dgl_id_t* eid_out = static_cast<dgl_id_t*>(subg_csr->edge_ids()->data);
  size_t collected_nedges = 0;

  // The data from the previous steps:
  // * node data: sub_vers (vid, layer), neigh_pos,
  // * edge data: neighbor_list, edge_list, probability.
  // * layer_offsets: the offset in sub_vers.
  dgl_id_t ver_id = 0;
  std::vector<std::unordered_map<dgl_id_t, dgl_id_t>> layer_ver_maps;
  layer_ver_maps.resize(num_hops);
  size_t out_node_idx = 0;
  for (int layer_id = num_hops - 1; layer_id >= 0; layer_id--) {
    // We sort the vertices in a layer so that we don't need to sort the neighbor Ids
    // after remap to a subgraph. However, we don't need to sort the first layer
    // because we want the order of the nodes in the first layer is the same as
    // the input seed nodes.
    if (layer_id > 0) {
      std::sort(sub_vers->begin() + layer_offsets[layer_id],
                sub_vers->begin() + layer_offsets[layer_id + 1],
                [](const std::pair<dgl_id_t, dgl_id_t> &a1,
                   const std::pair<dgl_id_t, dgl_id_t> &a2) {
        return a1.first < a2.first;
      });
    }

    // Save the sampled vertices and its layer Id.
    for (size_t i = layer_offsets[layer_id]; i < layer_offsets[layer_id + 1]; i++) {
      node_map_data[out_node_idx++] = sub_vers->at(i).first;
      layer_ver_maps[layer_id].insert(std::pair<dgl_id_t, dgl_id_t>(sub_vers->at(i).first,
                                                                    ver_id++));
      CHECK_EQ(sub_vers->at(i).second, layer_id);
    }
  }
  CHECK(out_node_idx == num_vertices);

  // sampling algorithms have to start from the seed nodes, so the seed nodes are
  // in the first layer and the input nodes are in the last layer.
  // When we expose the sampled graph to a Python user, we say the input nodes
  // are in the first layer and the seed nodes are in the last layer.
  // Thus, when we copy sampled results to a CSR, we need to reverse the order of layers.
  std::fill(indptr_out, indptr_out + num_vertices + 1, 0);
  size_t row_idx = layer_offsets[num_hops] - layer_offsets[num_hops - 1];
  layer_off_data[0] = 0;
  layer_off_data[1] = layer_offsets[num_hops] - layer_offsets[num_hops - 1];
  int out_layer_idx = 1;
  for (int layer_id = num_hops - 2; layer_id >= 0; layer_id--) {
    // Because we don't sort the vertices in the first layer above, we can't sort
    // the neighbor positions of the vertices in the first layer either.
    if (layer_id > 0) {
      std::sort(neigh_pos->begin() + layer_offsets[layer_id],
                neigh_pos->begin() + layer_offsets[layer_id + 1],
                [](const neighbor_info &a1, const neighbor_info &a2) {
                  return a1.id < a2.id;
                });
    }

    for (size_t i = layer_offsets[layer_id]; i < layer_offsets[layer_id + 1]; i++) {
      dgl_id_t dst_id = sub_vers->at(i).first;
      CHECK_EQ(dst_id, neigh_pos->at(i).id);
      size_t pos = neigh_pos->at(i).pos;
      CHECK_LE(pos, neighbor_list.size());
      const size_t nedges = neigh_pos->at(i).num_edges;
      if (neighbor_list.empty()) CHECK_EQ(nedges, 0);

      // We need to map the Ids of the neighbors to the subgraph.
      auto neigh_it = neighbor_list.begin() + pos;
      for (size_t i = 0; i < nedges; i++) {
        dgl_id_t neigh = *(neigh_it + i);
        CHECK(layer_ver_maps[layer_id + 1].find(neigh) != layer_ver_maps[layer_id + 1].end());
        col_list_out[collected_nedges + i] = layer_ver_maps[layer_id + 1][neigh];
      }
      // We can simply copy the edge Ids.
      std::copy_n(edge_list.begin() + pos,
                  nedges, edge_map_data + collected_nedges);
      collected_nedges += nedges;
      indptr_out[row_idx+1] = indptr_out[row_idx] + nedges;
      row_idx++;
    }
    layer_off_data[out_layer_idx + 1] = layer_off_data[out_layer_idx]
        + layer_offsets[layer_id + 1] - layer_offsets[layer_id];
    out_layer_idx++;
  }
  CHECK_EQ(row_idx, num_vertices);
  CHECK_EQ(indptr_out[row_idx], num_edges);
  CHECK_EQ(out_layer_idx, num_hops);
  CHECK_EQ(layer_off_data[out_layer_idx], num_vertices);

  // Copy flow offsets.
  flow_off_data[0] = 0;
  int out_flow_idx = 0;
  for (size_t i = 0; i < layer_offsets.size() - 2; i++) {
    size_t num_edges = indptr_out[layer_off_data[i + 2]] - indptr_out[layer_off_data[i + 1]];
    flow_off_data[out_flow_idx + 1] = flow_off_data[out_flow_idx] + num_edges;
    out_flow_idx++;
  }
  CHECK(out_flow_idx == num_hops - 1);
  CHECK(flow_off_data[num_hops - 1] == static_cast<uint64_t>(num_edges));

  std::iota(eid_out, eid_out + num_edges, 0);

  if (edge_type == std::string("in")) {
    nf->graph = GraphPtr(new ImmutableGraph(subg_csr, nullptr));
  } else {
    nf->graph = GraphPtr(new ImmutableGraph(nullptr, subg_csr));
  }

  return nf;
}

template<typename ValueType>
NodeFlow SampleSubgraph(const ImmutableGraph *graph,
                        const std::vector<dgl_id_t>& seeds,
                        const ValueType* probability,
                        const std::string &edge_type,
                        int num_hops,
                        size_t num_neighbor,
                        const bool add_self_loop) {
  CHECK_EQ(graph->NumBits(), 64) << "32 bit graph is not supported yet";
  const size_t num_seeds = seeds.size();
  auto orig_csr = edge_type == "in" ? graph->GetInCSR() : graph->GetOutCSR();
  const dgl_id_t* val_list = static_cast<dgl_id_t*>(orig_csr->edge_ids()->data);
  const dgl_id_t* col_list = static_cast<dgl_id_t*>(orig_csr->indices()->data);
  const dgl_id_t* indptr = static_cast<dgl_id_t*>(orig_csr->indptr()->data);

  std::unordered_set<dgl_id_t> sub_ver_map;  // The vertex Ids in a layer.
  std::vector<std::pair<dgl_id_t, int> > sub_vers;
  sub_vers.reserve(num_seeds * 10);
  // add seed vertices
  for (size_t i = 0; i < num_seeds; ++i) {
    auto ret = sub_ver_map.insert(seeds[i]);
    // If the vertex is inserted successfully.
    if (ret.second) {
      sub_vers.emplace_back(seeds[i], 0);
    }
  }
  std::vector<dgl_id_t> tmp_sampled_src_list;
  std::vector<dgl_id_t> tmp_sampled_edge_list;
  // ver_id, position
  std::vector<neighbor_info> neigh_pos;
  neigh_pos.reserve(num_seeds);
  std::vector<dgl_id_t> neighbor_list;
  std::vector<dgl_id_t> edge_list;
  std::vector<size_t> layer_offsets(num_hops + 1);
  int64_t num_edges = 0;

  layer_offsets[0] = 0;
  layer_offsets[1] = sub_vers.size();
  for (int layer_id = 1; layer_id < num_hops; layer_id++) {
    // We need to avoid resampling the same node in a layer, but we allow a node
    // to be resampled in multiple layers. We use `sub_ver_map` to keep track of
    // sampled nodes in a layer, and clear it when entering a new layer.
    sub_ver_map.clear();
    // Previous iteration collects all nodes in sub_vers, which are collected
    // in the previous layer. sub_vers is used both as a node collection and a queue.
    for (size_t idx = layer_offsets[layer_id - 1]; idx < layer_offsets[layer_id]; idx++) {
      dgl_id_t dst_id = sub_vers[idx].first;
      const int cur_node_level = sub_vers[idx].second;

      tmp_sampled_src_list.clear();
      tmp_sampled_edge_list.clear();
      dgl_id_t ver_len = *(indptr+dst_id+1) - *(indptr+dst_id);
      if (probability == nullptr) {  // uniform-sample
        GetUniformSample(val_list + *(indptr + dst_id),
                         col_list + *(indptr + dst_id),
                         ver_len,
                         num_neighbor,
                         &tmp_sampled_src_list,
                         &tmp_sampled_edge_list);
      } else {  // non-uniform-sample
        GetNonUniformSample(probability,
                            val_list + *(indptr + dst_id),
                            col_list + *(indptr + dst_id),
                            ver_len,
                            num_neighbor,
                            &tmp_sampled_src_list,
                            &tmp_sampled_edge_list);
      }
      // If we need to add self loop and it doesn't exist in the sampled neighbor list.
      if (add_self_loop && std::find(tmp_sampled_src_list.begin(), tmp_sampled_src_list.end(),
                                     dst_id) == tmp_sampled_src_list.end()) {
        tmp_sampled_src_list.push_back(dst_id);
        const dgl_id_t *src_list = col_list + *(indptr + dst_id);
        const dgl_id_t *eid_list = val_list + *(indptr + dst_id);
        // TODO(zhengda) this operation has O(N) complexity. It can be pretty slow.
        const dgl_id_t *src = std::find(src_list, src_list + ver_len, dst_id);
        // If there doesn't exist a self loop in the graph.
        // we have to add -1 as the edge id for the self-loop edge.
        if (src == src_list + ver_len)
          tmp_sampled_edge_list.push_back(-1);
        else
          tmp_sampled_edge_list.push_back(eid_list[src - src_list]);
      }
      CHECK_EQ(tmp_sampled_src_list.size(), tmp_sampled_edge_list.size());
      neigh_pos.emplace_back(dst_id, neighbor_list.size(), tmp_sampled_src_list.size());
      // Then push the vertices
      for (size_t i = 0; i < tmp_sampled_src_list.size(); ++i) {
        neighbor_list.push_back(tmp_sampled_src_list[i]);
      }
      // Finally we push the edge list
      for (size_t i = 0; i < tmp_sampled_edge_list.size(); ++i) {
        edge_list.push_back(tmp_sampled_edge_list[i]);
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
    CHECK_EQ(layer_offsets[layer_id + 1], sub_vers.size());
  }

  return ConstructNodeFlow(neighbor_list, edge_list, layer_offsets, &sub_vers, &neigh_pos,
                           edge_type, num_edges, num_hops, graph->IsMultigraph());
}

}  // namespace

DGL_REGISTER_GLOBAL("nodeflow._CAPI_NodeFlowGetGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NodeFlow nflow = args[0];
    *rv = nflow->graph;
  });

DGL_REGISTER_GLOBAL("nodeflow._CAPI_NodeFlowGetNodeMapping")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NodeFlow nflow = args[0];
    *rv = nflow->node_mapping;
  });

DGL_REGISTER_GLOBAL("nodeflow._CAPI_NodeFlowGetEdgeMapping")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NodeFlow nflow = args[0];
    *rv = nflow->edge_mapping;
  });

DGL_REGISTER_GLOBAL("nodeflow._CAPI_NodeFlowGetLayerOffsets")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NodeFlow nflow = args[0];
    *rv = nflow->layer_offsets;
  });

DGL_REGISTER_GLOBAL("nodeflow._CAPI_NodeFlowGetBlockOffsets")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NodeFlow nflow = args[0];
    *rv = nflow->flow_offsets;
  });

template<typename ValueType>
NodeFlow SamplerOp::NeighborSample(const ImmutableGraph *graph,
                                   const std::vector<dgl_id_t>& seeds,
                                   const std::string &edge_type,
                                   int num_hops, int expand_factor,
                                   const bool add_self_loop,
                                   const ValueType *probability) {
  return SampleSubgraph(graph,
                        seeds,
                        probability,
                        edge_type,
                        num_hops + 1,
                        expand_factor,
                        add_self_loop);
}

namespace {
  void ConstructLayers(const dgl_id_t *indptr,
                       const dgl_id_t *indices,
                       const std::vector<dgl_id_t>& seed_array,
                       IdArray layer_sizes,
                       std::vector<dgl_id_t> *layer_offsets,
                       std::vector<dgl_id_t> *node_mapping,
                       std::vector<int64_t> *actl_layer_sizes,
                       std::vector<float> *probabilities) {
    /*
     * Given a graph and a collection of seed nodes, this function constructs NodeFlow
     * layers via uniform layer-wise sampling, and return the resultant layers and their
     * corresponding probabilities.
     */
    std::copy(seed_array.begin(), seed_array.end(), std::back_inserter(*node_mapping));
    actl_layer_sizes->push_back(node_mapping->size());
    probabilities->insert(probabilities->end(), node_mapping->size(), 1);
    const int64_t* layer_sizes_data = static_cast<int64_t*>(layer_sizes->data);
    const int64_t num_layers = layer_sizes->shape[0];

    size_t curr = 0;
    size_t next = node_mapping->size();
    for (int64_t i = num_layers - 1; i >= 0; --i) {
      const int64_t layer_size = layer_sizes_data[i];
      std::unordered_set<dgl_id_t> candidate_set;
      for (auto j = curr; j != next; ++j) {
        auto src = (*node_mapping)[j];
        candidate_set.insert(indices + indptr[src], indices + indptr[src + 1]);
      }

      std::vector<dgl_id_t> candidate_vector;
      std::copy(candidate_set.begin(), candidate_set.end(),
                std::back_inserter(candidate_vector));

      std::unordered_map<dgl_id_t, size_t> n_occurrences;
      auto n_candidates = candidate_vector.size();
      for (int64_t j = 0; j != layer_size; ++j) {
        auto dst = candidate_vector[
          RandomEngine::ThreadLocal()->RandInt(n_candidates)];
        if (!n_occurrences.insert(std::make_pair(dst, 1)).second) {
          ++n_occurrences[dst];
        }
      }

      for (auto const &pair : n_occurrences) {
        node_mapping->push_back(pair.first);
        float p = pair.second * n_candidates / static_cast<float>(layer_size);
        probabilities->push_back(p);
      }

      actl_layer_sizes->push_back(node_mapping->size() - next);
      curr = next;
      next = node_mapping->size();
    }
    std::reverse(node_mapping->begin(), node_mapping->end());
    std::reverse(actl_layer_sizes->begin(), actl_layer_sizes->end());
    layer_offsets->push_back(0);
    for (const auto &size : *actl_layer_sizes) {
      layer_offsets->push_back(size + layer_offsets->back());
    }
  }

  void ConstructFlows(const dgl_id_t *indptr,
                      const dgl_id_t *indices,
                      const dgl_id_t *eids,
                      const std::vector<dgl_id_t> &node_mapping,
                      const std::vector<int64_t> &actl_layer_sizes,
                      std::vector<dgl_id_t> *sub_indptr,
                      std::vector<dgl_id_t> *sub_indices,
                      std::vector<dgl_id_t> *sub_eids,
                      std::vector<dgl_id_t> *flow_offsets,
                      std::vector<dgl_id_t> *edge_mapping) {
    /*
     * Given a graph and a sequence of NodeFlow layers, this function constructs dense
     * subgraphs (flows) between consecutive layers.
     */
    auto n_flows = actl_layer_sizes.size() - 1;
    for (int64_t i = 0; i < actl_layer_sizes.front() + 1; i++)
      sub_indptr->push_back(0);
    flow_offsets->push_back(0);
    int64_t first = 0;
    for (size_t i = 0; i < n_flows; ++i) {
      auto src_size = actl_layer_sizes[i];
      std::unordered_map<dgl_id_t, dgl_id_t> source_map;
      for (int64_t j = 0; j < src_size; ++j) {
        source_map.insert(std::make_pair(node_mapping[first + j], first + j));
      }
      auto dst_size = actl_layer_sizes[i + 1];
      for (int64_t j = 0; j < dst_size; ++j) {
        auto dst = node_mapping[first + src_size + j];
        typedef std::pair<dgl_id_t, dgl_id_t> id_pair;
        std::vector<id_pair> neighbor_indices;
        for (dgl_id_t k = indptr[dst]; k < indptr[dst + 1]; ++k) {
          // TODO(gaiyu): accelerate hash table lookup
          auto ret = source_map.find(indices[k]);
          if (ret != source_map.end()) {
            neighbor_indices.push_back(std::make_pair(ret->second, eids[k]));
          }
        }
        auto cmp = [](const id_pair p, const id_pair q)->bool { return p.first < q.first; };
        std::sort(neighbor_indices.begin(), neighbor_indices.end(), cmp);
        for (const auto &pair : neighbor_indices) {
          sub_indices->push_back(pair.first);
          edge_mapping->push_back(pair.second);
        }
        sub_indptr->push_back(sub_indices->size());
      }
      flow_offsets->push_back(sub_indices->size());
      first += src_size;
    }
    sub_eids->resize(sub_indices->size());
    std::iota(sub_eids->begin(), sub_eids->end(), 0);
  }
}  // namespace

NodeFlow SamplerOp::LayerUniformSample(const ImmutableGraph *graph,
                                       const std::vector<dgl_id_t>& seeds,
                                       const std::string &neighbor_type,
                                       IdArray layer_sizes) {
  const auto g_csr = neighbor_type == "in" ? graph->GetInCSR() : graph->GetOutCSR();
  const dgl_id_t *indptr = static_cast<dgl_id_t*>(g_csr->indptr()->data);
  const dgl_id_t *indices = static_cast<dgl_id_t*>(g_csr->indices()->data);
  const dgl_id_t *eids = static_cast<dgl_id_t*>(g_csr->edge_ids()->data);

  std::vector<dgl_id_t> layer_offsets;
  std::vector<dgl_id_t> node_mapping;
  std::vector<int64_t> actl_layer_sizes;
  std::vector<float> probabilities;
  ConstructLayers(indptr,
                  indices,
                  seeds,
                  layer_sizes,
                  &layer_offsets,
                  &node_mapping,
                  &actl_layer_sizes,
                  &probabilities);

  std::vector<dgl_id_t> sub_indptr, sub_indices, sub_edge_ids;
  std::vector<dgl_id_t> flow_offsets;
  std::vector<dgl_id_t> edge_mapping;
  ConstructFlows(indptr,
                 indices,
                 eids,
                 node_mapping,
                 actl_layer_sizes,
                 &sub_indptr,
                 &sub_indices,
                 &sub_edge_ids,
                 &flow_offsets,
                 &edge_mapping);
  // sanity check
  CHECK_GT(sub_indptr.size(), 0);
  CHECK_EQ(sub_indptr[0], 0);
  CHECK_EQ(sub_indptr.back(), sub_indices.size());
  CHECK_EQ(sub_indices.size(), sub_edge_ids.size());

  NodeFlow nf = NodeFlow::Create();
  auto sub_csr = CSRPtr(new CSR(aten::VecToIdArray(sub_indptr),
                                aten::VecToIdArray(sub_indices),
                                aten::VecToIdArray(sub_edge_ids)));

  if (neighbor_type == std::string("in")) {
    nf->graph = GraphPtr(new ImmutableGraph(sub_csr, nullptr));
  } else {
    nf->graph = GraphPtr(new ImmutableGraph(nullptr, sub_csr));
  }

  nf->node_mapping = aten::VecToIdArray(node_mapping);
  nf->edge_mapping = aten::VecToIdArray(edge_mapping);
  nf->layer_offsets = aten::VecToIdArray(layer_offsets);
  nf->flow_offsets = aten::VecToIdArray(flow_offsets);

  return nf;
}

void BuildCsr(const ImmutableGraph &g, const std::string neigh_type) {
  if (neigh_type == "in") {
    auto csr = g.GetInCSR();
    assert(csr);
  } else if (neigh_type == "out") {
    auto csr = g.GetOutCSR();
    assert(csr);
  } else {
    LOG(FATAL) << "We don't support sample from neighbor type " << neigh_type;
  }
}

template<typename ValueType>
std::vector<NodeFlow> NeighborSamplingImpl(const ImmutableGraphPtr gptr,
                                           const IdArray seed_nodes,
                                           const int64_t batch_start_id,
                                           const int64_t batch_size,
                                           const int64_t max_num_workers,
                                           const int64_t expand_factor,
                                           const int64_t num_hops,
                                           const std::string neigh_type,
                                           const bool add_self_loop,
                                           const ValueType *probability) {
    // process args
    CHECK(aten::IsValidIdArray(seed_nodes));
    const dgl_id_t* seed_nodes_data = static_cast<dgl_id_t*>(seed_nodes->data);
    const int64_t num_seeds = seed_nodes->shape[0];
    const int64_t num_workers = std::min(max_num_workers,
        (num_seeds + batch_size - 1) / batch_size - batch_start_id);
    // We need to make sure we have the right CSR before we enter parallel sampling.
    BuildCsr(*gptr, neigh_type);
    // generate node flows
    std::vector<NodeFlow> nflows(num_workers);
#pragma omp parallel for
    for (int i = 0; i < num_workers; i++) {
      // create per-worker seed nodes.
      const int64_t start = (batch_start_id + i) * batch_size;
      const int64_t end = std::min(start + batch_size, num_seeds);
      // TODO(minjie): the vector allocation/copy is unnecessary
      std::vector<dgl_id_t> worker_seeds(end - start);
      std::copy(seed_nodes_data + start, seed_nodes_data + end,
                worker_seeds.begin());
      nflows[i] = SamplerOp::NeighborSample(
          gptr.get(), worker_seeds, neigh_type, num_hops, expand_factor,
          add_self_loop, probability);
    }
    return nflows;
}

DGL_REGISTER_GLOBAL("sampling._CAPI_UniformSampling")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    // arguments
    const GraphRef g = args[0];
    const IdArray seed_nodes = args[1];
    const int64_t batch_start_id = args[2];
    const int64_t batch_size = args[3];
    const int64_t max_num_workers = args[4];
    const int64_t expand_factor = args[5];
    const int64_t num_hops = args[6];
    const std::string neigh_type = args[7];
    const bool add_self_loop = args[8];

    auto gptr = std::dynamic_pointer_cast<ImmutableGraph>(g.sptr());
    CHECK(gptr) << "sampling isn't implemented in mutable graph";

    std::vector<NodeFlow> nflows = NeighborSamplingImpl<float>(
        gptr, seed_nodes, batch_start_id, batch_size, max_num_workers,
        expand_factor, num_hops, neigh_type, add_self_loop, nullptr);

    *rv = List<NodeFlow>(nflows);
  });

DGL_REGISTER_GLOBAL("sampling._CAPI_NeighborSampling")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    // arguments
    const GraphRef g = args[0];
    const IdArray seed_nodes = args[1];
    const int64_t batch_start_id = args[2];
    const int64_t batch_size = args[3];
    const int64_t max_num_workers = args[4];
    const int64_t expand_factor = args[5];
    const int64_t num_hops = args[6];
    const std::string neigh_type = args[7];
    const bool add_self_loop = args[8];
    const NDArray probability = args[9];

    auto gptr = std::dynamic_pointer_cast<ImmutableGraph>(g.sptr());
    CHECK(gptr) << "sampling isn't implemented in mutable graph";

    std::vector<NodeFlow> nflows;

    CHECK(probability->dtype.code == kDLFloat)
      << "transition probability must be float";
    CHECK(probability->ndim == 1)
      << "transition probability must be a 1-dimensional vector";

    ATEN_FLOAT_TYPE_SWITCH(
      probability->dtype,
      FloatType,
      "transition probability",
      {
        const FloatType *prob;

        if (probability->ndim == 1 && probability->shape[0] == 0) {
          prob = nullptr;
        } else {
          CHECK(probability->shape[0] == gptr->NumEdges())
            << "transition probability must have same number of elements as edges";
          CHECK(probability.IsContiguous())
            << "transition probability must be contiguous tensor";
          prob = static_cast<const FloatType *>(probability->data);
        }

        nflows = NeighborSamplingImpl(
            gptr, seed_nodes, batch_start_id, batch_size, max_num_workers,
            expand_factor, num_hops, neigh_type, add_self_loop, prob);
    });

    *rv = List<NodeFlow>(nflows);
  });

DGL_REGISTER_GLOBAL("sampling._CAPI_LayerSampling")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    // arguments
    GraphRef g = args[0];
    const IdArray seed_nodes = args[1];
    const int64_t batch_start_id = args[2];
    const int64_t batch_size = args[3];
    const int64_t max_num_workers = args[4];
    const IdArray layer_sizes = args[5];
    const std::string neigh_type = args[6];
    // process args
    auto gptr = std::dynamic_pointer_cast<ImmutableGraph>(g.sptr());
    CHECK(gptr) << "sampling isn't implemented in mutable graph";
    CHECK(aten::IsValidIdArray(seed_nodes));
    const dgl_id_t* seed_nodes_data = static_cast<dgl_id_t*>(seed_nodes->data);
    const int64_t num_seeds = seed_nodes->shape[0];
    const int64_t num_workers = std::min(max_num_workers,
        (num_seeds + batch_size - 1) / batch_size - batch_start_id);
    // We need to make sure we have the right CSR before we enter parallel sampling.
    BuildCsr(*gptr, neigh_type);
    // generate node flows
    std::vector<NodeFlow> nflows(num_workers);
#pragma omp parallel for
    for (int i = 0; i < num_workers; i++) {
      // create per-worker seed nodes.
      const int64_t start = (batch_start_id + i) * batch_size;
      const int64_t end = std::min(start + batch_size, num_seeds);
      // TODO(minjie): the vector allocation/copy is unnecessary
      std::vector<dgl_id_t> worker_seeds(end - start);
      std::copy(seed_nodes_data + start, seed_nodes_data + end,
                worker_seeds.begin());
      nflows[i] = SamplerOp::LayerUniformSample(
          gptr.get(), worker_seeds, neigh_type, layer_sizes);
    }
    *rv = List<NodeFlow>(nflows);
  });

namespace {

void BuildCoo(const ImmutableGraph &g) {
  auto coo = g.GetCOO();
  assert(coo);
}


dgl_id_t global2local_map(dgl_id_t global_id,
                          std::unordered_map<dgl_id_t, dgl_id_t> *map) {
  auto it = map->find(global_id);
  if (it == map->end()) {
    dgl_id_t local_id = map->size();
    map->insert(std::pair<dgl_id_t, dgl_id_t>(global_id, local_id));
    return local_id;
  } else {
    return it->second;
  }
}

inline bool IsNegativeHeadMode(const std::string &mode) {
  return mode == "head";
}

IdArray GetGlobalVid(IdArray induced_nid, IdArray subg_nid) {
  IdArray gnid = IdArray::Empty({subg_nid->shape[0]}, subg_nid->dtype, subg_nid->ctx);
  const dgl_id_t *induced_nid_data = static_cast<dgl_id_t *>(induced_nid->data);
  const dgl_id_t *subg_nid_data = static_cast<dgl_id_t *>(subg_nid->data);
  dgl_id_t *gnid_data = static_cast<dgl_id_t *>(gnid->data);
  for (int64_t i = 0; i < subg_nid->shape[0]; i++) {
    gnid_data[i] = induced_nid_data[subg_nid_data[i]];
  }
  return gnid;
}

IdArray CheckExistence(GraphPtr gptr, IdArray neg_src, IdArray neg_dst,
                       IdArray induced_nid) {
  return gptr->HasEdgesBetween(GetGlobalVid(induced_nid, neg_src),
                               GetGlobalVid(induced_nid, neg_dst));
}

IdArray CheckExistence(GraphPtr gptr, IdArray relations,
                       IdArray neg_src, IdArray neg_dst,
                       IdArray induced_nid, IdArray neg_eid) {
  neg_src = GetGlobalVid(induced_nid, neg_src);
  neg_dst = GetGlobalVid(induced_nid, neg_dst);
  BoolArray exist = gptr->HasEdgesBetween(neg_src, neg_dst);
  dgl_id_t *neg_dst_data = static_cast<dgl_id_t *>(neg_dst->data);
  dgl_id_t *neg_src_data = static_cast<dgl_id_t *>(neg_src->data);
  dgl_id_t *neg_eid_data = static_cast<dgl_id_t *>(neg_eid->data);
  dgl_id_t *relation_data = static_cast<dgl_id_t *>(relations->data);
  // TODO(zhengda) is this right?
  dgl_id_t *exist_data = static_cast<dgl_id_t *>(exist->data);
  int64_t num_neg_edges = neg_src->shape[0];
  for (int64_t i = 0; i < num_neg_edges; i++) {
    // If the edge doesn't exist, we don't need to do anything.
    if (!exist_data[i])
      continue;
    // If the edge exists, we need to double check if the relations match.
    // If they match, this negative edge isn't really a negative edge.
    dgl_id_t eid1 = neg_eid_data[i];
    dgl_id_t orig_neg_rel1 = relation_data[eid1];
    IdArray eids = gptr->EdgeId(neg_src_data[i], neg_dst_data[i]);
    dgl_id_t *eid_data = static_cast<dgl_id_t *>(eids->data);
    int64_t num_edges_between = eids->shape[0];
    bool same_rel = false;
    for (int64_t j = 0; j < num_edges_between; j++) {
      dgl_id_t neg_rel1 = relation_data[eid_data[j]];
      if (neg_rel1 == orig_neg_rel1) {
        same_rel = true;
        break;
      }
    }
    exist_data[i] = same_rel;
  }
  return exist;
}

std::vector<dgl_id_t> Global2Local(const std::vector<size_t> &ids,
                                   const std::unordered_map<dgl_id_t, dgl_id_t> &map) {
  std::vector<dgl_id_t> local_ids(ids.size());
  for (size_t i = 0; i < ids.size(); i++) {
    auto it = map.find(ids[i]);
    assert(it != map.end());
    local_ids[i] = it->second;
  }
  return local_ids;
}

NegSubgraph NegEdgeSubgraph(GraphPtr gptr, IdArray relations, const Subgraph &pos_subg,
                            const std::string &neg_mode,
                            int neg_sample_size, bool exclude_positive,
                            bool check_false_neg) {
  int64_t num_tot_nodes = gptr->NumVertices();
  bool is_multigraph = gptr->IsMultigraph();
  std::vector<IdArray> adj = pos_subg.graph->GetAdj(false, "coo");
  IdArray coo = adj[0];
  int64_t num_pos_edges = coo->shape[0] / 2;
  int64_t num_neg_edges = num_pos_edges * neg_sample_size;
  IdArray neg_dst = IdArray::Empty({num_neg_edges}, coo->dtype, coo->ctx);
  IdArray neg_src = IdArray::Empty({num_neg_edges}, coo->dtype, coo->ctx);
  IdArray neg_eid = IdArray::Empty({num_neg_edges}, coo->dtype, coo->ctx);
  IdArray induced_neg_eid = IdArray::Empty({num_neg_edges}, coo->dtype, coo->ctx);

  // These are vids in the positive subgraph.
  const dgl_id_t *dst_data = static_cast<const dgl_id_t *>(coo->data);
  const dgl_id_t *src_data = static_cast<const dgl_id_t *>(coo->data) + num_pos_edges;
  const dgl_id_t *induced_vid_data = static_cast<const dgl_id_t *>(pos_subg.induced_vertices->data);
  const dgl_id_t *induced_eid_data = static_cast<const dgl_id_t *>(pos_subg.induced_edges->data);
  size_t num_pos_nodes = pos_subg.graph->NumVertices();
  std::vector<size_t> pos_nodes(induced_vid_data, induced_vid_data + num_pos_nodes);

  dgl_id_t *neg_dst_data = static_cast<dgl_id_t *>(neg_dst->data);
  dgl_id_t *neg_src_data = static_cast<dgl_id_t *>(neg_src->data);
  dgl_id_t *neg_eid_data = static_cast<dgl_id_t *>(neg_eid->data);
  dgl_id_t *induced_neg_eid_data = static_cast<dgl_id_t *>(induced_neg_eid->data);

  const dgl_id_t *unchanged;
  dgl_id_t *neg_unchanged;
  dgl_id_t *neg_changed;
  if (IsNegativeHeadMode(neg_mode)) {
    unchanged = dst_data;
    neg_unchanged = neg_dst_data;
    neg_changed = neg_src_data;
  } else {
    unchanged = src_data;
    neg_unchanged = neg_src_data;
    neg_changed = neg_dst_data;
  }

  std::unordered_map<dgl_id_t, dgl_id_t> neg_map;
  std::vector<dgl_id_t> local_pos_vids;
  local_pos_vids.reserve(num_pos_edges);

  dgl_id_t curr_eid = 0;
  std::vector<size_t> neg_vids;
  neg_vids.reserve(neg_sample_size);
  // If we don't exclude positive edges, we are actually sampling more than
  // the total number of nodes in the graph.
  if (!exclude_positive && neg_sample_size >= num_tot_nodes) {
    // We add all nodes as negative nodes.
    for (int64_t i = 0; i < num_tot_nodes; i++) {
      neg_vids.push_back(i);
      neg_map[i] = i;
    }

    // Get all nodes in the positive side.
    for (int64_t i = 0; i < num_pos_edges; i++) {
      dgl_id_t vid = induced_vid_data[unchanged[i]];
      local_pos_vids.push_back(neg_map[vid]);
    }
    // There is no guarantee that the nodes in the vector are unique.
    std::sort(local_pos_vids.begin(), local_pos_vids.end());
    auto it = std::unique(local_pos_vids.begin(), local_pos_vids.end());
    local_pos_vids.resize(it - local_pos_vids.begin());
  } else {
    // Collect nodes in the positive side.
    dgl_id_t local_vid = 0;
    for (int64_t i = 0; i < num_pos_edges; i++) {
      dgl_id_t vid = induced_vid_data[unchanged[i]];
      auto it = neg_map.find(vid);
      if (it == neg_map.end()) {
        local_pos_vids.push_back(local_vid);
        neg_map.insert(std::pair<dgl_id_t, dgl_id_t>(vid, local_vid++));
      }
    }
  }

  int64_t prev_neg_offset = 0;
  for (int64_t i = 0; i < num_pos_edges; i++) {
    size_t neg_idx = i * neg_sample_size;

    std::vector<size_t> neighbors;
    DGLIdIters neigh_it;
    if (IsNegativeHeadMode(neg_mode)) {
      neigh_it = gptr->PredVec(induced_vid_data[unchanged[i]]);
    } else {
      neigh_it = gptr->SuccVec(induced_vid_data[unchanged[i]]);
    }

    // If the number of negative nodes is smaller than the number of total nodes
    // in the graph.
    if (exclude_positive && neg_sample_size < num_tot_nodes) {
      std::vector<size_t> exclude;
      for (auto it = neigh_it.begin(); it != neigh_it.end(); it++) {
        dgl_id_t global_vid = *it;
        exclude.push_back(global_vid);
      }
      prev_neg_offset = neg_vids.size();
      RandomSample(num_tot_nodes, neg_sample_size, exclude, &neg_vids);
      assert(prev_neg_offset + neg_sample_size == neg_vids.size());
    } else if (neg_sample_size < num_tot_nodes) {
      prev_neg_offset = neg_vids.size();
      RandomSample(num_tot_nodes, neg_sample_size, &neg_vids);
      assert(prev_neg_offset + neg_sample_size == neg_vids.size());
    } else if (exclude_positive) {
      LOG(FATAL) << "We can't exclude positive edges when sampling negative edges with all nodes.";
    } else {
      // We don't need to do anything here.
      // In this case, every edge has the same negative edges. That is,
      // neg_vids contains all nodes of the graph. They have been generated
      // before the for loop.
    }

    dgl_id_t global_unchanged = induced_vid_data[unchanged[i]];
    dgl_id_t local_unchanged = global2local_map(global_unchanged, &neg_map);

    for (int64_t j = 0; j < neg_sample_size; j++) {
      neg_unchanged[neg_idx + j] = local_unchanged;
      neg_eid_data[neg_idx + j] = curr_eid++;
      dgl_id_t local_changed = global2local_map(neg_vids[j + prev_neg_offset], &neg_map);
      neg_changed[neg_idx + j] = local_changed;
      // induced negative eid references to the positive one.
      induced_neg_eid_data[neg_idx + j] = induced_eid_data[i];
    }
  }

  // Now we know the number of vertices in the negative graph.
  int64_t num_neg_nodes = neg_map.size();
  IdArray induced_neg_vid = IdArray::Empty({num_neg_nodes}, coo->dtype, coo->ctx);
  dgl_id_t *induced_neg_vid_data = static_cast<dgl_id_t *>(induced_neg_vid->data);
  for (auto it = neg_map.begin(); it != neg_map.end(); it++) {
    induced_neg_vid_data[it->second] = it->first;
  }

  NegSubgraph neg_subg;
  // We sample negative vertices without replacement.
  // There shouldn't be duplicated edges.
  COOPtr neg_coo(new COO(num_neg_nodes, neg_src, neg_dst, is_multigraph));
  neg_subg.graph = GraphPtr(new ImmutableGraph(neg_coo));
  neg_subg.induced_vertices = induced_neg_vid;
  neg_subg.induced_edges = induced_neg_eid;
  // If we didn't sample all nodes to form negative edges, some of the nodes
  // in the vector might be redundant.
  if (neg_sample_size < num_tot_nodes) {
    std::sort(neg_vids.begin(), neg_vids.end());
    auto it = std::unique(neg_vids.begin(), neg_vids.end());
    neg_vids.resize(it - neg_vids.begin());
  }
  if (IsNegativeHeadMode(neg_mode)) {
    neg_subg.head_nid = aten::VecToIdArray(Global2Local(neg_vids, neg_map));
    neg_subg.tail_nid = aten::VecToIdArray(local_pos_vids);
  } else {
    neg_subg.head_nid = aten::VecToIdArray(local_pos_vids);
    neg_subg.tail_nid = aten::VecToIdArray(Global2Local(neg_vids, neg_map));
  }
  // TODO(zhengda) we should provide an array of 1s if exclude_positive
  if (check_false_neg) {
    if (relations->shape[0] == 0) {
      neg_subg.exist = CheckExistence(gptr, neg_src, neg_dst, induced_neg_vid);
    } else {
      neg_subg.exist = CheckExistence(gptr, relations, neg_src, neg_dst,
                                      induced_neg_vid, induced_neg_eid);
    }
  }
  return neg_subg;
}

NegSubgraph PBGNegEdgeSubgraph(GraphPtr gptr, IdArray relations, const Subgraph &pos_subg,
                            const std::string &neg_mode,
                            int neg_sample_size, bool is_multigraph,
                            bool exclude_positive, bool check_false_neg) {
  int64_t num_tot_nodes = gptr->NumVertices();
  std::vector<IdArray> adj = pos_subg.graph->GetAdj(false, "coo");
  IdArray coo = adj[0];
  int64_t num_pos_edges = coo->shape[0] / 2;

  int64_t chunk_size = neg_sample_size;
  // If num_pos_edges isn't divisible by chunk_size, the actual number of chunks
  // is num_chunks + 1 and the last chunk size is last_chunk_size.
  // Otherwise, the actual number of chunks is num_chunks, the last chunk size
  // is 0.
  int64_t num_chunks = num_pos_edges / chunk_size;
  int64_t last_chunk_size = num_pos_edges - num_chunks * chunk_size;

  // The number of negative edges.
  int64_t num_neg_edges = neg_sample_size * chunk_size * num_chunks;
  int64_t num_neg_edges_last_chunk = neg_sample_size * last_chunk_size;
  int64_t num_all_neg_edges = num_neg_edges + num_neg_edges_last_chunk;

  // We should include the last chunk.
  if (last_chunk_size > 0)
    num_chunks++;

  IdArray neg_dst = IdArray::Empty({num_all_neg_edges}, coo->dtype, coo->ctx);
  IdArray neg_src = IdArray::Empty({num_all_neg_edges}, coo->dtype, coo->ctx);
  IdArray neg_eid = IdArray::Empty({num_all_neg_edges}, coo->dtype, coo->ctx);
  IdArray induced_neg_eid = IdArray::Empty({num_all_neg_edges}, coo->dtype, coo->ctx);

  // These are vids in the positive subgraph.
  const dgl_id_t *dst_data = static_cast<const dgl_id_t *>(coo->data);
  const dgl_id_t *src_data = static_cast<const dgl_id_t *>(coo->data) + num_pos_edges;
  const dgl_id_t *induced_vid_data = static_cast<const dgl_id_t *>(pos_subg.induced_vertices->data);
  const dgl_id_t *induced_eid_data = static_cast<const dgl_id_t *>(pos_subg.induced_edges->data);
  size_t num_pos_nodes = pos_subg.graph->NumVertices();
  std::vector<size_t> pos_nodes(induced_vid_data, induced_vid_data + num_pos_nodes);

  dgl_id_t *neg_dst_data = static_cast<dgl_id_t *>(neg_dst->data);
  dgl_id_t *neg_src_data = static_cast<dgl_id_t *>(neg_src->data);
  dgl_id_t *neg_eid_data = static_cast<dgl_id_t *>(neg_eid->data);
  dgl_id_t *induced_neg_eid_data = static_cast<dgl_id_t *>(induced_neg_eid->data);

  const dgl_id_t *unchanged;
  dgl_id_t *neg_unchanged;
  dgl_id_t *neg_changed;

  // corrupt head nodes.
  if (IsNegativeHeadMode(neg_mode)) {
    unchanged = dst_data;
    neg_unchanged = neg_dst_data;
    neg_changed = neg_src_data;
  } else {
    // corrupt tail nodes.
    unchanged = src_data;
    neg_unchanged = neg_src_data;
    neg_changed = neg_dst_data;
  }

  // We first sample all negative edges.
  std::vector<size_t> neg_vids;
  RandomSample(num_tot_nodes,
               num_chunks * neg_sample_size,
               &neg_vids);

  dgl_id_t curr_eid = 0;
  std::unordered_map<dgl_id_t, dgl_id_t> neg_map;
  dgl_id_t local_vid = 0;
  // Collect nodes in the positive side.
  std::vector<dgl_id_t> local_pos_vids;
  local_pos_vids.reserve(num_pos_edges);
  for (int64_t i = 0; i < num_pos_edges; i++) {
    dgl_id_t vid = induced_vid_data[unchanged[i]];
    auto it = neg_map.find(vid);
    if (it == neg_map.end()) {
      local_pos_vids.push_back(local_vid);
      neg_map.insert(std::pair<dgl_id_t, dgl_id_t>(vid, local_vid++));
    }
  }

  for (int64_t i_chunk = 0; i_chunk < num_chunks; i_chunk++) {
    // for each chunk.
    int64_t neg_idx = neg_sample_size * chunk_size * i_chunk;
    int64_t pos_edge_idx = chunk_size * i_chunk;
    int64_t neg_node_idx = neg_sample_size * i_chunk;
    // The actual chunk size. It'll be different for the last chunk.
    int64_t chunk_size1;
    if (i_chunk == num_chunks - 1 && last_chunk_size > 0)
      chunk_size1 = last_chunk_size;
    else
      chunk_size1 = chunk_size;

    for (int64_t in_chunk = 0; in_chunk != chunk_size1; ++in_chunk) {
      // For each positive node in a chunk.

      dgl_id_t global_unchanged = induced_vid_data[unchanged[pos_edge_idx + in_chunk]];
      dgl_id_t local_unchanged = global2local_map(global_unchanged, &neg_map);
      for (int64_t j = 0; j < neg_sample_size; ++j) {
        neg_unchanged[neg_idx] = local_unchanged;
        neg_eid_data[neg_idx] = curr_eid++;
        dgl_id_t global_changed_vid = neg_vids[neg_node_idx + j];

        // TODO(zhengda) we can avoid the hashtable lookup here.
        dgl_id_t local_changed = global2local_map(global_changed_vid, &neg_map);
        neg_changed[neg_idx] = local_changed;
        induced_neg_eid_data[neg_idx] = induced_eid_data[pos_edge_idx + in_chunk];
        neg_idx++;
      }
    }
  }

  // Now we know the number of vertices in the negative graph.
  int64_t num_neg_nodes = neg_map.size();
  IdArray induced_neg_vid = IdArray::Empty({num_neg_nodes}, coo->dtype, coo->ctx);
  dgl_id_t *induced_neg_vid_data = static_cast<dgl_id_t *>(induced_neg_vid->data);
  for (auto it = neg_map.begin(); it != neg_map.end(); it++) {
    induced_neg_vid_data[it->second] = it->first;
  }

  NegSubgraph neg_subg;
  // We sample negative vertices without replacement.
  // There shouldn't be duplicated edges.
  COOPtr neg_coo(new COO(num_neg_nodes, neg_src, neg_dst, is_multigraph));
  neg_subg.graph = GraphPtr(new ImmutableGraph(neg_coo));
  neg_subg.induced_vertices = induced_neg_vid;
  neg_subg.induced_edges = induced_neg_eid;
  if (IsNegativeHeadMode(neg_mode)) {
    neg_subg.head_nid = aten::VecToIdArray(Global2Local(neg_vids, neg_map));
    neg_subg.tail_nid = aten::VecToIdArray(local_pos_vids);
  } else {
    neg_subg.head_nid = aten::VecToIdArray(local_pos_vids);
    neg_subg.tail_nid = aten::VecToIdArray(Global2Local(neg_vids, neg_map));
  }
  if (check_false_neg) {
    if (relations->shape[0] == 0) {
      neg_subg.exist = CheckExistence(gptr, neg_src, neg_dst, induced_neg_vid);
    } else {
      neg_subg.exist = CheckExistence(gptr, relations, neg_src, neg_dst,
                                      induced_neg_vid, induced_neg_eid);
    }
  }
  return neg_subg;
}

inline SubgraphRef ConvertRef(const Subgraph &subg) {
    return SubgraphRef(std::shared_ptr<Subgraph>(new Subgraph(subg)));
}

inline SubgraphRef ConvertRef(const NegSubgraph &subg) {
    return SubgraphRef(std::shared_ptr<Subgraph>(new NegSubgraph(subg)));
}

}  // namespace

DGL_REGISTER_GLOBAL("sampling._CAPI_UniformEdgeSampling")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    // arguments
    GraphRef g = args[0];
    IdArray seed_edges = args[1];
    const int64_t batch_start_id = args[2];
    const int64_t batch_size = args[3];
    const int64_t max_num_workers = args[4];
    const std::string neg_mode = args[5];
    const int neg_sample_size = args[6];
    const bool exclude_positive = args[7];
    const bool check_false_neg = args[8];
    IdArray relations = args[9];
    // process args
    auto gptr = std::dynamic_pointer_cast<ImmutableGraph>(g.sptr());
    CHECK(gptr) << "sampling isn't implemented in mutable graph";
    CHECK(aten::IsValidIdArray(seed_edges));
    BuildCoo(*gptr);

    const int64_t num_seeds = seed_edges->shape[0];
    const int64_t num_workers = std::min(max_num_workers,
        (num_seeds + batch_size - 1) / batch_size - batch_start_id);
    // generate subgraphs.
    std::vector<SubgraphRef> positive_subgs(num_workers);
    std::vector<SubgraphRef> negative_subgs(num_workers);
#pragma omp parallel for
    for (int i = 0; i < num_workers; i++) {
      const int64_t start = (batch_start_id + i) * batch_size;
      const int64_t end = std::min(start + batch_size, num_seeds);
      const int64_t num_edges = end - start;
      IdArray worker_seeds = seed_edges.CreateView({num_edges}, DLDataType{kDLInt, 64, 1},
                                                   sizeof(dgl_id_t) * start);
      EdgeArray arr = gptr->FindEdges(worker_seeds);
      const dgl_id_t *src_ids = static_cast<const dgl_id_t *>(arr.src->data);
      const dgl_id_t *dst_ids = static_cast<const dgl_id_t *>(arr.dst->data);
      std::vector<dgl_id_t> src_vec(src_ids, src_ids + num_edges);
      std::vector<dgl_id_t> dst_vec(dst_ids, dst_ids + num_edges);
      // TODO(zhengda) what if there are duplicates in the src and dst vectors.

      Subgraph subg = gptr->EdgeSubgraph(worker_seeds, false);
      positive_subgs[i] = ConvertRef(subg);
      // For PBG negative sampling, we accept "PBG-head" for corrupting head
      // nodes and "PBG-tail" for corrupting tail nodes.
      if (neg_mode.substr(0, 3) == "PBG") {
        NegSubgraph neg_subg = PBGNegEdgeSubgraph(gptr, relations, subg,
                                                  neg_mode.substr(4), neg_sample_size,
                                                  gptr->IsMultigraph(), exclude_positive,
                                                  check_false_neg);
        negative_subgs[i] = ConvertRef(neg_subg);
      } else if (neg_mode.size() > 0) {
        NegSubgraph neg_subg = NegEdgeSubgraph(gptr, relations, subg, neg_mode, neg_sample_size,
                                               exclude_positive, check_false_neg);
        negative_subgs[i] = ConvertRef(neg_subg);
      }
    }
    if (neg_mode.size() > 0) {
      positive_subgs.insert(positive_subgs.end(), negative_subgs.begin(), negative_subgs.end());
    }
    *rv = List<SubgraphRef>(positive_subgs);
  });


DGL_REGISTER_GLOBAL("sampling._CAPI_GetNegEdgeExistence")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  SubgraphRef g = args[0];
  auto gptr = std::dynamic_pointer_cast<NegSubgraph>(g.sptr());
  *rv = gptr->exist;
});

DGL_REGISTER_GLOBAL("sampling._CAPI_GetEdgeSubgraphHead")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  SubgraphRef g = args[0];
  auto gptr = std::dynamic_pointer_cast<NegSubgraph>(g.sptr());
  *rv = gptr->head_nid;
});

DGL_REGISTER_GLOBAL("sampling._CAPI_GetEdgeSubgraphTail")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  SubgraphRef g = args[0];
  auto gptr = std::dynamic_pointer_cast<NegSubgraph>(g.sptr());
  *rv = gptr->tail_nid;
});

}  // namespace dgl
