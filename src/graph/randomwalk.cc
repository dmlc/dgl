/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampler.cc
 * \brief DGL sampler implementation
 */

#include <dgl/sampler.h>
#include <dgl/nodeflow.h>
#include <dmlc/omp.h>
#include <dgl/immutable_graph.h>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <functional>
#include "../c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::PackedFunc;
using dgl::runtime::NDArray;

namespace dgl {

using Walker = std::function<dgl_id_t(
    const GraphInterface *, unsigned int *, dgl_id_t)>;

namespace {

/*!
 * \brief Randomly select a single direct successor given the current vertex
 * \return Whether such a successor could be found
 */
dgl_id_t WalkOneHop(
    const GraphInterface *gptr,
    unsigned int *random_seed,
    dgl_id_t cur) {
  const auto succ = gptr->SuccVec(cur);
  const size_t size = succ.size();
  if (size == 0)
    return DGL_INVALID_ID;
  return succ[rand_r(random_seed) % size];
}

/*!
 * \brief Randomly select a single direct successor after \c hops hops given the current vertex
 * \return Whether such a successor could be found
 */
template<int hops>
dgl_id_t WalkMultipleHops(
    const GraphInterface *gptr,
    unsigned int *random_seed,
    dgl_id_t cur) {
  dgl_id_t next;
  for (int i = 0; i < hops; ++i) {
    if ((next = WalkOneHop(gptr, random_seed, cur)) == DGL_INVALID_ID)
      return DGL_INVALID_ID;
    cur = next;
  }
  return cur;
}

IdArray GenericRandomWalk(
    const GraphInterface *gptr,
    IdArray seeds,
    int num_traces,
    int num_hops,
    Walker walker) {
  const int64_t num_nodes = seeds->shape[0];
  const dgl_id_t *seed_ids = static_cast<dgl_id_t *>(seeds->data);
  IdArray traces = IdArray::Empty(
      {num_nodes, num_traces, num_hops + 1},
      DLDataType{kDLInt, 64, 1},
      DLContext{kDLCPU, 0});
  dgl_id_t *trace_data = static_cast<dgl_id_t *>(traces->data);

  // FIXME: does OpenMP work with exceptions?  Especially without throwing SIGABRT?
  unsigned int random_seed = randseed();
  dgl_id_t next;

  for (int64_t i = 0; i < num_nodes; ++i) {
    const dgl_id_t seed_id = seed_ids[i];

    for (int j = 0; j < num_traces; ++j) {
      dgl_id_t cur = seed_id;
      const int kmax = num_hops + 1;

      for (int k = 0; k < kmax; ++k) {
        const int64_t offset = (i * num_traces + j) * kmax + k;
        trace_data[offset] = cur;
        if ((next = walker(gptr, &random_seed, cur)) == DGL_INVALID_ID)
          LOG(FATAL) << "no successors from vertex " << cur;
        cur = next;
      }
    }
  }

  return traces;
}

RandomWalkTraces GenericRandomWalkWithRestart(
    const GraphInterface *gptr,
    Iter seed_iter,
    uint64_t num_nodes,
    double restart_prob,
    uint64_t visit_threshold_per_seed,
    uint64_t max_visit_counts,
    uint64_t max_frequent_visited_nodes,
    Walker walker) {
  std::vector<dgl_id_t> vertices;
  std::vector<size_t> trace_lengths, trace_counts, visit_counts;
  int64_t restart_bound = static_cast<int64_t>(restart_prob * RAND_MAX);

  if (!(restart_prob > 0)) {
    LOG(FATAL) << "restart_prob is not positive";
    return RandomWalkTraces();
  }

  visit_counts.resize(gptr->NumVertices());

  unsigned int random_seed = randseed();

  for (uint64_t i = 0; i < num_nodes; ++i) {
    int stop = 0;
    dgl_id_t cur = *seed_iter, next;
    size_t total_trace_length = 0;
    size_t num_traces = 0;
    uint64_t num_frequent_visited_nodes = 0;
    std::fill(visit_counts.begin(), visit_counts.end(), 0);

    while (1) {
      size_t trace_length = 0;

      for (; ; ++trace_length) {
        if ((trace_length > 0) &&
            (++visit_counts[cur] == max_visit_counts) &&
            (++num_frequent_visited_nodes == max_frequent_visited_nodes))
          stop = 1;

        if ((trace_length > 0) && (rand_r(&random_seed) < restart_bound))
          break;

        if ((next = walker(gptr, &random_seed, cur)) == DGL_INVALID_ID)
          LOG(FATAL) << "no successors from vertex " << cur;
        cur = next;
        vertices.push_back(cur);
      }

      total_trace_length += trace_length;
      ++num_traces;
      trace_lengths.push_back(trace_length);
      if ((total_trace_length >= visit_threshold_per_seed) || stop)
        break;
    }

    trace_counts.push_back(num_traces);
    ++seed_iter;
  }

  RandomWalkTraces traces;
  traces.trace_counts = IdArray::Empty(
      {static_cast<int64_t>(trace_counts.size())},
      DLDataType{kDLInt, 64, 1},
      DLContext{kDLCPU, 0});
  traces.trace_lengths = IdArray::Empty(
      {static_cast<int64_t>(trace_lengths.size())},
      DLDataType{kDLInt, 64, 1},
      DLContext{kDLCPU, 0});
  traces.vertices = IdArray::Empty(
      {static_cast<int64_t>(vertices.size())},
      DLDataType{kDLInt, 64, 1},
      DLContext{kDLCPU, 0});

  size_t *trace_counts_data = static_cast<size_t *>(traces.trace_counts->data);
  size_t *trace_lengths_data = static_cast<size_t *>(traces.trace_lengths->data);
  dgl_id_t *vertices_data = static_cast<dgl_id_t *>(traces.vertices->data);

  std::copy(trace_counts.begin(), trace_counts.end(), trace_counts_data);
  std::copy(trace_lengths.begin(), trace_lengths.end(), trace_lengths_data);
  std::copy(vertices.begin(), vertices.end(), vertices_data);

  return traces;
}

};  // namespace

PackedFunc ConvertRandomWalkTracesToPackedFunc(const RandomWalkTraces &t) {
  return ConvertNDArrayVectorToPackedFunc({
      t.trace_counts, t.trace_lengths, t.vertices});
}

IdArray RandomWalk(
    const GraphInterface *gptr,
    IdArray seeds,
    int num_traces,
    int num_hops) {
  return GenericRandomWalk(gptr, seeds, num_traces, num_hops, WalkMultipleHops<1>);
}

RandomWalkTraces RandomWalkWithRestart(
    const GraphInterface *gptr,
    IdArray seeds,
    double restart_prob,
    uint64_t visit_threshold_per_seed,
    uint64_t max_visit_counts,
    uint64_t max_frequent_visited_nodes) {
  const dgl_id_t *seed_data = static_cast<dgl_id_t *>(seeds->data);
  const size_t num_nodes = seeds->shape[0];
  return GenericRandomWalkWithRestart(
      gptr, seeds, restart_prob, visit_threshold_per_seed, max_visit_counts,
      max_frequent_visited_nodes, WalkMultipleHops<1>);
}

RandomWalkTraces BipartiteSingleSidedRandomWalkWithRestart(
    const GraphInterface *gptr,
    IdArray seeds,
    double restart_prob,
    uint64_t visit_threshold_per_seed,
    uint64_t max_visit_counts,
    uint64_t max_frequent_visited_nodes) {
  const dgl_id_t *seed_data = static_cast<dgl_id_t *>(seeds->data);
  const size_t num_nodes = seeds->shape[0];
  return GenericRandomWalkWithRestart(
      gptr, seeds, restart_prob, visit_threshold_per_seed, max_visit_counts,
      max_frequent_visited_nodes, WalkMultipleHops<2>);
}

// TODO: Abstract common logic with SampleSubgraph
template<typename Iter>
NodeFlow CreateNodeFlowWithPPRFromRandomWalk(
    const GraphInterface *gptr,
    Iter seed_data,
    size_t num_seeds,
    double restart_prob,
    uint64_t max_nodes_per_seed,
    uint64_t max_visit_counts,
    uint64_t max_frequent_visited_nodes,
    bool (*walker)(const GraphInterface *, unsigned int *, dgl_id_t, dgl_id_t *),
    int num_hops,
    uint64_t top_t,
    bool add_self_loop) {
  // TODO: better naming; these follow SampleSubgraph
  std::vector<dgl_id_t> neighbor_list, edge_list;
  std::vector<double> edge_data;
  std::vector<size_t> layer_offsets(num_hops + 1);
  std::vector<neighbor_info> neigh_pos;
  std::unordered_set<dgl_id_t> sub_ver_map;
  std::vector<dgl_id_t> sub_vers;
  std::map<dgl_id_t, size_t> visit_counter;   // ordered since we need to sort it

  for (size_t i = 0; i < num_seeds; ++i) {
    auto ret = sub_ver_map.insert(seed_data[i]);
    if (ret.second)
      sub_vers.push_back(seed_data[i]);
  }

  layer_offsets[0] = 0;
  layer_offsets[1] = sub_vers.size();

  for (int layer_id = 1; layer_id < num_hops; ++layer_id) {
    sub_ver_map.clear();

    const uint64_t layer_start_off = layer_offsets[layer_id - 1];
    const uint64_t num_nodes_in_layer = layer_offsets[layer_id] - layer_offsets[layer_id - 1];
    auto sub_vers_iter = sub_vers.cbegin() + layer_start_off;

    RandomWalkTraces traces = GenericRandomWalkWithRestart(
        gptr, sub_vers_iter, num_nodes_in_layer, restart_prob, max_nodes_per_seed,
        max_visit_counts, max_frequent_visited_nodes, walker);
    const size_t *trace_counts_ptr = static_cast<size_t *>(traces.trace_counts->data);
    const size_t *trace_lengths_ptr = static_cast<dgl_id_t *>(traces.trace_lengths->data);
    const dgl_id_t *trace_vertices_ptr = static_cast<dgl_id_t *>(traces.vertices->data);

    for (size_t seed_idx = 0; seed_idx < num_nodes_in_layer;
         ++seed_idx, ++trace_counts_ptr) {
      const dgl_id_t dst = sub_vers[seed_idx + layer_start_off];
      visit_counter.clear();

      for (size_t trace_idx = 0; trace_idx < *trace_counts_ptr;
           ++trace_idx, ++trace_lengths_ptr) {
        for (size_t vertex_idx = 0; vertex_idx < *trace_lengths_ptr;
             ++vertex_idx, ++trace_vertices_ptr)
          ++visit_counter[*trace_vertices_ptr];
      }

      // pick top t vertices and connect them to dst
      std::vector<std::pair<dgl_id_t, size_t>> visit_counter_vec(
          visit_counter.begin(), visit_counter.end());
      std::sort(
          visit_counter_vec.begin(),
          visit_counter_vec.end(),
          [] (std::pair<dgl_id_t, size_t> &a, std::pair<dgl_id_t, size_t> &b) {
            return a.second > b.second;
          });
      size_t total_visits = 0;
      uint64_t t = 0;
      auto it = visit_counter_vec.cbegin();
      for (; t < top_t && it != visit_counter_vec.cend(); ++t, ++it)
        total_visits += it->second;
      for (t = 0, it = visit_counter_vec.cbegin();
          t < top_t && it != visit_counter_vec.cend();
          ++t, ++it) {
        neighbor_list.push_back(it->first);
        edge_list.push_back(-1);    // not mapping edges to parent graph
        edge_data.push_back(1. * it->second / total_visits);

        auto ret = sub_ver_map.insert(it->first);
        if (ret.second)
          sub_vers.push_back(it->first);
      }
      if (add_self_loop) {
        neighbor_list.push_back(dst);
        edge_list.push_back(-1);
        edge_data.push_back(0.);    // not weighting over itself due to self loop
        ++t;
      }
      neigh_pos.emplace_back(dst, neighbor_list.size(), t);
    }

    layer_offsets[layer_id + 1] = layer_offsets[layer_id] + sub_ver_map.size();
    CHECK_EQ(layer_offsets[layer_id + 1], sub_vers.size());
  }

  std::vector<dgl_id_t> vertex_mapping, edge_mapping;
  NodeFlow nf;
  ConstructNodeFlow(
      neighbor_list, edge_list, layer_offsets, &sub_vers, &neigh_pos,
      "in", gptr->IsMultigraph(), &nf, &vertex_mapping, &edge_mapping);

  const int64_t num_edges = edge_list.size();
  nf.edge_data_available = true;
  nf.edge_data = NDArray::Empty(
      {num_edges}, DLDataType{kDLFloat, 64, 1}, DLContext{kDLCPU, 0});
  double *edge_data_out = static_cast<double *>(nf.edge_data->data);
  for (int64_t i = 0; i < num_edges; ++i)
    edge_data_out[i] = edge_data[edge_mapping[i]];
  return nf;
}

// TODO: extract common logic with UniformSampling & alike
std::vector<NodeFlow *> PPRNeighborSampling(
    const GraphInterface *gptr,
    IdArray seed_nodes,
    double restart_prob,
    uint64_t max_nodes_per_seed,
    uint64_t max_visit_counts,
    uint64_t max_frequent_visited_nodes,
    bool (*walker)(const GraphInterface *, unsigned int *, dgl_id_t, dgl_id_t *),
    int num_hops,
    uint64_t top_t,
    int64_t max_num_workers,
    int64_t batch_size,
    /*
     * FIXME: follows _CAPI_UniformSampling; why do we need this?
     * Is passing a single huge array + start offset faster and/or more scalable than
     * passing in multiple small arrays?
     */
    int64_t batch_start_id,
    bool add_self_loop) {
  CHECK(IsValidIdArray(seed_nodes));
  const dgl_id_t *seed_nodes_data = static_cast<dgl_id_t *>(seed_nodes->data);
  const int64_t num_seeds = seed_nodes->shape[0];
  const int64_t num_workers = std::min(max_num_workers,
      (num_seeds + batch_size - 1) / batch_size - batch_start_id);

  std::vector<NodeFlow *> nflows(num_workers);
#pragma omp parallel for
  for (int i = 0; i < num_workers; ++i) {
    const int64_t start = (batch_start_id + i) * batch_size;
    const int64_t end = std::min(start + batch_size, num_seeds);
    nflows[i] = new NodeFlow();
    *nflows[i] = SamplerOp::CreateNodeFlowWithPPRFromRandomWalk(
        gptr, seed_nodes_data + start, end - start, restart_prob, max_nodes_per_seed,
        max_visit_counts, max_frequent_visited_nodes, walker, num_hops, top_t,
        add_self_loop);
  }

  return nflows;
}

void PPRNeighborSamplingEntry(
    DGLArgs args,
    DGLRetValue* rv,
    bool (*walker)(const GraphInterface *, unsigned int *, dgl_id_t, dgl_id_t *)) {
  GraphHandle ghandle = args[0];
  const IdArray seed_nodes = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
  const int64_t batch_start_id = args[2];
  const int64_t batch_size = args[3];
  const int64_t max_num_workers = args[4];

  const double restart_prob = args[5];
  const uint64_t max_nodes_per_seed = args[6];
  const uint64_t max_visit_counts = args[7];
  const uint64_t max_frequent_visited_nodes = args[8];
  const int num_hops = args[9];
  const uint64_t top_t = args[10];
  const bool add_self_loop = args[11];

  const GraphInterface *gptr = static_cast<const GraphInterface *>(ghandle);

  std::vector<NodeFlow *> nflows = PPRNeighborSampling(
      gptr, seed_nodes, restart_prob, max_nodes_per_seed, max_visit_counts,
      max_frequent_visited_nodes, walker, num_hops, top_t,
      max_num_workers, batch_size, batch_start_id, add_self_loop);

  *rv = WrapVectorReturn(nflows);
}

DGL_REGISTER_GLOBAL("randomwalk._CAPI_RandomWalk")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const IdArray seeds = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    const int num_traces = args[2];
    const int num_hops = args[3];
    const GraphInterface *ptr = static_cast<const GraphInterface *>(ghandle);

    *rv = RandomWalk(ptr, seeds, num_traces, num_hops);
  });

DGL_REGISTER_GLOBAL("randomwalk._CAPI_RandomWalkWithRestart")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const IdArray seeds = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    const double restart_prob = args[2];
    const uint64_t visit_threshold_per_seed = args[3];
    const uint64_t max_visit_counts = args[4];
    const uint64_t max_frequent_visited_nodes = args[5];
    const GraphInterface *gptr = static_cast<const GraphInterface *>(ghandle);

    *rv = ConvertRandomWalkTracesToPackedFunc(
        RandomWalkWithRestart(gptr, seeds, restart_prob, visit_threshold_per_seed,
          max_visit_counts, max_frequent_visited_nodes));
  });

DGL_REGISTER_GLOBAL("randomwalk._CAPI_BipartiteSingleSidedRandomWalkWithRestart")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const IdArray seeds = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    const double restart_prob = args[2];
    const uint64_t visit_threshold_per_seed = args[3];
    const uint64_t max_visit_counts = args[4];
    const uint64_t max_frequent_visited_nodes = args[5];
    const GraphInterface *gptr = static_cast<const GraphInterface *>(ghandle);

    *rv = ConvertRandomWalkTracesToPackedFunc(
        BipartiteSingleSidedRandomWalkWithRestart(
          gptr, seeds, restart_prob, visit_threshold_per_seed,
          max_visit_counts, max_frequent_visited_nodes));
  });

DGL_REGISTER_GLOBAL("randomwalk._CAPI_PPRNeighborSampling")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    PPRNeighborSamplingEntry(args, rv, multihop_walker<1>);
  });

DGL_REGISTER_GLOBAL("randomwalk._CAPI_PPRBipartiteSingleSidedNeighborSampling")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    PPRNeighborSamplingEntry(args, rv, multihop_walker<2>);
  });

};  // namespace dgl
