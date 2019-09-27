/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampler.cc
 * \brief DGL sampler implementation
 */

#include <dmlc/omp.h>
#include <dgl/immutable_graph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/random.h>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <functional>
#include <vector>
#include "randomwalk.h"
#include "../../c_api_common.h"

using namespace dgl::runtime;

namespace dgl {

namespace sampling {

using Walker = std::function<dgl_id_t(const GraphInterface *, dgl_id_t)>;

namespace {

/*!
 * \brief Randomly select a single direct successor given the current vertex
 * \return Whether such a successor could be found
 */
dgl_id_t WalkOneHop(
    const GraphInterface *gptr,
    dgl_id_t cur) {
  const auto succ = gptr->SuccVec(cur);
  const size_t size = succ.size();
  if (size == 0)
    return DGL_INVALID_ID;
  return succ[RandomEngine::ThreadLocal()->RandInt(size)];
}

/*!
 * \brief Randomly select a single direct successor after \c hops hops given the current vertex
 * \return Whether such a successor could be found
 */
template<int hops>
dgl_id_t WalkMultipleHops(
    const GraphInterface *gptr,
    dgl_id_t cur) {
  dgl_id_t next;
  for (int i = 0; i < hops; ++i) {
    if ((next = WalkOneHop(gptr, cur)) == DGL_INVALID_ID)
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
  dgl_id_t next;

  for (int64_t i = 0; i < num_nodes; ++i) {
    const dgl_id_t seed_id = seed_ids[i];

    for (int j = 0; j < num_traces; ++j) {
      dgl_id_t cur = seed_id;
      const int kmax = num_hops + 1;

      for (int k = 0; k < kmax; ++k) {
        const int64_t offset = (i * num_traces + j) * kmax + k;
        trace_data[offset] = cur;
        if ((next = walker(gptr, cur)) == DGL_INVALID_ID)
          LOG(FATAL) << "no successors from vertex " << cur;
        cur = next;
      }
    }
  }

  return traces;
}

RandomWalkTracesPtr GenericRandomWalkWithRestart(
    const GraphInterface *gptr,
    IdArray seeds,
    double restart_prob,
    uint64_t visit_threshold_per_seed,
    uint64_t max_visit_counts,
    uint64_t max_frequent_visited_nodes,
    Walker walker) {
  std::vector<dgl_id_t> vertices;
  std::vector<size_t> trace_lengths, trace_counts, visit_counts;
  const dgl_id_t *seed_ids = static_cast<dgl_id_t *>(seeds->data);
  const uint64_t num_nodes = seeds->shape[0];

  visit_counts.resize(gptr->NumVertices());

  for (uint64_t i = 0; i < num_nodes; ++i) {
    int stop = 0;
    size_t total_trace_length = 0;
    size_t num_traces = 0;
    uint64_t num_frequent_visited_nodes = 0;
    std::fill(visit_counts.begin(), visit_counts.end(), 0);

    while (1) {
      dgl_id_t cur = seed_ids[i], next;
      size_t trace_length = 0;

      for (; ; ++trace_length) {
        if ((trace_length > 0) &&
            (++visit_counts[cur] == max_visit_counts) &&
            (++num_frequent_visited_nodes == max_frequent_visited_nodes))
          stop = 1;

        if ((trace_length > 0) &&
            (RandomEngine::ThreadLocal()->Uniform<double>() < restart_prob))
          break;

        if ((next = walker(gptr, cur)) == DGL_INVALID_ID)
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
  }

  RandomWalkTraces *traces = new RandomWalkTraces;
  traces->trace_counts = IdArray::Empty(
      {static_cast<int64_t>(trace_counts.size())},
      DLDataType{kDLInt, 64, 1},
      DLContext{kDLCPU, 0});
  traces->trace_lengths = IdArray::Empty(
      {static_cast<int64_t>(trace_lengths.size())},
      DLDataType{kDLInt, 64, 1},
      DLContext{kDLCPU, 0});
  traces->vertices = IdArray::Empty(
      {static_cast<int64_t>(vertices.size())},
      DLDataType{kDLInt, 64, 1},
      DLContext{kDLCPU, 0});

  dgl_id_t *trace_counts_data = static_cast<dgl_id_t *>(traces->trace_counts->data);
  dgl_id_t *trace_lengths_data = static_cast<dgl_id_t *>(traces->trace_lengths->data);
  dgl_id_t *vertices_data = static_cast<dgl_id_t *>(traces->vertices->data);

  std::copy(trace_counts.begin(), trace_counts.end(), trace_counts_data);
  std::copy(trace_lengths.begin(), trace_lengths.end(), trace_lengths_data);
  std::copy(vertices.begin(), vertices.end(), vertices_data);

  return RandomWalkTracesPtr(traces);
}

};  // namespace

IdArray RandomWalk(
    const GraphInterface *gptr,
    IdArray seeds,
    int num_traces,
    int num_hops) {
  return GenericRandomWalk(gptr, seeds, num_traces, num_hops, WalkMultipleHops<1>);
}

RandomWalkTracesPtr RandomWalkWithRestart(
    const GraphInterface *gptr,
    IdArray seeds,
    double restart_prob,
    uint64_t visit_threshold_per_seed,
    uint64_t max_visit_counts,
    uint64_t max_frequent_visited_nodes) {
  return GenericRandomWalkWithRestart(
      gptr, seeds, restart_prob, visit_threshold_per_seed, max_visit_counts,
      max_frequent_visited_nodes, WalkMultipleHops<1>);
}

RandomWalkTracesPtr BipartiteSingleSidedRandomWalkWithRestart(
    const GraphInterface *gptr,
    IdArray seeds,
    double restart_prob,
    uint64_t visit_threshold_per_seed,
    uint64_t max_visit_counts,
    uint64_t max_frequent_visited_nodes) {
  return GenericRandomWalkWithRestart(
      gptr, seeds, restart_prob, visit_threshold_per_seed, max_visit_counts,
      max_frequent_visited_nodes, WalkMultipleHops<2>);
}

DGL_REGISTER_GLOBAL("sampler.randomwalk._CAPI_DGLRandomWalk")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef g = args[0];
    const IdArray seeds = args[1];
    const int num_traces = args[2];
    const int num_hops = args[3];

    *rv = RandomWalk(g.sptr().get(), seeds, num_traces, num_hops);
  });

DGL_REGISTER_GLOBAL("sampler.randomwalk._CAPI_DGLRandomWalkWithRestart")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef g = args[0];
    const IdArray seeds = args[1];
    const double restart_prob = args[2];
    const uint64_t visit_threshold_per_seed = args[3];
    const uint64_t max_visit_counts = args[4];
    const uint64_t max_frequent_visited_nodes = args[5];

    *rv = RandomWalkTracesRef(
        RandomWalkWithRestart(g.sptr().get(), seeds, restart_prob, visit_threshold_per_seed,
          max_visit_counts, max_frequent_visited_nodes));
  });

DGL_REGISTER_GLOBAL("sampler.randomwalk._CAPI_DGLBipartiteSingleSidedRandomWalkWithRestart")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef g = args[0];
    const IdArray seeds = args[1];
    const double restart_prob = args[2];
    const uint64_t visit_threshold_per_seed = args[3];
    const uint64_t max_visit_counts = args[4];
    const uint64_t max_frequent_visited_nodes = args[5];

    *rv = RandomWalkTracesRef(
        BipartiteSingleSidedRandomWalkWithRestart(
          g.sptr().get(), seeds, restart_prob, visit_threshold_per_seed,
          max_visit_counts, max_frequent_visited_nodes));
  });

};  // namespace sampling

};  // namespace dgl
