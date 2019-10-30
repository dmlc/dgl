/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/sampler/metapath.cc
 * \brief Metapath sampling
 */

#include <dgl/array.h>
#include <dgl/random.h>
#include <dgl/packed_func_ext.h>
#include "../../c_api_common.h"
#include "randomwalk.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {

namespace sampling {

namespace {

/*!
 * \brief Random walk based on the given metapath.
 *
 * Raises an error if the trace could not completely traverse the given metapath.
 *
 * \param hg The heterograph
 * \param etypes The metapath as an array of edge type IDs
 * \param seeds The array of starting vertices for random walks
 * \param num_traces Number of traces to generate for each starting vertex
 * \note The metapath should have the same starting and ending node type
 * \return A 3D tensor traces[seed_id][trace_id][step]
 */
IdArray MetapathRandomWalkAligned(
    const HeteroGraphPtr hg,
    const IdArray etypes,
    const IdArray seeds,
    int num_traces) {
  const auto metagraph = hg->meta_graph();
  int64_t num_etypes = etypes->shape[0];
  int64_t num_seeds = seeds->shape[0];
  const dgl_type_t *etype_data = static_cast<dgl_type_t *>(etypes->data);
  const dgl_id_t *seed_data = static_cast<dgl_id_t *>(seeds->data);

  IdArray traces = IdArray::Empty(
      {num_seeds, num_traces, num_etypes},
      DLDataType{kDLInt, hg->NumBits(), 1},
      hg->Context());
  dgl_id_t *traces_data = static_cast<dgl_id_t *>(traces->data);

#pragma omp parallel for
  for (int64_t seed_id = 0; seed_id < num_seeds; ++seed_id) {
    int curr_num_traces = 0;
    int64_t pos = seed_id * num_traces * num_etypes;
    for (; curr_num_traces < num_traces; ++curr_num_traces) {
      dgl_id_t curr = seed_data[seed_id];

      for (int64_t i = 0; i < num_etypes; ++i) {
        const auto &succ = hg->SuccVec(etype_data[i], curr);
        if (succ.size() == 0) {
          LOG(FATAL) << "no successors of edge type " << etype_data[i] << " for vertex " << curr;
          break;
        }
        curr = succ[RandomEngine::ThreadLocal()->RandInt(succ.size())];
        traces_data[pos++] = curr;
      }
    }
  }

  return traces;
}

/*!
 * \brief Random walk based on the given metapath.
 *
 * Continues even if the trace could not completely traverse the given metapath.
 * This method would be slower than MetapathRandomWalkAligned()
 *
 * \param hg The heterograph
 * \param etypes The metapath as an array of edge type IDs
 * \param seeds The array of starting vertices for random walks
 * \param num_traces Number of traces to generate for each starting vertex
 * \note The metapath should have the same starting and ending node type.
 */
RandomWalkTracesPtr MetapathRandomWalk(
    const HeteroGraphPtr hg,
    const IdArray etypes,
    const IdArray seeds,
    int num_traces) {
  const auto metagraph = hg->meta_graph();
  uint64_t num_etypes = etypes->shape[0];
  uint64_t num_seeds = seeds->shape[0];
  const dgl_type_t *etype_data = static_cast<dgl_type_t *>(etypes->data);
  const dgl_id_t *seed_data = static_cast<dgl_id_t *>(seeds->data);

  // vertices[seed][concatenated_traces]
  std::vector< std::vector<dgl_id_t> > vertices_per_seed(num_seeds);
  // trace_lengths[seed][trace_counts[seed]]
  std::vector< std::vector<size_t> > trace_lengths_per_seed(num_seeds);
  // trace_counts[seed]
  std::vector<size_t> trace_counts(num_seeds);
  std::vector<dgl_id_t> vertices;
  std::vector<size_t> trace_lengths;

  // TODO(quan): use omp to parallelize this loop
#pragma omp parallel for
  for (int64_t seed_id = 0; seed_id < num_seeds; ++seed_id) {
    std::vector<dgl_id_t> curr_vertices;
    std::vector<dgl_id_t> curr_trace_lengths;
    int curr_num_traces = 0;

    for (; curr_num_traces < num_traces; ++curr_num_traces) {
      dgl_id_t curr = seed_data[seed_id];

      size_t trace_length = 0;

      for (size_t i = 0; i < num_etypes; ++i) {
        const auto &succ = hg->SuccVec(etype_data[i], curr);
        if (succ.size() == 0)
          break;
        curr = succ[RandomEngine::ThreadLocal()->RandInt(succ.size())];
        curr_vertices.push_back(curr);
        ++trace_length;
      }

      curr_trace_lengths.push_back(trace_length);
    }

    vertices_per_seed[seed_id] = curr_vertices;
    trace_lengths_per_seed[seed_id] = curr_trace_lengths;
    trace_counts[seed_id] = curr_num_traces;
  }

  for (uint64_t seed_id = 0; seed_id < num_seeds; ++seed_id) {
    vertices.insert(
        vertices.end(),
        vertices_per_seed[seed_id].begin(),
        vertices_per_seed[seed_id].end());
    trace_lengths.insert(
        trace_lengths.end(),
        trace_lengths_per_seed[seed_id].begin(),
        trace_lengths_per_seed[seed_id].end());
  }

  RandomWalkTraces *tl = new RandomWalkTraces;
  tl->vertices = VecToIdArray(vertices);
  tl->trace_lengths = VecToIdArray(trace_lengths);
  tl->trace_counts = VecToIdArray(trace_counts);

  return RandomWalkTracesPtr(tl);
}

};  // namespace

DGL_REGISTER_GLOBAL("sampler.randomwalk._CAPI_DGLMetapathRandomWalkAligned")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    const HeteroGraphRef hg = args[0];
    const IdArray etypes = args[1];
    const IdArray seeds = args[2];
    int num_traces = args[3];

    const IdArray traces = MetapathRandomWalkAligned(hg.sptr(), etypes, seeds, num_traces);
    *rv = traces;
  });

DGL_REGISTER_GLOBAL("sampler.randomwalk._CAPI_DGLMetapathRandomWalk")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    const HeteroGraphRef hg = args[0];
    const IdArray etypes = args[1];
    const IdArray seeds = args[2];
    int num_traces = args[3];

    const auto tl = MetapathRandomWalk(hg.sptr(), etypes, seeds, num_traces);
    *rv = RandomWalkTracesRef(tl);
  });

};  // namespace sampling

};  // namespace dgl
