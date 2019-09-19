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
 * \brief Random walk based on indefinite cycle of given metapaths.
 *
 * \param hg The heterograph
 * \param etypes The metapath as an array of edge type IDs
 * \param seeds The array of starting vertices for random walks
 * \param num_traces Number of traces to generate for each starting vertex
 * \param max_cycles Maximum number of metapath traversals for each starting vertex
 * \note The metapath should have the same starting and ending node type.
 */
RandomWalkTracesPtr MetapathCycleRandomWalk(
    const HeteroGraphPtr hg,
    const IdArray etypes,
    const IdArray seeds,
    int num_traces,
    int max_cycles) {
  const auto metagraph = hg->meta_graph();
  uint64_t num_etypes = etypes->shape[0];
  uint64_t num_seeds = seeds->shape[0];
  const dgl_type_t *etype_data = static_cast<dgl_type_t *>(etypes->data);
  const dgl_id_t *seed_data = static_cast<dgl_id_t *>(seeds->data);

  std::vector<dgl_id_t> vertices;
  std::vector<size_t> trace_lengths, trace_counts;

  for (uint64_t seed_id = 0; seed_id < num_seeds; ++seed_id) {
    int curr_num_traces = 0;

    for (; curr_num_traces < num_traces; ++curr_num_traces) {
      dgl_id_t curr = seed_data[seed_id];

      int cycles = 0;
      size_t trace_length = 0;
      bool stop = false;

      for (; cycles < max_cycles; ++cycles) {
        for (size_t i = 0; i < num_etypes; ++i) {
          const auto &succ = hg->SuccVec(etype_data[i], curr);
          if (succ.size() == 0) {
            stop = true;
            break;
          }
          curr = succ[RandomEngine::ThreadLocal()->RandInt(succ.size())];
          vertices.push_back(curr);
          ++trace_length;
        }
        if (stop)
          break;
      }

      trace_lengths.push_back(trace_length);
    }

    trace_counts.push_back(curr_num_traces);
  }

  RandomWalkTraces *tl = new RandomWalkTraces;
  tl->vertices = VecToIdArray(vertices);
  tl->trace_lengths = VecToIdArray(trace_lengths);
  tl->trace_counts = VecToIdArray(trace_counts);

  return RandomWalkTracesPtr(tl);
}

};  // namespace

DGL_REGISTER_GLOBAL("sampler.randomwalk._CAPI_DGLMetapathCycleRandomWalk")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    const HeteroGraphRef hg = args[0];
    const IdArray etypes = args[1];
    const IdArray seeds = args[2];
    int num_traces = args[3];
    int max_cycles = args[4];

    const auto tl = MetapathCycleRandomWalk(hg.sptr(), etypes, seeds, num_traces, max_cycles);
    *rv = RandomWalkTracesRef(tl);
  });

};  // namespace sampling

};  // namespace dgl
