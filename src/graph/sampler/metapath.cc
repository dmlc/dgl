/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/sampler/metapath.cc
 * \brief Metapath sampling
 */

#include <dgl/array.h>
#include <dgl/random.h>
#include <dgl/packed_func_ext.h>
#include <dgl/array.h>
#include "../../c_api_common.h"
#include "../unit_graph.h"
#include "randomwalk.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {

namespace sampling {

namespace {

/*!
 * \brief Random walk based on the given metapath.
 * 
 * \tparam IdType Index dtype of graph
 * \param hg The heterograph
 * \param etypes The metapath as an array of edge type IDs
 * \param seeds The array of starting vertices for random walks
 * \param num_traces Number of traces to generate for each starting vertex
 * \note The metapath should have the same starting and ending node type.
 */
template <typename T>
RandomWalkTracesPtr MetapathRandomWalk(
    const HeteroGraphPtr hg,
    const IdArray etypes,
    const IdArray seeds,
    int num_traces);

template <>
RandomWalkTracesPtr MetapathRandomWalk<int64_t>(
    const HeteroGraphPtr hg,
    const IdArray etypes,
    const IdArray seeds,
    int num_traces) {
  const auto metagraph = hg->meta_graph();
  uint64_t num_etypes = etypes->shape[0];
  uint64_t num_seeds = seeds->shape[0];
  const dgl_type_t *etype_data = static_cast<dgl_type_t *>(etypes->data);
  const dgl_id_t *seed_data = static_cast<dgl_id_t *>(seeds->data);

  std::vector<dgl_id_t> vertices;
  std::vector<size_t> trace_lengths, trace_counts;

  // TODO(quan): use omp to parallelize this loop
  for (uint64_t seed_id = 0; seed_id < num_seeds; ++seed_id) {
    int curr_num_traces = 0;

    for (; curr_num_traces < num_traces; ++curr_num_traces) {
      dgl_id_t curr = seed_data[seed_id];

      size_t trace_length = 0;

      for (size_t i = 0; i < num_etypes; ++i) {
        const auto &succ = hg->SuccVec(etype_data[i], curr);
        if (succ.size() == 0)
          break;
        curr = succ[RandomEngine::ThreadLocal()->RandInt(succ.size())];
        vertices.push_back(curr);
        ++trace_length;
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

/*!
 * \brief This is a patch function for int32 HeteroGraph
 * TODO: Refactor this with CSR and COO operations
 */
template <>
RandomWalkTracesPtr MetapathRandomWalk<int32_t>(
    const HeteroGraphPtr hg,
    const IdArray etypes,
    const IdArray seeds,
    int num_traces) {
  const auto metagraph = hg->meta_graph();
  uint64_t num_etypes = etypes->shape[0];
  uint64_t num_seeds = seeds->shape[0];
  const dgl_type_t *etype_data = static_cast<dgl_type_t *>(etypes->data);
  const int32_t *seed_data = static_cast<int32_t *>(seeds->data);

  std::vector<int32_t> vertices;
  std::vector<size_t> trace_lengths, trace_counts;

  // TODO(quan): use omp to parallelize this loop
  for (uint64_t seed_id = 0; seed_id < num_seeds; ++seed_id) {
    int curr_num_traces = 0;

    for (; curr_num_traces < num_traces; ++curr_num_traces) {
      int32_t curr = seed_data[seed_id];

      size_t trace_length = 0;

      for (size_t i = 0; i < num_etypes; ++i) {
        auto ug = std::dynamic_pointer_cast<UnitGraph>(hg->GetRelationGraph(etype_data[i]));
        CHECK_NOTNULL(ug);
        const auto &succ = ug->SuccVec32(etype_data[i], curr);
        if (succ.size() == 0)
          break;
        curr = succ[RandomEngine::ThreadLocal()->RandInt(succ.size())];
        vertices.push_back(curr);
        ++trace_length;
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

DGL_REGISTER_GLOBAL("sampler.randomwalk._CAPI_DGLMetapathRandomWalk")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    const HeteroGraphRef hg = args[0];
    const IdArray etypes = args[1];
    const IdArray seeds = args[2];
    int num_traces = args[3];

    CHECK(aten::IsValidIdArray(etypes));
    CHECK_EQ(etypes->ctx.device_type, kDLCPU)
      << "MetapathRandomWalk only support CPU sampling";
    CHECK(aten::IsValidIdArray(seeds));
    CHECK_EQ(seeds->ctx.device_type, kDLCPU)
      << "MetapathRandomWalk only support CPU sampling";
    const int64_t bits = hg->NumBits();
    RandomWalkTracesPtr tl;
    ATEN_ID_BITS_SWITCH(bits, IdType, {
      tl = MetapathRandomWalk<IdType>(hg.sptr(), etypes, seeds, num_traces);
    });
    *rv = RandomWalkTracesRef(tl);
  });

};  // namespace sampling

};  // namespace dgl
