/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampling/randomwalk_cpu.cc
 * \brief DGL sampler - CPU implementation of metapath-based random walk with OpenMP
 */

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/random.h>
#include <utility>
#include <vector>
#include "randomwalks.h"
#include "randomwalks_cpu.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

template<DLDeviceType XPU, typename IdxType>
std::pair<dgl_id_t, bool> _MetapathRandomWalkStep(
    void *data,
    dgl_id_t curr,
    int64_t len,
    const std::vector<std::vector<IdArray> > &edges_by_type,
    const IdxType *metapath_data,
    const std::vector<FloatArray> &prob) {
  dgl_type_t etype = metapath_data[len];

  // Note that since the selection of successors is very lightweight (especially in the
  // uniform case), we want to reduce the overheads (even from object copies or object
  // construction) as much as possible.
  // Using Successors() slows down by 2x.
  // Using OutEdges() slows down by 10x.
  const auto &csr_arrays = edges_by_type[etype];
  const IdxType *offsets = static_cast<IdxType *>(csr_arrays[0]->data);
  const IdxType *all_succ = static_cast<IdxType *>(csr_arrays[1]->data);
  const IdxType *succ = all_succ + offsets[curr];

  int64_t size = offsets[curr + 1] - offsets[curr];
  if (size == 0)
    return std::make_pair(-1, true);

  FloatArray prob_etype = prob[etype];
  if (prob_etype->shape[0] == 0) {
    // empty probability array; assume uniform
    IdxType idx = RandomEngine::ThreadLocal()->RandInt(size);
    curr = succ[idx];
  } else {
    // non-uniform random walk
    const IdxType *all_eids = static_cast<IdxType *>(csr_arrays[2]->data);
    const IdxType *eids = all_eids + offsets[curr];

    ATEN_FLOAT_TYPE_SWITCH(prob_etype->dtype, DType, "probability", {
      const DType *prob_etype_data = static_cast<DType *>(prob_etype->data);
      std::vector<DType> prob_selected;
      for (int64_t j = 0; j < size; ++j)
        prob_selected.push_back(prob_etype_data[eids[j]]);

      curr = succ[RandomEngine::ThreadLocal()->Choice<int64_t>(prob_selected)];
    });
  }

  return std::make_pair(curr, false);
}

template<DLDeviceType XPU, typename IdxType>
IdArray RandomWalk(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob) {
  int64_t max_num_steps = metapath->shape[0];
  const IdxType *metapath_data = static_cast<IdxType *>(metapath->data);

  // Prefetch all edges.
  // This forces the heterograph to materialize all OutCSR's before the OpenMP loop;
  // otherwise data races will happen.
  // TODO(BarclayII): should we later on materialize COO/CSR/CSC anyway unless told otherwise?
  std::vector<std::vector<IdArray> > edges_by_type;
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype)
    edges_by_type.push_back(hg->GetAdj(etype, true, "csr"));

  StepFunc step =
    [&edges_by_type, metapath_data, &prob]
    (void *data, dgl_id_t curr, int64_t len) {
      return _MetapathRandomWalkStep<XPU, IdxType>(
          data, curr, len, edges_by_type, metapath_data, prob);
    };

  return GenericRandomWalk<XPU, IdxType>(hg, seeds, max_num_steps, step);
}

template
IdArray RandomWalk<kDLCPU, int32_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob);
template
IdArray RandomWalk<kDLCPU, int64_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob);

};  // namespace impl

};  // namespace sampling

};  // namespace dgl
