/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampler/generic_randomwalk_cpu.h
 * \brief DGL sampler - templated implementation definition of random walks on CPU
 */

#ifndef DGL_GRAPH_SAMPLING_RANDOMWALKS_METAPATH_RANDOMWALK_H_
#define DGL_GRAPH_SAMPLING_RANDOMWALKS_METAPATH_RANDOMWALK_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/random.h>
#include <utility>
#include <vector>
#include "randomwalks_impl.h"
#include "randomwalks_cpu.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

namespace {

// bool WhetherToTerminate(
//     IdxType *node_ids_generated_so_far,
//     dgl_id_t last_node_id_generated,
//     int64_t number_of_nodes_generated_so_far)
template<typename IdxType>
using TerminatePredicate = std::function<bool(IdxType *, dgl_id_t, int64_t)>;

/*!
 * \brief Select one successor of metapath-based random walk, given the path generated
 * so far.
 *
 * \param data The path generated so far, of type \c IdxType.
 * \param curr The last node ID generated.
 * \param len The number of nodes generated so far.  Note that the seed node is always
 *            included as \c data[0], and the successors start from \c data[1].
 *
 * \param edges_by_type Vector of results from \c GetAdj() by edge type.
 * \param metapath_data Edge types of given metapath.
 * \param prob Transition probability per edge type.
 * \param terminate Predicate for terminating the current random walk path.
 *
 * \return A pair of ID of next successor (-1 if not exist), as well as whether to terminate.
 */
template<DLDeviceType XPU, typename IdxType>
std::pair<dgl_id_t, bool> MetapathRandomWalkStep(
    IdxType *data,
    dgl_id_t curr,
    int64_t len,
    const std::vector<std::vector<IdArray> > &edges_by_type,
    const IdxType *metapath_data,
    const std::vector<FloatArray> &prob,
    TerminatePredicate<IdxType> terminate) {
  dgl_type_t etype = metapath_data[len];

  // Note that since the selection of successors is very lightweight (especially in the
  // uniform case), we want to reduce the overheads (even from object copies or object
  // construction) as much as possible.
  // Using Successors() slows down by 2x.
  // Using OutEdges() slows down by 10x.
  const std::vector<NDArray> &csr_arrays = edges_by_type[etype];
  const IdxType *offsets = static_cast<IdxType *>(csr_arrays[0]->data);
  const IdxType *all_succ = static_cast<IdxType *>(csr_arrays[1]->data);
  const IdxType *succ = all_succ + offsets[curr];

  const int64_t size = offsets[curr + 1] - offsets[curr];
  if (size == 0)
    return std::make_pair(-1, true);

  FloatArray prob_etype = prob[etype];
  IdxType idx;
  if (prob_etype->shape[0] == 0) {
    // empty probability array; assume uniform
    idx = RandomEngine::ThreadLocal()->RandInt(size);
  } else {
    // non-uniform random walk
    const IdxType *all_eids = static_cast<IdxType *>(csr_arrays[2]->data);
    const IdxType *eids = all_eids + offsets[curr];

    ATEN_FLOAT_TYPE_SWITCH(prob_etype->dtype, DType, "probability", {
      FloatArray prob_selected = FloatArray::Empty({size}, prob_etype->dtype, prob_etype->ctx);
      DType *prob_selected_data = static_cast<DType *>(prob_selected->data);
      const DType *prob_etype_data = static_cast<DType *>(prob_etype->data);
      for (int64_t j = 0; j < size; ++j)
        prob_selected_data[j] = prob_etype_data[eids[j]];
      idx = RandomEngine::ThreadLocal()->Choice<IdxType>(prob_selected);
    });
  }
  curr = succ[idx];

  return std::make_pair(curr, terminate(data, curr, len));
}

/*!
 * \brief Metapath-based random walk.
 * \param hg The heterograph.
 * \param seeds A 1D array of seed nodes, with the type the source type of the first
 *        edge type in the metapath.
 * \param metapath A 1D array of edge types representing the metapath.
 * \param prob A vector of 1D float arrays, indicating the transition probability of
 *        each edge by edge type.  An empty float array assumes uniform transition.
 * \param terminate Predicate for terminating a random walk path.
 * \return A 2D array of shape (len(seeds), len(metapath) + 1) with node IDs.
 */
template<DLDeviceType XPU, typename IdxType>
IdArray MetapathBasedRandomWalk(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    TerminatePredicate<IdxType> terminate) {
  int64_t max_num_steps = metapath->shape[0];
  const IdxType *metapath_data = static_cast<IdxType *>(metapath->data);

  // Prefetch all edges.
  // This forces the heterograph to materialize all OutCSR's before the OpenMP loop;
  // otherwise data races will happen.
  // TODO(BarclayII): should we later on materialize COO/CSR/CSC anyway unless told otherwise?
  std::vector<std::vector<IdArray> > edges_by_type;
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype)
    edges_by_type.push_back(hg->GetAdj(etype, true, "csr"));

  StepFunc<IdxType> step =
    [&edges_by_type, metapath_data, &prob, terminate]
    (IdxType *data, dgl_id_t curr, int64_t len) {
      return MetapathRandomWalkStep<XPU, IdxType>(
          data, curr, len, edges_by_type, metapath_data, prob, terminate);
    };

  return GenericRandomWalk<XPU, IdxType>(seeds, max_num_steps, step);
}

};  // namespace

};  // namespace impl

};  // namespace sampling

};  // namespace dgl

#endif  // DGL_GRAPH_SAMPLING_RANDOMWALKS_METAPATH_RANDOMWALK_H_
