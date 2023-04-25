/**
 *  Copyright (c) 2018 by Contributors
 * @file graph/sampler/generic_randomwalk_cpu.h
 * @brief DGL sampler - templated implementation definition of random walks on
 *     CPU.
 */

#ifndef DGL_GRAPH_SAMPLING_RANDOMWALKS_METAPATH_RANDOMWALK_H_
#define DGL_GRAPH_SAMPLING_RANDOMWALKS_METAPATH_RANDOMWALK_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/random.h>

#include <tuple>
#include <utility>
#include <vector>

#include "randomwalks_cpu.h"
#include "randomwalks_impl.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

namespace {

template <typename IdxType>
using TerminatePredicate = std::function<bool(IdxType *, dgl_id_t, int64_t)>;

/**
 * @brief Select one successor of metapath-based random walk, given the path
 *     generated so far.
 *
 * @param data The path generated so far, of type \c IdxType.
 * @param curr The last node ID generated.
 * @param len The number of nodes generated so far.  Note that the seed node is
 *     always included as \c data[0], and the successors start from \c data[1].
 *
 * @param edges_by_type Vector of results from \c GetAdj() by edge type.
 * @param metapath_data Edge types of given metapath.
 * @param prob Transition probability per edge type.
 * @param terminate Predicate for terminating the current random walk path.
 *
 * @return A tuple of ID of next successor (-1 if not exist), the last traversed
 *     edge ID, as well as whether to terminate.
 */
template <DGLDeviceType XPU, typename IdxType>
std::tuple<dgl_id_t, dgl_id_t, bool> MetapathRandomWalkStep(
    IdxType *data, dgl_id_t curr, int64_t len,
    const std::vector<CSRMatrix> &edges_by_type,
    const std::vector<bool> &csr_has_data, const IdxType *metapath_data,
    const std::vector<FloatArray> &prob,
    TerminatePredicate<IdxType> terminate) {
  dgl_type_t etype = metapath_data[len];

  // Note that since the selection of successors is very lightweight (especially
  // in the uniform case), we want to reduce the overheads (even from object
  // copies or object construction) as much as possible. Using Successors()
  // slows down by 2x. Using OutEdges() slows down by 10x.
  const CSRMatrix &csr = edges_by_type[etype];
  const IdxType *offsets = csr.indptr.Ptr<IdxType>();
  const IdxType *all_succ = csr.indices.Ptr<IdxType>();
  const IdxType *all_eids =
      csr_has_data[etype] ? csr.data.Ptr<IdxType>() : nullptr;
  const IdxType *succ = all_succ + offsets[curr];
  const IdxType *eids = all_eids ? (all_eids + offsets[curr]) : nullptr;

  const int64_t size = offsets[curr + 1] - offsets[curr];
  if (size == 0) return std::make_tuple(-1, -1, true);

  // Use a reference to the original array instead of copying. This avoids
  // updating the ref counts atomically from different threads and avoids cache
  // ping-ponging in the tight loop.
  const FloatArray &prob_etype = prob[etype];
  IdxType idx = 0;
  if (IsNullArray(prob_etype)) {
    // empty probability array; assume uniform
    idx = RandomEngine::ThreadLocal()->RandInt(size);
  } else {
    ATEN_FLOAT_TYPE_SWITCH(prob_etype->dtype, DType, "probability", {
      FloatArray prob_selected =
          FloatArray::Empty({size}, prob_etype->dtype, prob_etype->ctx);
      DType *prob_selected_data = prob_selected.Ptr<DType>();
      const DType *prob_etype_data = prob_etype.Ptr<DType>();
      for (int64_t j = 0; j < size; ++j)
        prob_selected_data[j] =
            prob_etype_data[eids ? eids[j] : j + offsets[curr]];
      idx = RandomEngine::ThreadLocal()->Choice<IdxType>(prob_selected);
    });
  }
  dgl_id_t eid = eids ? eids[idx] : (idx + offsets[curr]);

  return std::make_tuple(succ[idx], eid, terminate(data, curr, len));
}

/**
 * @brief Select one successor of metapath-based random walk, given the path
 *     generated so far specifically for the uniform probability distribution.
 *
 * @param data The path generated so far, of type \c IdxType.
 * @param curr The last node ID generated.
 * @param len The number of nodes generated so far.  Note that the seed node is
 *     always included as \c data[0], and the successors start from \c data[1].
 *
 * @param edges_by_type Vector of results from \c GetAdj() by edge type.
 * @param metapath_data Edge types of given metapath.
 * @param prob Transition probability per edge type, for this special case this
 *     will be a NullArray.
 * @param terminate Predicate for terminating the current random walk path.
 *
 * @return A pair of ID of next successor (-1 if not exist), as well as whether
 *     to terminate. \note This function is called only if all the probability
 *     arrays are null.
 */
template <DGLDeviceType XPU, typename IdxType>
std::tuple<dgl_id_t, dgl_id_t, bool> MetapathRandomWalkStepUniform(
    IdxType *data, dgl_id_t curr, int64_t len,
    const std::vector<CSRMatrix> &edges_by_type,
    const std::vector<bool> &csr_has_data, const IdxType *metapath_data,
    const std::vector<FloatArray> &prob,
    TerminatePredicate<IdxType> terminate) {
  dgl_type_t etype = metapath_data[len];

  // Note that since the selection of successors is very lightweight (especially
  // in the uniform case), we want to reduce the overheads (even from object
  // copies or object construction) as much as possible. Using Successors()
  // slows down by 2x. Using OutEdges() slows down by 10x.
  const CSRMatrix &csr = edges_by_type[etype];
  const IdxType *offsets = csr.indptr.Ptr<IdxType>();
  const IdxType *all_succ = csr.indices.Ptr<IdxType>();
  const IdxType *all_eids =
      csr_has_data[etype] ? csr.data.Ptr<IdxType>() : nullptr;
  const IdxType *succ = all_succ + offsets[curr];
  const IdxType *eids = all_eids ? (all_eids + offsets[curr]) : nullptr;

  const int64_t size = offsets[curr + 1] - offsets[curr];
  if (size == 0) return std::make_tuple(-1, -1, true);

  IdxType idx = 0;
  // Guaranteed uniform distribution
  idx = RandomEngine::ThreadLocal()->RandInt(size);
  dgl_id_t eid = eids ? eids[idx] : (idx + offsets[curr]);

  return std::make_tuple(succ[idx], eid, terminate(data, curr, len));
}

/**
 * @brief Metapath-based random walk.
 * @param hg The heterograph.
 * @param seeds A 1D array of seed nodes, with the type the source type of the
 *     first edge type in the metapath.
 * @param metapath A 1D array of edge types representing the metapath.
 * @param prob A vector of 1D float arrays, indicating the transition
 *     probability of each edge by edge type.  An empty float array assumes
 *     uniform transition.
 * @param terminate Predicate for terminating a random walk path.
 * @return A 2D array of shape (len(seeds), len(metapath) + 1) with node IDs,
 *     and A 2D array of shape (len(seeds), len(metapath)) with edge IDs.
 */
template <DGLDeviceType XPU, typename IdxType>
std::pair<IdArray, IdArray> MetapathBasedRandomWalk(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    TerminatePredicate<IdxType> terminate) {
  int64_t max_num_steps = metapath->shape[0];
  const IdxType *metapath_data = static_cast<IdxType *>(metapath->data);
  const int64_t begin_ntype =
      hg->meta_graph()->FindEdge(metapath_data[0]).first;
  const int64_t max_nodes = hg->NumVertices(begin_ntype);

  // Prefetch all edges.
  // This forces the heterograph to materialize all OutCSR's before the OpenMP
  // loop; otherwise data races will happen.
  // TODO(BarclayII): should we later on materialize COO/CSR/CSC anyway unless
  // told otherwise?
  int64_t num_etypes = hg->NumEdgeTypes();
  std::vector<CSRMatrix> edges_by_type(num_etypes);
  std::vector<bool> csr_has_data(num_etypes);
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const CSRMatrix &csr = hg->GetCSRMatrix(etype);
    edges_by_type[etype] = csr;
    csr_has_data[etype] = CSRHasData(csr);
  }

  // Hoist the check for Uniform vs Non uniform edge distribution
  // to avoid putting it on the hot path
  bool isUniform = true;
  for (const auto &etype_prob : prob) {
    if (!IsNullArray(etype_prob)) {
      isUniform = false;
      break;
    }
  }
  if (!isUniform) {
    StepFunc<IdxType> step = [&edges_by_type, &csr_has_data, metapath_data,
                              &prob, terminate](
                                 IdxType *data, dgl_id_t curr, int64_t len) {
      return MetapathRandomWalkStep<XPU, IdxType>(
          data, curr, len, edges_by_type, csr_has_data, metapath_data, prob,
          terminate);
    };
    return GenericRandomWalk<XPU, IdxType>(
        seeds, max_num_steps, step, max_nodes);
  } else {
    StepFunc<IdxType> step = [&edges_by_type, &csr_has_data, metapath_data,
                              &prob, terminate](
                                 IdxType *data, dgl_id_t curr, int64_t len) {
      return MetapathRandomWalkStepUniform<XPU, IdxType>(
          data, curr, len, edges_by_type, csr_has_data, metapath_data, prob,
          terminate);
    };
    return GenericRandomWalk<XPU, IdxType>(
        seeds, max_num_steps, step, max_nodes);
  }
}

};  // namespace

};  // namespace impl

};  // namespace sampling

};  // namespace dgl

#endif  // DGL_GRAPH_SAMPLING_RANDOMWALKS_METAPATH_RANDOMWALK_H_
