/**
 *  Copyright (c) 2021 by Contributors
 * @file graph/sampling/node2vec_randomwalk.cc
 * @brief DGL sampler - CPU implementation of node2vec random walk.
 */

#ifndef DGL_GRAPH_SAMPLING_RANDOMWALKS_NODE2VEC_RANDOMWALK_H_
#define DGL_GRAPH_SAMPLING_RANDOMWALKS_NODE2VEC_RANDOMWALK_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/random.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <tuple>
#include <utility>
#include <vector>

#include "metapath_randomwalk.h"  // for TerminatePredicate
#include "node2vec_impl.h"
#include "randomwalks_cpu.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

namespace {

template <typename IdxType>
bool has_edge_between(const CSRMatrix &csr, dgl_id_t u, dgl_id_t v) {
  const IdxType *offsets = csr.indptr.Ptr<IdxType>();
  const IdxType *all_succ = csr.indices.Ptr<IdxType>();
  const IdxType *u_succ = all_succ + offsets[u];
  const int64_t size = offsets[u + 1] - offsets[u];

  if (csr.sorted)
    return std::binary_search(u_succ, u_succ + size, v);
  else
    return std::find(u_succ, u_succ + size, v) != u_succ + size;
}

/**
 * @brief Node2vec random walk step function
 * @param data The path generated so far, of type \c IdxType.
 * @param curr The last node ID generated.
 * @param pre The last last node ID generated
 * @param p Float, indicating likelihood of immediately revisiting a node in the
 *        walk.
 * @param q Float, control parameter to interpolate between breadth-first
 *        strategy and depth-first strategy.
 * @param len The number of nodes generated so far.  Note that the seed node is
 * always included as \c data[0], and the successors start from \c data[1].
 * @param csr The CSR matrix
 * @param prob Transition probability
 * @param terminate Predicate for terminating the current random walk path.
 * @return A tuple of ID of next successor (-1 if not exist), the edge ID
 * traversed, as well as whether to terminate.
 */

template <DGLDeviceType XPU, typename IdxType>
std::tuple<dgl_id_t, dgl_id_t, bool> Node2vecRandomWalkStep(
    IdxType *data, dgl_id_t curr, dgl_id_t pre, const double p, const double q,
    int64_t len, const CSRMatrix &csr, bool csr_has_data,
    const FloatArray &probs, TerminatePredicate<IdxType> terminate) {
  const IdxType *offsets = csr.indptr.Ptr<IdxType>();
  const IdxType *all_succ = csr.indices.Ptr<IdxType>();
  const IdxType *all_eids = csr_has_data ? csr.data.Ptr<IdxType>() : nullptr;
  const IdxType *succ = all_succ + offsets[curr];
  const IdxType *eids = all_eids ? (all_eids + offsets[curr]) : nullptr;

  const int64_t size = offsets[curr + 1] - offsets[curr];

  // Isolated node
  if (size == 0) return std::make_tuple(-1, -1, true);

  IdxType idx = 0;

  // Normalize the weights to compute rejection probabilities
  double max_prob = std::max({1 / p, 1.0, 1 / q});
  // rejection prob for back to the previous node
  double prob0 = 1 / p / max_prob;
  // rejection prob for visiting the node with the distance of 1 between the
  // previous node
  double prob1 = 1 / max_prob;
  // rejection prob for visiting the node with the distance of 2 between the
  // previous node
  double prob2 = 1 / q / max_prob;
  dgl_id_t next_node;
  double r;  // rejection probability.
  if (IsNullArray(probs)) {
    if (len == 0) {
      idx = RandomEngine::ThreadLocal()->RandInt(size);
      next_node = succ[idx];
    } else {
      while (true) {
        idx = RandomEngine::ThreadLocal()->RandInt(size);
        r = RandomEngine::ThreadLocal()->Uniform(0., 1.);
        next_node = succ[idx];
        if (next_node == pre) {
          if (r < prob0) break;
        } else if (has_edge_between<IdxType>(csr, next_node, pre)) {
          if (r < prob1) break;
        } else if (r < prob2) {
          break;
        }
      }
    }
  } else {
    FloatArray prob_selected;
    ATEN_FLOAT_TYPE_SWITCH(probs->dtype, DType, "probability", {
      prob_selected = FloatArray::Empty({size}, probs->dtype, probs->ctx);
      DType *prob_selected_data = prob_selected.Ptr<DType>();
      const DType *prob_etype_data = probs.Ptr<DType>();
      for (int64_t j = 0; j < size; ++j)
        prob_selected_data[j] =
            prob_etype_data[eids ? eids[j] : j + offsets[curr]];
    });

    if (len == 0) {
      idx = RandomEngine::ThreadLocal()->Choice<IdxType>(prob_selected);
      next_node = succ[idx];
    } else {
      while (true) {
        idx = RandomEngine::ThreadLocal()->Choice<IdxType>(prob_selected);
        r = RandomEngine::ThreadLocal()->Uniform(0., 1.);
        next_node = succ[idx];
        if (next_node == pre) {
          if (r < prob0) break;
        } else if (has_edge_between<IdxType>(csr, next_node, pre)) {
          if (r < prob1) break;
        } else if (r < prob2) {
          break;
        }
      }
    }
  }
  dgl_id_t eid = eids ? eids[idx] : (idx + offsets[curr]);

  return std::make_tuple(next_node, eid, terminate(data, next_node, len));
}

template <DGLDeviceType XPU, typename IdxType>
std::pair<IdArray, IdArray> Node2vecRandomWalk(
    const HeteroGraphPtr g, const IdArray seeds, const double p, const double q,
    const int64_t max_num_steps, const FloatArray &prob,
    TerminatePredicate<IdxType> terminate) {
  const CSRMatrix &edges = g->GetCSRMatrix(0);  // homogeneous graph.
  bool csr_has_data = CSRHasData(edges);

  StepFunc<IdxType> step = [&edges, csr_has_data, &prob, p, q, terminate](
                               IdxType *data, dgl_id_t curr, int64_t len) {
    dgl_id_t pre = (len != 0) ? data[len - 1] : curr;
    return Node2vecRandomWalkStep<XPU, IdxType>(
        data, curr, pre, p, q, len, edges, csr_has_data, prob, terminate);
  };

  return GenericRandomWalk<XPU, IdxType>(
      seeds, max_num_steps, step, g->NumVertices(0));
}

};  // namespace

};  // namespace impl

};  // namespace sampling

};      // namespace dgl
#endif  // DGL_GRAPH_SAMPLING_RANDOMWALKS_NODE2VEC_RANDOMWALK_H_
