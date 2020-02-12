/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/sampling/neighbor_cpu.cc
 * \brief CPU implementation of neighborhood-based sampling algorithms.
 */

#include <dgl/random.h>
#include <numeric>
#include "./neighbor_cpu.h"
#include "./neighbor_impl.h"

namespace dgl {
namespace sampling {
namespace impl {

template<DLDeviceType XPU, typename IdxType>
HeteroGraphPtr SampleNeighbors(
    const HeteroGraphPtr hg,
    const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& fanouts,
    EdgeDir dir,
    const std::vector<FloatArray>& prob,
    bool replace) {
  ChooseFunc<IdxType> choose_fn =
    [replace] (int64_t num, int64_t population, const FloatArray prob) {
      if (prob->shape[0] == 0) {
        // empty prob array; assume uniform
        return RandomEngine::ThreadLocal()->UniformChoice<IdxType>(num, population, replace);
      } else {
        return RandomEngine::ThreadLocal()->Choice<IdxType>(num, prob, replace);
      }
    };
  return CPUSampleNeighbors<IdxType>(hg, nodes, fanouts, dir, prob, replace, choose_fn);
}

template HeteroGraphPtr SampleNeighbors<kDLCPU, int32_t>(
    const HeteroGraphPtr hg,
    const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& fanouts,
    EdgeDir dir,
    const std::vector<FloatArray>& prob,
    bool replace);

template HeteroGraphPtr SampleNeighbors<kDLCPU, int64_t>(
    const HeteroGraphPtr hg,
    const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& fanouts,
    EdgeDir dir,
    const std::vector<FloatArray>& prob,
    bool replace);

template<DLDeviceType XPU, typename IdxType>
HeteroGraphPtr SampleNeighborsTopk(
    const HeteroGraphPtr hg,
    const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& k,
    EdgeDir dir,
    const std::vector<FloatArray>& weight) {
  ChooseFunc<IdxType> choose_fn =
    [] (int64_t num, int64_t population, const FloatArray w) {
      IdArray ret;
      ATEN_FLOAT_TYPE_SWITCH(w->dtype, DType, "edge_weight", {
        const DType* w_data = static_cast<DType*>(w->data);
        std::vector<IdxType> idx(population);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [w_data] (IdxType i, IdxType j) { return w_data[i] > w_data[j]; });
        idx.resize(num);
        ret = IdArray::FromVector(idx);
      });
      return ret;
    };
  return CPUSampleNeighbors<IdxType>(hg, nodes, k, dir, weight, false, choose_fn);
}

template HeteroGraphPtr SampleNeighborsTopk<kDLCPU, int32_t>(
    const HeteroGraphPtr hg,
    const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& k,
    EdgeDir dir,
    const std::vector<FloatArray>& weight);

template HeteroGraphPtr SampleNeighborsTopk<kDLCPU, int64_t>(
    const HeteroGraphPtr hg,
    const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& k,
    EdgeDir dir,
    const std::vector<FloatArray>& weight);

}  // namespace impl
}  // namespace sampling
}  // namespace dgl
