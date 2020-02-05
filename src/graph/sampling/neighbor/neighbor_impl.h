/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/sampling/neighbor_impl.h
 * \brief Template implementation of neighborhood-based sampling algorithms.
 */
#ifndef DGL_GRAPH_SAMPLING_NEIGHBOR_NEIGHBOR_IMPL_H_
#define DGL_GRAPH_SAMPLING_NEIGHBOR_NEIGHBOR_IMPL_H_

#include <dgl/base_heterograph.h>
#include <dgl/array.h>
#include <vector>

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
    bool replace);

template<DLDeviceType XPU, typename IdxType>
HeteroGraphPtr SampleNeighborsTopk(
    const HeteroGraphPtr hg,
    const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& k,
    EdgeDir dir,
    const std::vector<FloatArray>& weight);

}  // namespace impl
}  // namespace sampling
}  // namespace dgl

#endif  // DGL_GRAPH_SAMPLING_NEIGHBOR_NEIGHBOR_IMPL_H_
