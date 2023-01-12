/**
 *  Copyright (c) 2018 by Contributors
 * @file graph/sampling/randomwalks_impl.h
 * @brief DGL sampler - templated implementation definition of random walks
 */

#ifndef DGL_GRAPH_SAMPLING_RANDOMWALKS_RANDOMWALKS_IMPL_H_
#define DGL_GRAPH_SAMPLING_RANDOMWALKS_RANDOMWALKS_IMPL_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>

#include <functional>
#include <tuple>
#include <utility>
#include <vector>

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

/**
 * @brief Random walk step function
 */
template <typename IdxType>
using StepFunc = std::function<
    //        ID        Edge ID   terminate?
    std::tuple<dgl_id_t, dgl_id_t, bool>(
        IdxType *,  // node IDs generated so far
        dgl_id_t,   // last node ID
        int64_t)>;  // # of steps

/**
 * @brief Get the node types traversed by the metapath.
 * @return A 1D array of shape (len(metapath) + 1,) with node type IDs.
 */
template <DGLDeviceType XPU, typename IdxType>
TypeArray GetNodeTypesFromMetapath(
    const HeteroGraphPtr hg, const TypeArray metapath);

/**
 * @brief Metapath-based random walk.
 * @param hg The heterograph.
 * @param seeds A 1D array of seed nodes, with the type the source type of the
 * first edge type in the metapath.
 * @param metapath A 1D array of edge types
 * representing the metapath.
 * @param prob A vector of 1D float arrays,
 * indicating the transition probability of each edge by edge type.  An empty
 * float array assumes uniform transition.
 * @return A 2D array of shape
 * (len(seeds), len(metapath) + 1) with node IDs.  The paths that terminated
 * early are padded with -1. A 2D array of shape (len(seeds), len(metapath))
 * with edge IDs.  The paths that terminated early are padded with -1. \note
 * This function should be called together with GetNodeTypesFromMetapath to
 *       determine the node type of each node in the random walk traces.
 */
template <DGLDeviceType XPU, typename IdxType>
std::pair<IdArray, IdArray> RandomWalk(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob);

/**
 * @brief Metapath-based random walk with restart probability.
 * @param hg The heterograph.
 * @param seeds A 1D array of seed nodes, with the type the source type of the
 * first edge type in the metapath.
 * @param metapath A 1D array of edge types
 * representing the metapath.
 * @param prob A vector of 1D float arrays,
 * indicating the transition probability of each edge by edge type.  An empty
 * float array assumes uniform transition.
 * @param restart_prob Restart
 * probability
 * @return A 2D array of shape (len(seeds), len(metapath) + 1) with
 * node IDs.  The paths that terminated early are padded with -1. A 2D array of
 * shape (len(seeds), len(metapath)) with edge IDs.  The paths that terminated
 * early are padded with -1. \note This function should be called together with
 * GetNodeTypesFromMetapath to determine the node type of each node in the
 * random walk traces.
 */
template <DGLDeviceType XPU, typename IdxType>
std::pair<IdArray, IdArray> RandomWalkWithRestart(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, double restart_prob);

/**
 * @brief Metapath-based random walk with stepwise restart probability.  Useful
 *        for PinSAGE-like models.
 * @param hg The heterograph.
 * @param seeds A 1D array of seed nodes, with the type the source type of the
 * first edge type in the metapath.
 * @param metapath A 1D array of edge types
 * representing the metapath.
 * @param prob A vector of 1D float arrays,
 * indicating the transition probability of each edge by edge type.  An empty
 * float array assumes uniform transition.
 * @param restart_prob Restart
 * probability array which has the same number of elements as \c metapath,
 * indicating the probability to terminate after transition.
 * @return A 2D array
 * of shape (len(seeds), len(metapath) + 1) with node IDs.  The paths that
 * terminated early are padded with -1. A 2D array of shape (len(seeds),
 * len(metapath)) with edge IDs.  The paths that terminated early are padded
 * with -1. \note This function should be called together with
 * GetNodeTypesFromMetapath to determine the node type of each node in the
 * random walk traces.
 */
template <DGLDeviceType XPU, typename IdxType>
std::pair<IdArray, IdArray> RandomWalkWithStepwiseRestart(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, FloatArray restart_prob);

template <DGLDeviceType XPU, typename IdxType>
std::tuple<IdArray, IdArray, IdArray> SelectPinSageNeighbors(
    const IdArray src, const IdArray dst, const int64_t num_samples_per_node,
    const int64_t k);

};  // namespace impl

};  // namespace sampling

};  // namespace dgl

#endif  // DGL_GRAPH_SAMPLING_RANDOMWALKS_RANDOMWALKS_IMPL_H_
