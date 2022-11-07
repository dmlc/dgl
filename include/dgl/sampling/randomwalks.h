/**
 *  Copyright (c) 2019 by Contributors
 * @file dgl/samplinig/randomwalks.h
 * @brief Random walk functions.
 */
#ifndef DGL_SAMPLING_RANDOMWALKS_H_
#define DGL_SAMPLING_RANDOMWALKS_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>

#include <tuple>
#include <utility>
#include <vector>

namespace dgl {

namespace sampling {

/**
 * @brief Metapath-based random walk.
 * @param hg The heterograph.
 * @param seeds A 1D array of seed nodes, with the type the source type of the
 * first edge type in the metapath.
 * @param metapath A 1D array of edge types representing the metapath.
 * @param prob A vector of 1D float arrays, indicating the transition
 * probability of each edge by edge type. An empty float array assumes uniform
 * transition.
 * @return A pair of
 *         1. One 2D array of shape (len(seeds), len(metapath) + 1) with node
 *            IDs. The paths that terminated early are padded with -1.
 *         2. One 2D array of shape (len(seeds), len(metapath)) with edge IDs.
 *            The paths that terminated early are padded with -1.
 *         3. One 1D array of shape (len(metapath) + 1) with node type IDs.
 */
std::tuple<IdArray, IdArray, TypeArray> RandomWalk(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob);

/**
 * @brief Metapath-based random walk with restart probability.
 * @param hg The heterograph.
 * @param seeds A 1D array of seed nodes, with the type the source type of the
 * first edge type in the metapath.
 * @param metapath A 1D array of edge types representing the metapath.
 * @param prob A vector of 1D float arrays, indicating the transition
 * probability of each edge by edge type. An empty float array assumes uniform
 * transition.
 * @param restart_prob Restart probability.
 * @return A pair of
 *         1. One 2D array of shape (len(seeds), len(metapath) + 1) with node
 *            IDs. The paths that terminated early are padded with -1.
 *         2. One 2D array of shape (len(seeds), len(metapath)) with edge IDs.
 *            The paths that terminated early are padded with -1.
 *         3. One 1D array of shape (len(metapath) + 1) with node type IDs.
 */
std::tuple<IdArray, IdArray, TypeArray> RandomWalkWithRestart(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, double restart_prob);

/**
 * @brief Metapath-based random walk with stepwise restart probability. Useful
 *        for PinSAGE-like models.
 * @param hg The heterograph.
 * @param seeds A 1D array of seed nodes, with the type the source type of the
 * first edge type in the metapath.
 * @param metapath A 1D array of edge types representing the metapath.
 * @param prob A vector of 1D float arrays, indicating the transition
 * probability of each edge by edge type. An empty float array assumes uniform
 * transition.
 * @param restart_prob Restart probability array which has the same number of
 * elements as \c metapath, indicating the probability to terminate after
 * transition.
 * @return A pair of
 *         1. One 2D array of shape (len(seeds), len(metapath) + 1) with node
 *            IDs. The paths that terminated early are padded with -1.
 *         2. One 2D array of shape (len(seeds), len(metapath)) with edge IDs.
 *            The paths that terminated early are padded with -1.
 *         3. One 1D array of shape (len(metapath) + 1) with node type IDs.
 */
std::tuple<IdArray, IdArray, TypeArray> RandomWalkWithStepwiseRestart(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, FloatArray restart_prob);

};  // namespace sampling

};  // namespace dgl

#endif  // DGL_SAMPLING_RANDOMWALKS_H_
