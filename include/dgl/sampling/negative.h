/**
 *  Copyright (c) 2020 by Contributors
 * @file dgl/sampling/negative.h
 * @brief Negative sampling.
 */
#ifndef DGL_SAMPLING_NEGATIVE_H_
#define DGL_SAMPLING_NEGATIVE_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>

#include <utility>

namespace dgl {
namespace sampling {

/**
 * @brief Given an edge type, uniformly sample source-destination pairs that do
 * not have an edge in between using rejection sampling.
 *
 * @note This function may not return the same number of elements as the given
 * number of samples.
 * @note This function requires sorting the CSR or CSC matrix of the graph
 * in-place.  It prefers CSC over CSR.
 *
 * @param hg The graph.
 * @param etype The edge type.
 * @param num_samples The number of negative examples to sample.
 * @param num_trials The number of rejection sampling trials.
 * @param exclude_self_loops Do not include the examples where the source equals
 * the destination.
 * @param replace Whether to sample with replacement.
 * @param redundancy How much redundant negative examples to take in case of
 * duplicate examples.
 * @return The pair of source and destination tensors.
 */
std::pair<IdArray, IdArray> GlobalUniformNegativeSampling(
    HeteroGraphPtr hg, dgl_type_t etype, int64_t num_samples, int num_trials,
    bool exclude_self_loops, bool replace, double redundancy);

};  // namespace sampling
};  // namespace dgl

#endif  // DGL_SAMPLING_NEGATIVE_H_
