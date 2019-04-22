/*!
 *  Copyright (c) 2018 by Contributors
 * \file dgl/sampler.h
 * \brief DGL sampler header.
 */
#ifndef DGL_SAMPLER_H_
#define DGL_SAMPLER_H_

#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include "graph_interface.h"
#include "nodeflow.h"

#ifdef _MSC_VER
// rand in MS compiler works well in multi-threading.
inline int rand_r(unsigned *seed) {
  return rand();
}

inline unsigned int randseed() {
  unsigned int seed = time(nullptr);
  srand(seed);  // need to set seed manually since there's no rand_r
  return seed;
}
#define _CRT_RAND_S
#else
inline unsigned int randseed() {
  return time(nullptr);
}
#endif

namespace dgl {

class ImmutableGraph;

struct RandomWalkTraces {
  /*! \brief number of traces generated for each seed */
  IdArray trace_counts;
  /*! \brief length of each trace, concatenated */
  IdArray trace_lengths;
  /*! \brief the vertices, concatenated */
  IdArray vertices;
};

class SamplerOp {
 public:
  /*!
   * \brief Sample a graph from the seed vertices with neighbor sampling.
   * The neighbors are sampled with a uniform distribution.
   *
   * \param graphs A graph for sampling.
   * \param seeds the nodes where we should start to sample.
   * \param edge_type the type of edges we should sample neighbors.
   * \param num_hops the number of hops to sample neighbors.
   * \param expand_factor the max number of neighbors to sample.
   * \param add_self_loop whether to add self loop to the sampled subgraph
   * \return a NodeFlow graph.
   */
  static NodeFlow NeighborUniformSample(const ImmutableGraph *graph,
                                        const std::vector<dgl_id_t>& seeds,
                                        const std::string &edge_type,
                                        int num_hops, int expand_factor,
                                        const bool add_self_loop);

  /*!
   * \brief Sample a graph from the seed vertices with layer sampling.
   * The layers are sampled with a uniform distribution.
   *
   * \param graphs A graph for sampling.
   * \param seeds the nodes where we should start to sample.
   * \param edge_type the type of edges we should sample neighbors.
   * \param layer_sizes The size of layers.
   * \return a NodeFlow graph.
   */
  static NodeFlow LayerUniformSample(const ImmutableGraph *graph,
                                     const std::vector<dgl_id_t>& seeds,
                                     const std::string &neigh_type,
                                     IdArray layer_sizes);
};

/*!
 * \brief Batch-generate random walk traces
 * \param seeds The array of starting vertex IDs
 * \param num_traces The number of traces to generate for each seed
 * \param num_hops The number of hops for each trace
 * \return a flat ID array with shape (num_seeds, num_traces, num_hops + 1)
 */
IdArray RandomWalk(const GraphInterface *gptr,
                   IdArray seeds,
                   int num_traces,
                   int num_hops);

/*!
 * \brief Batch-generate random walk traces with restart
 *
 * Stop generating traces if max_frequrent_visited_nodes nodes are visited more than
 * max_visit_counts times.
 *
 * \param seeds The array of starting vertex IDs
 * \param restart_prob The restart probability
 * \param visit_threshold_per_seed Stop generating more traces once the number of nodes
 * visited for a seed exceeds this number.  (Algorithm 1 in [1])
 * \param max_visit_counts Alternatively, stop generating traces for a seed if no less
 * than \c max_frequent_visited_nodes are visited no less than \c max_visit_counts
 * times.  (Algorithm 2 in [1])
 * \param max_frequent_visited_nodes See \c max_visit_counts
 * \return A RandomWalkTraces instance.
 *
 * \sa [1] Eksombatchai et al., 2017 https://arxiv.org/abs/1711.07601
 */
RandomWalkTraces RandomWalkWithRestart(
    const GraphInterface *gptr,
    IdArray seeds,
    double restart_prob,
    uint64_t visit_threshold_per_seed,
    uint64_t max_visit_counts,
    uint64_t max_frequent_visited_nodes);

/*
 * \brief Batch-generate random walk traces with restart on a bipartite graph, walking two
 * hops at a time.
 *
 * Since it is walking on a bipartite graph, the vertices of a trace will always stay on the
 * same side.
 *
 * Stop generating traces if max_frequrent_visited_nodes nodes are visited more than
 * max_visit_counts times.
 *
 * \param seeds The array of starting vertex IDs
 * \param restart_prob The restart probability
 * \param visit_threshold_per_seed Stop generating more traces once the number of nodes
 * visited for a seed exceeds this number.  (Algorithm 1 in [1])
 * \param max_visit_counts Alternatively, stop generating traces for a seed if no less
 * than \c max_frequent_visited_nodes are visited no less than \c max_visit_counts
 * times.  (Algorithm 2 in [1])
 * \param max_frequent_visited_nodes See \c max_visit_counts
 * \return A RandomWalkTraces instance.
 *
 * \note Doesn't verify whether the graph is indeed a bipartite graph
 *
 * \sa [1] Eksombatchai et al., 2017 https://arxiv.org/abs/1711.07601
 */
RandomWalkTraces BipartiteSingleSidedRandomWalkWithRestart(
    const GraphInterface *gptr,
    IdArray seeds,
    double restart_prob,
    uint64_t visit_threshold_per_seed,
    uint64_t max_visit_counts,
    uint64_t max_frequent_visited_nodes);

}  // namespace dgl

#endif  // DGL_SAMPLER_H_
