/*!
 *  Copyright (c) 2018 by Contributors
 * \file dgl/sampler.h
 * \brief DGL sampler header.
 */
#ifndef DGL_SAMPLER_H_
#define DGL_SAMPLER_H_

#include <vector>
#include <string>
#include "graph_interface.h"
#include "nodeflow.h"

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

  /*!
   * \brief Batch-generate random walk traces
   * \param seeds The array of starting vertex IDs
   * \param num_traces The number of traces to generate for each seed
   * \param num_hops The number of hops for each trace
   * \return a flat ID array with shape (num_seeds, num_traces, num_hops + 1)
   */
  static IdArray RandomWalk(const GraphInterface *gptr,
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
   * \param max_nodes_per_seed Stop generating traces if this many nodes are visited for each seed
   * \param num_hops The number of hops for each trace
   * \return a flat ID array with shape (num_seeds, num_traces, num_hops + 1)
   */
  static RandomWalkTraces RandomWalkWithRestart(
      const GraphInterface *gptr,
      IdArray seeds,
      float restart_prob,
      uint64_t max_nodes_per_seed,
      uint64_t max_visit_counts,
      uint64_t max_frequent_visited_nodes);

  /*
   * \note Doesn't verify whether the graph is indeed a bipartite graph
   */
  static RandomWalkTraces BipartiteSingleSidedRandomWalkWithRestart(
      const GraphInterface *gptr,
      IdArray seeds,
      float restart_prob,
      uint64_t max_nodes_per_seed,
      uint64_t max_visit_counts,
      uint64_t max_frequent_visited_nodes);

private:
};

}  // namespace dgl

#endif  // DGL_SAMPLER_H_
