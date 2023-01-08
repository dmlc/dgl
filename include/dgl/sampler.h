/**
 *  Copyright (c) 2018 by Contributors
 * @file dgl/sampler.h
 * @brief DGL sampler header.
 */
#ifndef DGL_SAMPLER_H_
#define DGL_SAMPLER_H_

#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

#include "graph_interface.h"
#include "nodeflow.h"

namespace dgl {

class ImmutableGraph;

class SamplerOp {
 public:
  /**
   * @brief Sample a graph from the seed vertices with neighbor sampling.
   * The neighbors are sampled with a uniform distribution.
   *
   * @param graph A graph for sampling.
   * @param seeds the nodes where we should start to sample.
   * @param edge_type the type of edges we should sample neighbors.
   * @param num_hops the number of hops to sample neighbors.
   * @param expand_factor the max number of neighbors to sample.
   * @param add_self_loop whether to add self loop to the sampled subgraph
   * @param probability the transition probability (float/double).
   * @return a NodeFlow graph.
   */
  template <typename ValueType>
  static NodeFlow NeighborSample(
      const ImmutableGraph *graph, const std::vector<dgl_id_t> &seeds,
      const std::string &edge_type, int num_hops, int expand_factor,
      const bool add_self_loop, const ValueType *probability);

  /**
   * @brief Sample a graph from the seed vertices with layer sampling.
   * The layers are sampled with a uniform distribution.
   *
   * @param graph A graph for sampling.
   * @param seeds the nodes where we should start to sample.
   * @param edge_type the type of edges we should sample neighbors.
   * @param layer_sizes The size of layers.
   * @return a NodeFlow graph.
   */
  static NodeFlow LayerUniformSample(
      const ImmutableGraph *graph, const std::vector<dgl_id_t> &seeds,
      const std::string &neigh_type, IdArray layer_sizes);
};

}  // namespace dgl

#endif  // DGL_SAMPLER_H_
