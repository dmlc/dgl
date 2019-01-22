/*!
 *  Copyright (c) 2018 by Contributors
 * \file dgl/sampler.h
 * \brief DGL sampler header.
 */
#ifndef DGL_SAMPLER_H_
#define DGL_SAMPLER_H_

#include "graph_interface.h"

namespace dgl {

/*!
 * \brief When we sample a subgraph, we need to store extra information,
 * such as the layer Ids of the vertices and the sampling probability.
 */
struct SampledSubgraph: public Subgraph {
  /*!
   * \brief the offsets of the layers in the subgraph.
   */
  IdArray layer_offsets;
  /*!
   * \brief the probability that a vertex is sampled.
   */
  runtime::NDArray sample_prob;
};

}  // dgl

#endif  // DGL_SAMPLER_H_
