/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/src/neighbor.cc
 * @brief Source file of neighbor sampling.
 */

#include <torch/torch.h>
#include <iostream>

namespace graphbolt {
namespace sampling {
   // Return 2 * N tensor represent COO graph(node pairs).
   torch::Tensor NeighborSampling(
    CSR mat, torch::Tensor rows, int64_t num_samples, bool replace);
}  // namespace sampling
}  // namespace dgl