/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/include/compact.h
 * @brief Header file of compact.
 */

#include <torch/torch.h>
#include <vector>

namespace graphbolt {
namespace sampling {
   // Input is 2 * N tensor and output is a compacted 2 * N tensor + 1D tensor of unique Ids.
   std::vector<torch::Tensor> Compact(
    CSR mat, torch::Tensor rows, int64_t num_samples, bool replace);
}  // namespace sampling
}  // namespace dgl