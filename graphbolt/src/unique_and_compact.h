/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file unique_and_compact.h
 * @brief Unique and compact op.
 */
#ifndef GRAPHBOLT_SHM_UTILS_H_
#define GRAPHBOLT_SHM_UTILS_H_

#include <torch/torch.h>


namespace graphbolt {
namespace sampling {

std::vector<torch::Tensor> Unique_and_compact(
    const torch::Tensor& ids, int64_t num_seeds,
    const std::vector<torch::Tensor>& original_ids);

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_SHM_UTILS_H_
