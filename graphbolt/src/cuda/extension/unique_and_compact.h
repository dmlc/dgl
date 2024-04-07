/**
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/unique_and_compact.h
 * @brief Unique and compact operator utilities on CUDA using hash table.
 */

#ifndef GRAPHBOLT_CUDA_UNIQUE_AND_COMPACT_H_
#define GRAPHBOLT_CUDA_UNIQUE_AND_COMPACT_H_

#include <torch/script.h>

#include <vector>

namespace graphbolt {
namespace ops {

std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> >
UniqueAndCompactBatchedHashMapBased(
    const std::vector<torch::Tensor>& src_ids,
    const std::vector<torch::Tensor>& dst_ids,
    const std::vector<torch::Tensor>& unique_dst_ids);

}  // namespace ops
}  // namespace graphbolt

#endif  // GRAPHBOLT_CUDA_UNIQUE_AND_COMPACT_H_
