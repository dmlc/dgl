/**
 *   Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 *   All rights reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 * @file cuda/unique_and_compact.h
 * @brief Unique and compact operator utilities on CUDA using hash table.
 */

#ifndef GRAPHBOLT_CUDA_UNIQUE_AND_COMPACT_H_
#define GRAPHBOLT_CUDA_UNIQUE_AND_COMPACT_H_

#include <torch/script.h>

#include <vector>

namespace graphbolt {
namespace ops {

std::vector<
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
UniqueAndCompactBatchedHashMapBased(
    const std::vector<torch::Tensor>& src_ids,
    const std::vector<torch::Tensor>& dst_ids,
    const std::vector<torch::Tensor>& unique_dst_ids, const int64_t rank,
    const int64_t world_size);

}  // namespace ops
}  // namespace graphbolt

#endif  // GRAPHBOLT_CUDA_UNIQUE_AND_COMPACT_H_
