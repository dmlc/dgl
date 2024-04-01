/**
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/unique_and_compact_map.cu
 * @brief Unique and compact operator implementation on CUDA using hash table.
 */

#ifndef GRAPHBOLT_CUDA_UNIQUE_AND_COMPACT_H_
#define GRAPHBOLT_CUDA_UNIQUE_AND_COMPACT_H_

#include <nv/target>  // __CUDA_MINIMUM_ARCH__ and friends
#if defined(__CUDA_MINIMUM_ARCH__) &&                       \
    ((!defined(_MSC_VER) && __CUDA_MINIMUM_ARCH__ < 600) || \
     (defined(_MSC_VER) && __CUDA_MINIMUM_ARCH__ < 700))
#else
#define USE_MAP_BASED_IMPL

#include <torch/script.h>

#include <vector>

std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> >
UniqueAndCompactBatchedMap(
    const std::vector<torch::Tensor>& src_ids,
    const std::vector<torch::Tensor>& dst_ids,
    const std::vector<torch::Tensor>& unique_dst_ids);
#endif

#endif  // GRAPHBOLT_CUDA_UNIQUE_AND_COMPACT_H_
