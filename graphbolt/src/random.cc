/**
 *  Copyright (c) 2023 by Contributors
 * @file random.cc
 * @brief Random Engine.
 */

#include "./random.h"

#include <torch/torch.h>

namespace graphbolt {

void SetSeed(int64_t seed) {
  int64_t num_threads = torch::get_num_threads();
  torch::parallel_for(0, num_threads, 1, [&](size_t start, size_t end) {
    for (auto i = start; i < end; ++i) {
      RandomEngine::ThreadLocal()->SetSeed(seed);
    }
  });
}

}  // namespace graphbolt
