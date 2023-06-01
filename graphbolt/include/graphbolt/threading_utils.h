/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/utils.h
 * @brief Threading utils in graphbolt.
 */
#ifndef GRAPHBOLT_THREADING_UTILS_H_
#define GRAPHBOLT_THREADING_UTILS_H_

#include <omp.h>

namespace graphbolt {
namespace runtime {

inline size_t compute_num_threads(size_t begin, size_t end, size_t grain_size) {
#ifdef _OPENMP
  if (omp_in_parallel() || end - begin <= grain_size || end - begin == 1)
    return 1;

  return std::min(
      static_cast<int64_t>(torch::get_num_threads()),
      torch::divup(end - begin, grain_size));
#else
  return 1;
#endif
}

}  // namespace runtime
}  // namespace graphbolt

#endif  // GRAPHBOLT_THREADING_UTILS_H_
