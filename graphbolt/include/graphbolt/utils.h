/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/utils.h
 * @brief Header file of utils in graphbolt.
 */
#ifndef GRAPHBOLT_UTILS_H_
#define GRAPHBOLT_UTILS_H_

#include <omp.h>

namespace graphbolt {
namespace utils {

using RangePickFn = std::function<torch::Tensor(
    int64_t start, int64_t end, int64_t num_samples)>;

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

inline torch::Tensor UniformRangePickWithRepeat(
    int64_t start, int64_t end, int64_t num_samples) {
  return torch::randint(
      start, end,
      {
          num_samples,
      });
}

inline torch::Tensor UniformRangePickWithoutRepeat(
    int64_t start, int64_t end, int64_t num_samples) {
  auto perm = torch::randperm(end - start) + start;
  return perm.slice(0, 0, num_samples);
}

RangePickFn GetRangePickFn(
    const torch::optional<torch::Tensor>& probs, bool replace) {
  RangePickFn pick_fn;
  if (probs.has_value()) {
    if (probs.value().dtype() == torch::kBool) {
      pick_fn = [probs, replace](
                    int64_t start, int64_t end, int64_t num_samples) {
        auto local_probs = probs.value().slice(0, start, end);
        auto true_indices = local_probs.nonzero().view(-1);
        auto true_num = true_indices.size(0);
        auto choosed =
            replace ? UniformRangePickWithRepeat(0, true_num, num_samples)
                    : UniformRangePickWithoutRepeat(0, true_num, num_samples);
        return true_indices[choosed];
      };
    } else {
      pick_fn = [probs, replace](
                    int64_t start, int64_t end, int64_t num_samples) {
        auto local_probs = probs.value().slice(0, start, end);
        return torch::multinomial(local_probs, num_samples, replace) + start;
      };
    }
  } else {
    pick_fn =
        replace ? UniformRangePickWithRepeat : UniformRangePickWithoutRepeat;
  }
  return pick_fn;
}
}  // namespace utils
}  // namespace graphbolt

#endif
