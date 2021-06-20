/*!
 *  Copyright (c) 2021 by Contributors
 * \file runtime/container.h
 * \brief Defines the container object data structures.
 */
#ifndef DGL_RUNTIME_PARALLEL_FOR_H_
#define DGL_RUNTIME_PARALLEL_FOR_H_

#include <dmlc/omp.h>
#include <algorithm>
#include <string>
#include <cstdlib>

namespace {
int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}
}

namespace dgl {
namespace runtime {
namespace {
size_t compute_num_threads(size_t begin, size_t end, size_t grain_size) {
  if (omp_in_parallel() || end - begin <= grain_size || end - begin == 1)
    return 1;

  return std::min(static_cast<int64_t>(omp_get_max_threads()), divup(end - begin, grain_size));
}

struct DefaultGrainSizeT {
  size_t grain_size;

  DefaultGrainSizeT() {
    auto var = std::getenv("DGL_PARALLEL_FOR_GRAIN_SIZE");

    if (!var) {
      grain_size = 1;
    } else {
      grain_size = std::stoul(var);
    }
  }

  size_t operator()() {
    return grain_size;
  }
};
}  // namespace

static DefaultGrainSizeT default_grain_size;

/*!
 * \brief OpenMP-based parallel for loop.
 *
 * It requires each thread's workload to have at least \a grain_size elements.
 * The loop body will be a function that takes in a single argument \a i, which
 * stands for the index of the workload.
 */
template <typename F>
void parallel_for(
    const size_t begin,
    const size_t end,
    const size_t grain_size,
    F&& f) {
  if (begin >= end) {
    return;
  }

#ifdef _OPENMP
  auto num_threads = compute_num_threads(begin, end, grain_size);

#pragma omp parallel num_threads(num_threads)
  {
    auto tid = omp_get_thread_num();
    auto chunk_size = divup((end - begin), num_threads);
    auto begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
      for (auto i = begin_tid; i < std::min(end, chunk_size + begin_tid); i++) {
        f(i);
      }
    }
  }
#else
  for (auto i = begin; i < end; i++)
    f(i);
#endif
}

/*!
 * \brief OpenMP-based parallel for loop with default grain size.
 *
 * parallel_for with grain size to default value, either 1 or controlled through
 * environment variable DGL_PARALLEL_FOR_GRAIN_SIZE.
 * If grain size is set to 1, the function behaves the same way as OpenMP
 * parallel for pragma with static scheduling.
 */
template <typename F>
void parallel_for(
    const size_t begin,
    const size_t end,
    F&& f) {
  parallel_for(begin, end, default_grain_size(), f);
}
}  // namespace runtime
}  // namespace dgl

#endif  // DGL_RUNTIME_PARALLEL_FOR_H_
