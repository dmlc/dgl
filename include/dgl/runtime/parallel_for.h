/**
 *  Copyright (c) 2021 by Contributors
 * @file runtime/container.h
 * @brief Defines the container object data structures.
 */
#ifndef DGL_RUNTIME_PARALLEL_FOR_H_
#define DGL_RUNTIME_PARALLEL_FOR_H_

#include <dgl/env_variable.h>
#include <dmlc/omp.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace {
int64_t divup(int64_t x, int64_t y) { return (x + y - 1) / y; }
}  // namespace

namespace dgl {
namespace runtime {
namespace {
struct DefaultGrainSizeT {
  size_t grain_size;

  DefaultGrainSizeT() : DefaultGrainSizeT(1) {}

  explicit DefaultGrainSizeT(size_t default_grain_size) {
    auto var = dgl::kDGLParallelForGrainSize;

    if (var) {
      grain_size = std::stoul(var);
    } else {
      grain_size = default_grain_size;
    }
  }

  size_t operator()() { return grain_size; }
};
}  // namespace

inline size_t compute_num_threads(size_t begin, size_t end, size_t grain_size) {
#ifdef _OPENMP
  if (omp_in_parallel() || end - begin <= grain_size || end - begin == 1)
    return 1;

  return std::min(
      static_cast<int64_t>(omp_get_max_threads()),
      divup(end - begin, grain_size));
#else
  return 1;
#endif
}

static DefaultGrainSizeT default_grain_size;

/**
 * @brief OpenMP-based parallel for loop.
 *
 * It requires each thread's workload to have at least \a grain_size elements.
 * The loop body will be a function that takes in two arguments \a begin and \a
 * end, which stands for the starting (inclusive) and ending index (exclusive)
 * of the workload.
 */
template <typename F>
void parallel_for(
    const size_t begin, const size_t end, const size_t grain_size, F&& f) {
  if (begin >= end) {
    return;
  }

#ifdef _OPENMP
  auto num_threads = compute_num_threads(begin, end, grain_size);
  // (BarclayII) the exception code is borrowed from PyTorch.
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;

#pragma omp parallel num_threads(num_threads)
  {
    auto tid = omp_get_thread_num();
    auto chunk_size = divup((end - begin), num_threads);
    auto begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
      auto end_tid = std::min(end, chunk_size + begin_tid);
      try {
        f(begin_tid, end_tid);
      } catch (...) {
        if (!err_flag.test_and_set()) eptr = std::current_exception();
      }
    }
  }
  if (eptr) std::rethrow_exception(eptr);
#else
  f(begin, end);
#endif
}

/**
 * @brief OpenMP-based parallel for loop with default grain size.
 *
 * parallel_for with grain size to default value, either 1 or controlled through
 * environment variable DGL_PARALLEL_FOR_GRAIN_SIZE.
 * If grain size is set to 1, the function behaves the same way as OpenMP
 * parallel for pragma with static scheduling.
 */
template <typename F>
void parallel_for(const size_t begin, const size_t end, F&& f) {
  parallel_for(begin, end, default_grain_size(), std::forward<F>(f));
}

/**
 * @brief OpenMP-based two-stage parallel reduction.
 *
 * The first-stage reduction function \a f works in parallel.  Each thread's
 * workload has at least \a grain_size elements.  The loop body will be a
 * function that takes in the starting index (inclusive), the ending index
 * (exclusive), and the reduction identity.
 *
 * The second-stage reduction function \a sf is a binary function working in the
 * main thread. It aggregates the partially reduced result computed from each
 * thread.
 *
 * Example to compute a parallelized max reduction of an array \c a:
 *
 *     parallel_reduce(
 *       0,        // starting index
 *       100,      // ending index
 *       1,        // grain size
 *       -std::numeric_limits<float>::infinity,     // identity
 *       [&a] (int begin, int end, float ident) {   // first-stage partial
 * reducer float result = ident; for (int i = begin; i < end; ++i) result =
 * std::max(result, a[i]); return result;
 *       },
 *       [] (float result, float partial_result) {
 *         return std::max(result, partial_result);
 *       });
 */
template <typename DType, typename F, typename SF>
DType parallel_reduce(
    const size_t begin, const size_t end, const size_t grain_size,
    const DType ident, const F& f, const SF& sf) {
  if (begin >= end) {
    return ident;
  }

  int num_threads = compute_num_threads(begin, end, grain_size);
  if (num_threads == 1) {
    return f(begin, end, ident);
  }

  std::vector<DType> results(num_threads, ident);
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;
#pragma omp parallel num_threads(num_threads)
  {
    auto tid = omp_get_thread_num();
    auto chunk_size = divup((end - begin), num_threads);
    auto begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
      auto end_tid = std::min(end, static_cast<size_t>(chunk_size + begin_tid));
      try {
        results[tid] = f(begin_tid, end_tid, ident);
      } catch (...) {
        if (!err_flag.test_and_set()) eptr = std::current_exception();
      }
    }
  }
  if (eptr) std::rethrow_exception(eptr);

  DType out = ident;
  for (int64_t i = 0; i < num_threads; ++i) out = sf(out, results[i]);
  return out;
}

}  // namespace runtime
}  // namespace dgl

#endif  // DGL_RUNTIME_PARALLEL_FOR_H_
