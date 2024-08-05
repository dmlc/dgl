/**
 *   Copyright (c) 2024, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
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
 * @file graphbolt/async.h
 * @brief Provides asynchronous task utilities for GraphBolt.
 */
#ifndef GRAPHBOLT_ASYNC_H_
#define GRAPHBOLT_ASYNC_H_

#include <ATen/Parallel.h>
#include <torch/script.h>

#include <atomic>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <type_traits>

#ifdef BUILD_WITH_TASKFLOW
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/taskflow.hpp>
#endif

namespace graphbolt {

enum ThreadPool { intraop, interop };

#ifdef BUILD_WITH_TASKFLOW

template <ThreadPool pool_type>
inline tf::Executor& _get_thread_pool() {
  static std::unique_ptr<tf::Executor> pool;
  static std::once_flag flag;
  std::call_once(flag, [&] {
    const int num_threads = pool_type == ThreadPool::intraop
                                ? torch::get_num_threads()
                                : torch::get_num_interop_threads();
    pool = std::make_unique<tf::Executor>(num_threads);
  });
  return *pool.get();
}

inline tf::Executor& intraop_pool() {
  return _get_thread_pool<ThreadPool::intraop>();
}

inline tf::Executor& interop_pool() {
  return _get_thread_pool<ThreadPool::interop>();
}

inline tf::Executor& get_thread_pool(ThreadPool pool_type) {
  return pool_type == ThreadPool::intraop ? intraop_pool() : interop_pool();
}
#endif  // BUILD_WITH_TASKFLOW

inline int get_num_threads() {
#ifdef BUILD_WITH_TASKFLOW
  return intraop_pool().num_workers();
#else
  return torch::get_num_threads();
#endif
}

inline int get_num_interop_threads() {
#ifdef BUILD_WITH_TASKFLOW
  return interop_pool().num_workers();
#else
  return torch::get_num_interop_threads();
#endif
}

template <typename T>
class Future : public torch::CustomClassHolder {
 public:
  Future(std::future<T>&& future) : future_(std::move(future)) {}

  Future() = default;

  T Wait() { return future_.get(); }

 private:
  std::future<T> future_;
};

/**
 * @brief Utilizes at::launch to launch an async task in the interop thread
 * pool. We should not make use of any native CPU torch ops inside the launched
 * task to avoid spawning a new OpenMP threadpool on each interop thread.
 */
template <typename F>
inline auto async(F&& function) {
  using T = decltype(function());
#ifdef BUILD_WITH_TASKFLOW
  auto future = interop_pool().async(std::move(function));
#else
  auto promise = std::make_shared<std::promise<T>>();
  auto future = promise->get_future();
  at::launch([promise, func = std::move(function)]() {
    if constexpr (std::is_void_v<T>) {
      func();
      promise->set_value();
    } else
      promise->set_value(func());
  });
#endif
  return c10::make_intrusive<Future<T>>(std::move(future));
}

template <ThreadPool pool_type, typename F>
inline void _parallel_for(
    const int64_t begin, const int64_t end, const int64_t grain_size,
    const F& f) {
  if (begin >= end) return;
  int64_t num_threads = get_num_threads();
  const auto num_iter = end - begin;
  const bool use_parallel =
      (num_iter > grain_size && num_iter > 1 && num_threads > 1);
  if (!use_parallel) {
    f(begin, end);
    return;
  }
  if (grain_size > 0) {
    num_threads = std::min(num_threads, at::divup(end - begin, grain_size));
  }
  int64_t chunk_size = at::divup((end - begin), num_threads);
#ifdef BUILD_WITH_TASKFLOW
  tf::Taskflow flow;
  flow.for_each_index(int64_t{0}, num_threads, int64_t{1}, [=](int64_t tid) {
    const int64_t begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
      const int64_t end_tid = std::min(end, begin_tid + chunk_size);
      f(begin_tid, end_tid);
    }
  });
  _get_thread_pool<pool_type>().run(flow).get();
#else
  std::promise<void> promise;
  std::future<void> future;
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;
  int num_launched = 0;
  std::atomic<int> num_finished = 0;
  for (int tid = num_threads - 1; tid >= 0; tid--) {
    const int64_t begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
      const int64_t end_tid = std::min(end, begin_tid + chunk_size);
      if (tid == 0) {
        // Launch the thread 0's work inline.
        f(begin_tid, end_tid);
        continue;
      }
      if (!future.valid()) {
        future = promise.get_future();
        num_launched = tid;
      }
      at::launch([&f, &err_flag, &eptr, &promise, &num_finished, num_launched,
                  begin_tid, end_tid] {
        try {
          f(begin_tid, end_tid);
        } catch (...) {
          if (!err_flag.test_and_set()) {
            eptr = std::current_exception();
          }
        }
        auto ticket = num_finished.fetch_add(1, std::memory_order_release);
        if (1 + ticket == num_launched) {
          // The last thread signals the end of execution.
          promise.set_value();
        }
      });
    }
  }
  // Wait for the launched work to finish.
  if (num_launched > 0) {
    future.get();
    if (eptr) {
      std::rethrow_exception(eptr);
    }
  }
#endif
}

/**
 * @brief GraphBolt's version of torch::parallel_for. Since torch::parallel_for
 * uses OpenMP threadpool, async tasks can not make use of it due to multiple
 * OpenMP threadpools being created for each async thread. Moreover, inside
 * graphbolt::parallel_for, we should not make use of any native CPU torch ops
 * as they will spawn an OpenMP threadpool.
 */
template <typename F>
inline void parallel_for(
    const int64_t begin, const int64_t end, const int64_t grain_size,
    const F& f) {
  _parallel_for<ThreadPool::intraop>(begin, end, grain_size, f);
}

template <typename F>
inline void parallel_for_interop(
    const int64_t begin, const int64_t end, const int64_t grain_size,
    const F& f) {
  _parallel_for<ThreadPool::interop>(begin, end, grain_size, f);
}

}  // namespace graphbolt

#endif  // GRAPHBOLT_ASYNC_H_
