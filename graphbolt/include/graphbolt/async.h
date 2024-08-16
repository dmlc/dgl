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

#include <future>
#include <memory>
#include <mutex>
#include <variant>

#ifdef BUILD_WITH_TASKFLOW
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/taskflow.hpp>
#else
#include <atomic>
#include <exception>
#include <type_traits>
#endif

#ifdef GRAPHBOLT_USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/api/include/torch/cuda.h>
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
#ifdef GRAPHBOLT_USE_CUDA
  using T_no_event = std::conditional_t<std::is_void_v<T>, std::monostate, T>;
  using T_with_event = std::conditional_t<
      std::is_void_v<T>, at::cuda::CUDAEvent,
      std::pair<T, at::cuda::CUDAEvent>>;
  using future_type = std::future<std::variant<T_no_event, T_with_event>>;
#else
  using future_type = std::future<T>;
#endif

 public:
#ifdef GRAPHBOLT_USE_CUDA
  using return_type = std::variant<T_no_event, T_with_event>;
#else
  using return_type = T;
#endif

  Future(future_type&& future) : future_(std::move(future)) {}

  Future() = default;

  T Wait() {
#ifdef GRAPHBOLT_USE_CUDA
    auto result = future_.get();
    if constexpr (std::is_void_v<T>) {
      if (std::holds_alternative<T_with_event>(result)) {
        auto&& event = std::get<T_with_event>(result);
        event.block(c10::cuda::getCurrentCUDAStream());
      }
      return;
    } else if (std::holds_alternative<T_with_event>(result)) {
      auto&& [value, event] = std::get<T_with_event>(result);
      event.block(c10::cuda::getCurrentCUDAStream());
      return value;
    } else {
      return std::get<T_no_event>(result);
    }
#else
    return future_.get();
#endif
  }

 private:
  future_type future_;
};

/**
 * @brief Utilizes at::launch to launch an async task in the interop thread
 * pool. We should not make use of any native CPU torch ops inside the launched
 * task to avoid spawning a new OpenMP threadpool on each interop thread.
 */
template <typename F>
inline auto async(F&& function, bool is_cuda = false) {
  using T = decltype(function());
#ifdef GRAPHBOLT_USE_CUDA
  struct c10::StreamData3 stream_data;
  if (is_cuda) {
    stream_data = c10::cuda::getCurrentCUDAStream().pack3();
  }
#endif
  using return_type = typename Future<T>::return_type;
  auto fn = [=, func = std::move(function)]() -> return_type {
#ifdef GRAPHBOLT_USE_CUDA
    // We make sure to use the same CUDA stream as the thread launching the
    // async operation.
    if (is_cuda) {
      auto stream = c10::cuda::CUDAStream::unpack3(
          stream_data.stream_id, stream_data.device_index,
          stream_data.device_type);
      c10::cuda::CUDAStreamGuard guard(stream);
      at::cuda::CUDAEvent event;
      // Might be executed on the GPU so we record an event to be able to
      // synchronize with it later, in case it is executed on an alternative
      // CUDA stream.
      if constexpr (std::is_void_v<T>) {
        func();
        event.record();
        return event;
      } else {
        auto result = func();
        event.record();
        return std::make_pair(std::move(result), std::move(event));
      }
    }
    if constexpr (std::is_void_v<T>) {
      func();
      return std::monostate{};
    } else {
      return func();
    }
#else
    return func();
#endif
  };
#ifdef BUILD_WITH_TASKFLOW
  auto future = interop_pool().async(std::move(fn));
#else
  auto promise = std::make_shared<std::promise<return_type>>();
  auto future = promise->get_future();
  at::launch([promise, func = std::move(fn)]() {
    if constexpr (std::is_void_v<return_type>) {
      func();
      promise->set_value();
    } else
      promise->set_value(func());
  });
#endif
  return c10::make_intrusive<Future<T>>(std::move(future));
}

template <ThreadPool pool_type, bool for_each, typename F>
inline void _parallel_for(
    const int64_t begin, const int64_t end, const int64_t grain_size,
    const F& f) {
  if (begin >= end) return;
  int64_t num_threads = get_num_threads();
  const auto num_iter = end - begin;
  const bool use_parallel =
      (num_iter > grain_size && num_iter > 1 && num_threads > 1);
  if (!use_parallel) {
    if constexpr (for_each) {
      for (int64_t i = begin; i < end; i++) f(i);
    } else {
      f(begin, end);
    }
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
      if constexpr (for_each) {
        for (int64_t i = begin_tid; i < end_tid; i++) f(i);
      } else {
        f(begin_tid, end_tid);
      }
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
        if constexpr (for_each) {
          for (int64_t i = begin_tid; i < end_tid; i++) f(i);
        } else {
          f(begin_tid, end_tid);
        }
        continue;
      }
      if (!future.valid()) {
        future = promise.get_future();
        num_launched = tid;
      }
      at::launch([&f, &err_flag, &eptr, &promise, &num_finished, num_launched,
                  begin_tid, end_tid] {
        try {
          if constexpr (for_each) {
            for (int64_t i = begin_tid; i < end_tid; i++) f(i);
          } else {
            f(begin_tid, end_tid);
          }
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
  _parallel_for<ThreadPool::intraop, false>(begin, end, grain_size, f);
}

/**
 * @brief Compared to parallel_for, it expects the passed function to take a
 * single argument for each iteration.
 */
template <typename F>
inline void parallel_for_each(
    const int64_t begin, const int64_t end, const int64_t grain_size,
    const F& f) {
  _parallel_for<ThreadPool::intraop, true>(begin, end, grain_size, f);
}

/**
 * @brief Same as parallel_for but uses the interop thread pool.
 */
template <typename F>
inline void parallel_for_interop(
    const int64_t begin, const int64_t end, const int64_t grain_size,
    const F& f) {
  _parallel_for<ThreadPool::interop, false>(begin, end, grain_size, f);
}

/**
 * @brief Compared to parallel_for_interop, it expects the passed function to
 * take a single argument for each iteration.
 */
template <typename F>
inline void parallel_for_each_interop(
    const int64_t begin, const int64_t end, const int64_t grain_size,
    const F& f) {
  _parallel_for<ThreadPool::interop, true>(begin, end, grain_size, f);
}

}  // namespace graphbolt

#endif  // GRAPHBOLT_ASYNC_H_
