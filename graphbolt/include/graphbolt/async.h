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
#include <type_traits>

namespace graphbolt {

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
inline auto async(F function) {
  using T = decltype(function());
  auto promise = std::make_shared<std::promise<T>>();
  auto future = promise->get_future();
  at::launch([=]() {
    if constexpr (std::is_void_v<T>) {
      function();
      promise->set_value();
    } else
      promise->set_value(function());
  });
  return c10::make_intrusive<Future<T>>(std::move(future));
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
  if (begin >= end) return;
  std::promise<void> promise;
  std::future<void> future;
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;

  int64_t num_threads = torch::get_num_threads();
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
  int num_launched = 0;
  std::atomic<int> num_finished = 0;
  for (int tid = num_threads - 1; tid >= 0; tid--) {
    const int64_t begin_tid = begin + tid * chunk_size;
    const int64_t end_tid = std::min(end, begin_tid + chunk_size);
    if (begin_tid < end) {
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
}

}  // namespace graphbolt

#endif  // GRAPHBOLT_ASYNC_H_
